# Deep Dive: vLLM v1 Continuous Batching Scheduler

**Target**: Understand vLLM's request-gathering + batch-formation patterns for VLA-specific scaling (8-16 concurrent robots on A100).

**Date**: April 2026  
**Research Time**: ~20 minutes  
**Source**: vLLM v1 reference codebase in `/reference/vllm/`

---

## 1. Request Gathering & Batch Formation Algorithm

### Request Queue Structure
- **Location**: `vllm/v1/core/sched/scheduler.py:165-169` (v1 scheduler init)
- **Queues**:
  - `self.waiting`: Primary request queue (created via `create_request_queue(self.policy)`)
  - `self.skipped_waiting`: Secondary queue for requests blocked by async dependencies (KV loading, streaming input, grammar)
  - `self.running`: List of currently-executing requests `self.running: list[Request] = []`

### Batch Formation Loop (Token Budget Model)
- **Location**: `vllm/v1/core/sched/scheduler.py:348-954` (the `schedule()` method)
- **Key Algorithm**:
  ```
  token_budget = self.max_num_scheduled_tokens  [line 367]
  
  while token_budget > 0 and more requests exist:
      1. Schedule RUNNING requests first [lines 385-506]
         - Try to allocate `num_new_tokens` from token_budget
         - If KV cache allocation fails, preempt lowest-priority request [lines 474-507]
      
      2. Then schedule WAITING requests [lines 507-855]
         - Pop from waiting/skipped_waiting queue
         - Try to allocate slots in KV cache
         - Advance to RUNNING list
         - Deduct scheduled tokens from token_budget [line 834]
  ```

- **Token Budget Definition** (line 106-109):
  ```python
  self.max_num_scheduled_tokens = (
      self.scheduler_config.max_num_scheduled_tokens
      if self.scheduler_config.max_num_scheduled_tokens
      else self.scheduler_config.max_num_batched_tokens
  )
  ```
  - Configurable per scheduler_config (no hardcoded default in v1)
  - Determines max tokens processed per iteration, not max batch count

### Batch Release Decision
- **No explicit "wait for batch size" logic** — token budget is the only constraint
- **Preemption Strategy** (priority-based, line 475-507):
  - If new request needs blocks but cache is full:
    - Find lowest-priority running request: `max(self.running, key=lambda r: (r.priority, r.arrival_time))`
    - Remove from batch, free KV cache, put back in waiting queue
  - This allows higher-priority requests to preempt lower ones mid-batch
- **No artificial wait-for-max-batch logic** — scheduling happens immediately each step

### Engine Loop Tick Rate
- **Location**: `vllm/v1/engine/core.py:1160-1167` (run_busy_loop) & `vllm/v1/engine/core.py:1170-1200` (_process_input_queue)
- **Pattern**:
  ```python
  while _handle_shutdown():
      _process_input_queue()  # Poll for new requests (blocking or non-blocking)
      _process_engine_step()  # Call scheduler.schedule() + model.forward()
  ```
- **Input Queue Polling**:
  - `block = self.process_input_queue_block` (settable, default True)
  - `req = input_queue.get(block=block)` — blocking GET with optional timeout
  - Non-blocking drain loop: `while not input_queue.empty(): get_nowait()`
  - **No configurable max-wait-time** in core loop; all control is in the input queue timeout

### Max Batch Constraints
- **Location**: `vllm/v1/core/sched/scheduler.py:105` (max_num_running_reqs)
  ```python
  self.max_num_running_reqs = self.scheduler_config.max_num_seqs
  ```
- **Assertion Check** (line 862):
  ```python
  assert len(self.running) <= self.max_num_running_reqs
  ```
- **Configured via `scheduler_config.max_num_seqs`**, not auto-derived from batch size

### Request Addition
- **Location**: `vllm/v1/core/sched/scheduler.py:1737-1757` (add_request)
- **Pattern**:
  - If request_id already exists: treat as streaming continuation, append to `streaming_queue`
  - Else: add to waiting queue via `_enqueue_waiting_request(request)` [line 1754]
  - No blocking; caller gets immediate acknowledgment

---

## 2. Backpressure & Queue Management

### Queue Depth & Admission Control
- **No explicit max queue depth** — waiting queue can grow unbounded
- **Backpressure source**: KV cache allocation failure (line 765-772)
  ```python
  new_blocks = self.kv_cache_manager.allocate_slots(...)
  if new_blocks is None:
      # Request cannot be scheduled (cache full)
      break  # Stop trying to schedule more requests
  ```
- **Result**: Request stays in waiting queue; not added to running

### Preemption (Cache Pressure Response)
- **Trigger**: `allocate_slots()` fails AND policy == PRIORITY (line 475)
- **Action** (lines 476-507):
  1. Find worst priority request in running list
  2. Preempt it (free KV blocks, move to waiting, increment preemption counter)
  3. Retry current request's allocation
  4. If still fails → break, don't schedule current request

- **Preemption Cost**:
  - Token budget is refunded: `token_budget += num_scheduled_tokens.pop(preempted_req_id)` [line 484]
  - Request must start over from beginning (num_computed_tokens = 0)
  - Request.num_preemptions incremented (line 976)

### Streaming Input Backpressure
- **Location**: `vllm/v1/core/sched/scheduler.py:1593-1604` (streaming queue handling)
- **Blocked State**: `RequestStatus.WAITING_FOR_STREAMING_REQ` [line 1600]
- **Counter**: `self.num_waiting_for_streaming_input` [line 179]
- **Unblocks**: When next streaming chunk arrives or stream finishes

### KV Connector (Remote KV Load) Backpressure
- **Blocked State**: `RequestStatus.WAITING_FOR_REMOTE_KVS` [line 798]
- **Logic**: When loading KV cache asynchronously from remote store
- **Unblocked**: When connector signals `finished_recving_kv_req_ids` (line 2123)
- **Impact**: Request held in `skipped_waiting` queue, not tried in scheduler loop

### Encoder Cache Backpressure
- **Location**: `vllm/v1/core/sched/scheduler.py:1112-1273` (_try_schedule_encoder_inputs)
- **Trigger**: `encoder_cache_manager.can_allocate()` returns False [line 1218]
- **Response**: Reduce `num_new_tokens` to skip encoder input scheduling, or set to 0 (no schedule)
- **Budget**: `encoder_compute_budget` separate from token budget (line 703)

---

## 3. Metrics & Observability

### Scheduler Stats
- **Location**: `vllm/v1/core/sched/scheduler.py:1935-1971` (make_stats)
- **Exposed Metrics**:
  ```python
  SchedulerStats(
      num_running_reqs=len(self.running),           # Currently executing
      num_waiting_reqs=len(self.waiting),           # Unblocked waiting
      num_skipped_waiting_reqs=len(self.skipped_waiting),  # Blocked waiting
      kv_cache_usage=...,                           # Cache fill %
      prefix_cache_stats=...,                       # Hit/miss rates
      spec_decoding_stats=...,                      # Token acceptance %
  )
  ```
- **Sent to Client**: Once per step (line 1558)

### Request Count Query
- **Location**: `vllm/v1/core/sched/scheduler.py:1733-1735` (get_request_counts)
  ```python
  def get_request_counts(self) -> tuple[int, int]:
      return len(self.running), len(self.waiting) + len(self.skipped_waiting)
  ```
- Used by engine to determine if there's work to do (has_work)

### Per-Request Tracking
- **Location**: `vllm/v1/core/sched/output.py` (SchedulerOutput dataclass)
- **Exposed per-batch**:
  - `num_scheduled_tokens: dict[str, int]` — tokens scheduled per request [line 191]
  - `total_num_scheduled_tokens: int` — sum of above [line 194]
  - `scheduled_spec_decode_tokens: dict[str, list[int]]` — draft tokens scheduled [line 198]
  - `preempted_req_ids: set[str]` — requests preempted this step [line 217]
  - `finished_req_ids: set[str]` — requests finished this step [line 210]

### No Explicit Batch Metrics
- **NOT exposed**: Average batch fill (how many requests per batch)
- **NOT exposed**: Queue wait time (time from arrival to scheduling)
- **NOT exposed**: Preemption rate or cost
- These would require adding custom instrumentation

---

## 4. Patterns That Transfer to VLA Batching

### Core Transferable Concepts

#### 1. **Token Budget Model** ✓
- **Status**: DIRECTLY APPLICABLE
- **Current Reflex**: Naive timeout-based batching (gather requests for `batch_timeout_ms`, max `max_batch`)
- **vLLM Pattern**: Token budget + priority-based scheduling
- **VLA Adaptation**:
  ```
  action_chunk_budget = max_chunks_per_batch
  
  while chunk_budget > 0 and requests_waiting:
      if try_schedule_request(chunk_budget):
          chunk_budget -= chunks_needed
      else:
          break
  ```
- **Benefit**: Fills batch to GPU capacity, not arbitrary request count
- **Reflex Impact**: Replace `max_batch=N` with `max_chunks=C` where `C ≈ episode_duration / num_robots`
- **Ref File**: `vllm/v1/core/sched/scheduler.py:367-834` (the core loop)

#### 2. **Priority-Based Preemption** ✓
- **Status**: USEFUL FOR MULTI-ROBOT (but different priority model)
- **vLLM Use**: Preempt low-priority requests to admit high-priority ones
- **VLA Adaptation**:
  - Could weight robots by urgency (e.g., collision avoidance > normal manipulation)
  - Or fairness-based: preempt requests that've had longest compute time
  - Or deadline-based: preempt requests whose deadline will miss if not preempted now
- **Benefit**: Graceful degradation under load (don't drop requests, delay them)
- **Ref File**: `vllm/v1/core/sched/scheduler.py:475-507` (preemption logic)

#### 3. **Request Queue Separation** ✓
- **Status**: USEFUL FOR SPECIAL CASES
- **Pattern**: Separate queues for blocked (waiting for KV) vs. ready (can schedule now)
- **VLA Adaptation**:
  - Keep waiting list for requests awaiting sensor data
  - Keep skipped list for requests that just got preempted
  - Scheduler tries ready-list first, then skipped-list
- **Benefit**: Avoids retry overhead on permanently-blocked requests
- **Ref File**: `vllm/v1/core/sched/scheduler.py:166-168, 1576-1586` (queue selection)

#### 4. **Unblocking Mechanisms** ✓
- **Status**: DIRECTLY APPLICABLE FOR STREAMING ROBOT DATA
- **Pattern**:
  - `RequestStatus.WAITING_FOR_REMOTE_KVS` — waiting for async KV load
  - `RequestStatus.WAITING_FOR_STREAMING_REQ` — waiting for next input chunk
- **VLA Adaptation**:
  - `WAITING_FOR_SENSOR_DATA` — waiting for next camera frame
  - `WAITING_FOR_STATE_UPDATE` — waiting for next proprioceptive data
  - Unblock when new sensor data arrives
- **Benefit**: Batch formation respects data arrival times (don't batch stale requests)
- **Ref Files**: 
  - `vllm/v1/core/sched/scheduler.py:1593-1604` (streaming queue)
  - `vllm/v1/core/sched/scheduler.py:2070-2101` (promotion logic)

---

### Patterns to SKIP for VLA

#### ✗ Token-Level Paging
- vLLM's `paged_attention` + block allocation is for tokens
- VLA has fixed-size action chunks (e.g., 50 timesteps)
- Chunk granularity is already coarse; don't implement block-level tracking

#### ✗ Speculative Decoding
- vLLM schedules draft tokens separately
- VLA has no "draft" mode; all chunks are fixed-size
- Skip `scheduled_spec_decode_tokens` and grammar-based validation

#### ✗ Chunked Prefill
- vLLM can split prompt across multiple batches
- VLA input is a single image + instruction (small context)
- Not a bottleneck; keep prompts whole

#### ✗ Pipeline Parallelism
- vLLM uses `num_scheduled_tokens - len(spec_tokens)` to handle P/D disaggregation
- VLA on single A100 has no pipeline parallelism
- Skip KV connector and async KV loading infrastructure

---

## 5. Delta vs. Reflex VLA Current Batching (`src/reflex/runtime/server.py`)

### Current Implementation (Lines 771-866)
```python
async def _batch_worker_loop(self) -> None:
    while True:
        batch: list[tuple] = []
        first = await self._batch_queue.get()  # Block on first request
        batch.append(first)
        
        # Drain up to max_batch within timeout window
        deadline = loop.time() + self._batch_timeout_s
        while len(batch) < self._max_batch:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(
                    self._batch_queue.get(), timeout=remaining
                )
                batch.append(item)
            except asyncio.TimeoutError:
                break
        
        # Run batched inference (sync)
        results = self._predict_batch_sync(batch)
```

### Issues & vLLM Improvements

| Aspect | Current (Reflex) | vLLM v1 | VLA Gain |
|--------|------------------|---------|----------|
| **Batch Size Control** | `max_batch` (request count) | `token_budget` (work units) | Batch by actual GPU load, not request count |
| **Wait Time** | `batch_timeout_ms` (fixed) | Input queue polling + eager scheduling | Tighter latency; schedule as soon as ready |
| **Preemption** | None (drop or wait) | Priority-based, with budget refund | Graceful degradation; don't lose work |
| **Queue Separation** | Single queue | waiting + skipped_waiting | Faster dequeue for ready requests |
| **Backpressure** | Queue fills indefinitely | KV cache allocation failure | Explicit "too much work" signal |
| **Observability** | `_batches_run`, `_batched_requests` | Full stats (waiting depth, cache %, preemptions) | Debug batch formation bottlenecks |
| **Blocking Requests** | No concept | WAITING_FOR_REMOTE_KVS, etc. | Unblock when sensor data arrives |

### Recommended Reflex Changes (Prioritized)

#### **Priority 1: Chunk Budget Model**
- Replace `max_batch=N` with `max_chunks_per_batch=C`
- `C ≈ 50-100` (1-2 episodes per batch on A100)
- Track `episode_length` from request metadata
- Deduct from budget as requests are scheduled

**File to modify**: `src/reflex/runtime/server.py:771-866` (_batch_worker_loop)  
**Effort**: Moderate (1-2 hours)  
**Impact**: +10-20% throughput (better GPU utilization)

#### **Priority 2: Queue Separation**
- Add `ready_queue` (can schedule now) vs. `waiting_queue` (need sensor data)
- Mark requests with `metadata.awaiting_sensor_data = False` after frame arrives
- Scheduler tries ready_queue first

**File to modify**: `src/reflex/runtime/server.py:102-111` (queue init)  
**Effort**: Low (30 mins)  
**Impact**: +5% throughput (avoid re-attempting blocked requests)

#### **Priority 3: Basic Preemption**
- When GPU is full, preempt oldest running request (FCFS)
- Refund its chunks to budget
- Put back in queue

**File to modify**: `src/reflex/runtime/server.py:858-861` (_predict_batch_sync)  
**Effort**: High (4+ hours; needs proper state tracking)  
**Impact**: +5-10% latency improvement (some requests get faster service)

#### **Priority 4: Observability**
- Track queue depth, wait time, preemptions
- Log per-batch: `num_scheduled, chunks_used, preemptions`
- Export to Prometheus/datadog

**File to modify**: `src/reflex/runtime/server.py:825-866` (_batch_worker_loop)  
**Effort**: Low (1 hour)  
**Impact**: 0% performance; enables debugging

---

## 6. Exact File & Line References

### vLLM Scheduler Core
| Concept | File | Lines | Key Code |
|---------|------|-------|----------|
| Scheduler class init | `vllm/v1/core/sched/scheduler.py` | 67-180 | Queue creation, max_num_running_reqs, token_budget |
| schedule() method | `vllm/v1/core/sched/scheduler.py` | 348-954 | Main loop: RUNNING → WAITING → allocate → batch |
| Running requests loop | `vllm/v1/core/sched/scheduler.py` | 385-506 | while req_index < len(self.running): try_schedule() |
| Preemption logic | `vllm/v1/core/sched/scheduler.py` | 475-507 | max(self.running, priority); preempt_request() |
| Waiting requests | `vllm/v1/core/sched/scheduler.py` | 507-855 | Queue selection + allocation loop |
| Request addition | `vllm/v1/core/sched/scheduler.py` | 1737-1757 | add_request(); _enqueue_waiting_request() |
| Blocked state promotions | `vllm/v1/core/sched/scheduler.py` | 2070-2101 | _try_promote_blocked_waiting_request() |
| Engine busy loop | `vllm/v1/engine/core.py` | 1160-1219 | run_busy_loop(); _process_input_queue(); step() |
| Input queue polling | `vllm/v1/engine/core.py` | 1170-1200 | _process_input_queue(); blocking/non-blocking get |
| SchedulerOutput | `vllm/v1/core/sched/output.py` | 179-253 | num_scheduled_tokens, preempted_req_ids, etc. |
| SchedulerStats | `vllm/v1/core/sched/scheduler.py` | 1935-1971 | make_stats(); num_running, num_waiting, cache_usage |

### Reflex Batching Current
| Concept | File | Lines | Key Code |
|---------|------|-------|----------|
| Batch worker init | `src/reflex/runtime/server.py` | 103-111 | _max_batch, _batch_timeout_s |
| Batch worker start | `src/reflex/runtime/server.py` | 771-788 | start_batch_worker(); create asyncio queue |
| Main loop | `src/reflex/runtime/server.py` | 825-866 | _batch_worker_loop(); drain queue with timeout |
| Batch sync inference | `src/reflex/runtime/server.py` | 868-942 | _predict_batch_sync(); run ONNX, split results |

---

## Summary: What Transfers, What Doesn't

### **TRANSFER THESE** (high ROI)
1. **Token/Chunk budget model** — scale batch by work, not request count (✓ fits VLA)
2. **Queue separation** — ready vs. blocked requests (✓ fits sensor streaming)
3. **Basic preemption** — graceful degradation when overloaded (✓ fits multi-robot)
4. **Unblocking signals** — WAITING_FOR_DATA status + promotion (✓ fits sensor sync)
5. **Batch-level stats** — queue depth, preemptions, cache usage (✓ enabler for debugging)

### **SKIP THESE** (low ROI or wrong model)
1. **Token-level paging** — vLLM's paged_attention (✗ overkill for 50-token chunks)
2. **Speculative decoding** — draft tokens + grammar validation (✗ no speculation in VLA)
3. **Pipeline parallelism** — P/D disaggregation + async KV (✗ single A100, no pipelining)
4. **Chunked prefill** — split prompts across batches (✗ small context, not bottleneck)

### **DESIGN PRINCIPLES** (from vLLM)
- **Schedule greedily**: Don't wait for full batch; go as soon as ready
- **Preemption over rejection**: Keep requests, delay them instead of losing them
- **Budget-driven**: Constrain by actual GPU work (tokens/chunks), not arbitrary counts
- **Observable**: Expose queue depth, preemptions, cache usage in stats
- **Unblock on data arrival**: Don't retry permanently-blocked requests

---

## Appendix: vLLM Scheduling Algorithm Pseudocode

```python
def schedule() -> SchedulerOutput:
    token_budget = max_num_scheduled_tokens
    scheduled_running = []
    scheduled_waiting = []
    preempted = []
    
    # Phase 1: Schedule RUNNING requests (highest priority)
    for request in running:
        if token_budget <= 0:
            break
        
        num_new_tokens = compute_new_tokens(request, token_budget)
        if num_new_tokens == 0:
            continue  # Skip this request, try next
        
        # Try to allocate KV cache blocks
        new_blocks = kv_cache_manager.allocate_slots(request, num_new_tokens)
        
        if new_blocks is None:
            # Cache full; try preemption
            if policy == PRIORITY:
                victim = max(running, key=lambda r: (r.priority, r.arrival_time))
                preempt_request(victim)  # Free blocks, return to waiting
                preempted.append(victim)
                
                # Retry current request
                new_blocks = kv_cache_manager.allocate_slots(request, num_new_tokens)
            
            if new_blocks is None:
                # Still can't fit; skip this request and all future ones
                break
        
        # Successfully scheduled
        scheduled_running.append(request)
        token_budget -= num_new_tokens
    
    # Phase 2: Schedule WAITING requests (if budget remains)
    while token_budget > 0:
        request_queue = select_queue(waiting, skipped_waiting)
        if not request_queue:
            break
        
        request = request_queue.pop()
        num_new_tokens = min(compute_new_tokens(request), token_budget)
        
        # Try to allocate
        new_blocks = kv_cache_manager.allocate_slots(request, num_new_tokens)
        if new_blocks is None:
            # Can't schedule; put back (will try again next iteration)
            request_queue.prepend(request)
            break
        
        # Successfully moved to running
        running.append(request)
        scheduled_waiting.append(request)
        token_budget -= num_new_tokens
    
    return SchedulerOutput(
        scheduled_new_reqs=scheduled_waiting,
        scheduled_cached_reqs=scheduled_running,
        num_scheduled_tokens={...},
        preempted_req_ids={r.id for r in preempted},
    )
```

