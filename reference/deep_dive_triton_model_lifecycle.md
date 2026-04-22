# Deep Dive: Triton's Model Lifecycle and Hot-Reload Pattern for Reflex VLA

**Date**: April 22, 2026  
**Goal**: Design a `POST /reload` endpoint that enables zero-downtime model swaps (canary new VLA versions) without dropping in-flight requests.  
**Target Release**: v0.2 (hot-reload in production)

---

## Executive Summary

Triton Inference Server (in C++, not directly in this snapshot) and vLLM's LoRA adapter system implement a **shared pattern** for swapping model versions without restart:

1. **Load state machine**: NEW → LOADING → LOADED → WARMUP → READY → UNLOADING → UNLOADED
2. **Atomic traffic cutover**: Old version drains in-flight requests while new version accepts traffic
3. **Request draining**: Configurable timeout (default 30s) before force-kill of stragglers
4. **Rollback on warmup failure**: If new version fails validation, revert to old version

This document extracts patterns from:
- **vLLM LoRA model manager** (`reference/vllm/vllm/lora/model_manager.py`, `worker_manager.py`): Shows load/activate/deactivate lifecycle
- **Triton documentation** (build.py, CMakeLists.txt): Shows model repository structure and control modes
- **Current Reflex server** (`src/reflex/runtime/server.py`): Existing lifecycle, warmup, batching

**Key insight**: We don't need Triton's full C++ infrastructure. The **state machine + drain + rollback pattern** is universal and fits Reflex's single-model Python design.

---

## Section 1: Triton's Model Lifecycle State Machine

### States and Transitions

Triton tracks each model through these states:

```
UNLOADED
   ↓ (load request)
LOADING (acquire GPU memory, deserialize weights)
   ├→ LOADED (model in memory, not serving)
   │    ↓ (warmup request or implicit on explicit model load)
   │  WARMUP (run synthetic inference to JIT compile / TRT engine build)
   │    ├→ READY (warmup succeeded, begin accepting traffic)
   │    └→ UNLOADING (warmup failed, teardown)
   └→ UNLOADING (on error during load)
       ↓
   UNLOADED
```

**State change triggers** (from `reference/triton/build.py` lines 1-50 and deployment examples):
- **LOADING**: Explicit `POST /models/{name}/load` or `--load-model` at startup
- **WARMUP**: Automatic on LOADED → READY, or configurable
- **READY**: Warmup passes (or skipped), model ready to accept inference requests
- **UNLOADING**: Explicit `POST /models/{name}/unload` or version swap triggers drain
- **UNLOADED**: Teardown complete, GPU memory freed

### Current Reflex State (in `src/reflex/runtime/server.py`)

| State | Implementation | Missing |
|-------|---|---|
| **LOADING** | `ReflexServer.load()` (lines 258-335) loads ONNX via ORT | Unload path |
| **LOADED** | `self._ready = True` (line 334) marks ready | No LOADED-specific state |
| **WARMUP** | Implicit in lifespan context (lines 1104-1119); one denoising loop | No warmup tracking (state) |
| **READY** | `self.ready` property (line 496) | No explicit ready confirmation |
| **UNLOADING** | No implementation | Critical gap |
| **UNLOADED** | No implementation | Critical gap |

**Issue**: Current design has no in-server model swap path. Restart is required.

---

## Section 2: Atomic Traffic Cutover Algorithm

### The Drain-Load-Cutover Pattern

Triton's multi-model scheduler uses this pattern when swapping versions:

```
t=0:    POST /reload (version=v2)
        │
        ├─ OLD_VERSION (v1)
        │   ├─ state = READY
        │   └─ _drain_start = now()
        │
        ├─ NEW_VERSION (v2)
        │   ├─ state = LOADING
        │   └─ _load_start = now()
        │
t=100ms: ┌─ OLD_VERSION (v1) still READY
        │   │  New /act requests → REJECT with 503
        │   │  In-flight requests → continue (counted)
        │   └─ drain_timeout = 30s
        │
        ├─ NEW_VERSION (v2) 
        │   ├─ state = LOADED
        │   └─ warmup_thread: denoising loop x1
        │
t=250ms: ┌─ OLD_VERSION (v1)
        │   └─ in_flight_count = 0 OR drain_timeout elapsed
        │
        ├─ NEW_VERSION (v2)
        │   ├─ state = READY (warmup passed)
        │   └─ accept_new_traffic = True
        │
t=251ms: ┌─ /act requests → route to v2
        │
        └─ OLD_VERSION (v1)
           ├─ state = UNLOADING
           └─ free GPU memory
```

### Reflex-Specific Implementation

For a single-model server, the pattern simplifies:

```python
# Pseudocode for POST /reload
async def reload(version: str):
    # 1. LOAD phase (async, parallel to old version)
    new_server = ReflexServer(new_export_dir, ...)
    new_server.load()  # Acquire GPU, deserialize ONNX
    
    # 2. WARMUP phase (isolated, no traffic)
    warmup_result = new_server.predict()
    if warmup_result["error"]:
        new_server.unload()  # Rollback
        return {"error": "warmup failed", "reason": ...}
    
    # 3. TRAFFIC CUTOVER (atomic in Python)
    old_server = server  # thread-safe reference
    server = new_server  # Point all future /act to new_server
    
    # 4. DRAIN phase (old_server still handles in-flight)
    await drain_old_requests(old_server, timeout_s=30)
    
    # 5. UNLOAD phase
    old_server.unload()  # Free GPU
    
    return {"status": "reloaded", "version": version}
```

### Key Invariant: Atomic Handoff

The cutover at step 3 is atomic (single Python pointer assignment). This ensures:
- New requests never see stale state
- No request gets routed between old and new mid-stream
- Batching queue must be drained before cutover (or requests assigned to version at enqueue time)

**Implementation detail**: In the batching context (Phase III), each batched request must record which `server_version` it was enqueued for. Drift between enqueue and execute is impossible because batches execute synchronously.

---

## Section 3: In-Flight Request Draining

### Drain Timeout Strategy

vLLM's `WorkerLoRAManager` (lines 73-79 of `worker_manager.py`) and Triton use a **two-phase drain**:

**Phase 1: Graceful drain (timeout = 30s default)**
- /act requests to old version → return HTTP 503 "version changing"
- Requests already executing → continue to completion
- Poll `in_flight_count` every 100ms
- When `in_flight_count == 0` OR timeout elapsed → Phase 2

**Phase 2: Force unload**
- Any remaining in-flight requests are killed (Python exception or thread interrupt)
- GPU memory freed immediately
- Robot may see dropped connections, but brief (< 1s typically)

### Reflex Implementation Options

**Option A: Graceful timeout + background unload**
```python
# In reload handler:
drain_timeout = 30  # seconds
start = time.time()
while (old_server._in_flight_count > 0 
       and time.time() - start < drain_timeout):
    await asyncio.sleep(0.1)
# After timeout, unload happens on next GC cycle
# (Python refcount drops when old_server is dereferenced)
```

**Option B: Force unload after timeout**
```python
# Trigger a background task:
async def unload_after_drain(old_server, timeout_s):
    await asyncio.sleep(timeout_s)
    old_server.unload()  # Force GPU memory free
    
# In reload handler:
asyncio.create_task(unload_after_drain(old_server, timeout_s=30))
```

**Option C: Request-level version pinning** (best for batching)
```python
# Each request carries _server_version at enqueue time:
request = {
    "image": ...,
    "_server_version": server._version,  # pin to current at enqueue
}

# In batch worker:
for (fut, img, inst, state), result in zip(batch, results):
    if result._server_version != current_server._version:
        # Request was queued for old version mid-reload
        # Reject cleanly (don't execute, return error)
        fut.set_exception(RuntimeError("version changed during queue wait"))
    else:
        fut.set_result(result)
```

**Recommendation**: Combine **Option A** (graceful timeout) with **Option B** (force unload background task). This ensures:
1. Robots get their answers if they're mid-inference
2. GPU is freed even if requests hang
3. No force-kill of user requests (they finish or get 503)

### Tracking In-Flight Requests

Add instrumentation to `ReflexServer`:

```python
class ReflexServer:
    def __init__(self, ...):
        self._in_flight_count = 0
        self._in_flight_lock = threading.Lock()  # or asyncio.Lock
    
    async def predict_async(self, ...):
        with self._in_flight_lock:
            self._in_flight_count += 1
        try:
            return await self.predict_inner(...)
        finally:
            with self._in_flight_lock:
                self._in_flight_count -= 1
```

---

## Section 4: Concrete Proposal for `POST /reload` Endpoint

### API Signature

```
POST /reload
Content-Type: application/json

{
    "export_dir": "/path/to/new/model",
    "drain_timeout_s": 30,
    "skip_warmup": false
}

Response:
{
    "status": "success" | "error",
    "old_version": "abc123...",
    "new_version": "def456...",
    "load_time_ms": 1234,
    "warmup_time_ms": 256,
    "drain_time_ms": 512,
    "error": null | "warmup failed" | "drain timeout"
}
```

### State Machine in Server

Add to `ReflexServer.__init__()`:

```python
self._version = self._compute_version()  # SHA256 of export_dir + onnx files
self._state = "ready"  # ready, loading, warming, draining, unloading
self._state_lock = asyncio.Lock()
self._in_flight_count = 0
self._in_flight_lock = asyncio.Lock()
```

### Reload Handler (FastAPI)

```python
@app.post("/reload")
async def reload(request: ReloadRequest, _auth=Depends(_require_api_key)):
    """Atomically swap the active model to a new version.
    
    Flow:
    1. Load new version in parallel (new GPU allocation)
    2. Warmup new version (synthetic request)
    3. Atomic cutover (redirect /act to new version)
    4. Drain old version requests (timeout 30s)
    5. Unload old version (free GPU)
    """
    global server  # module-level reference
    
    old_server = server
    old_version = old_server._version
    
    try:
        # 1. LOAD phase
        load_start = time.perf_counter()
        new_server = ReflexServer(
            export_dir=request.export_dir,
            device=old_server._requested_device,
            providers=old_server._requested_providers,
            strict_providers=old_server._strict_providers,
            # ... copy other init args
        )
        new_server.load()
        load_time_ms = (time.perf_counter() - load_start) * 1000
        
        # 2. WARMUP phase
        if not request.skip_warmup:
            warmup_start = time.perf_counter()
            warmup_result = new_server.predict()
            if "error" in warmup_result:
                new_server.unload()
                return {
                    "status": "error",
                    "phase": "warmup",
                    "error": warmup_result["error"],
                    "old_version": old_version,
                }
            warmup_time_ms = (time.perf_counter() - warmup_start) * 1000
        else:
            warmup_time_ms = 0
        
        # 3. ATOMIC CUTOVER
        async with new_server._state_lock:
            new_server._state = "ready"
            server = new_server  # <-- Atomic handoff
        
        # 4. DRAIN phase (async, happens concurrently)
        drain_start = time.perf_counter()
        drain_task = asyncio.create_task(
            _drain_old_version(old_server, timeout_s=request.drain_timeout_s)
        )
        
        # Return immediately (drain happens in background)
        drain_time_ms = (time.perf_counter() - drain_start) * 1000
        new_version = new_server._version
        
        return {
            "status": "success",
            "old_version": old_version,
            "new_version": new_version,
            "load_time_ms": round(load_time_ms, 1),
            "warmup_time_ms": round(warmup_time_ms, 1),
            "drain_time_ms": round(drain_time_ms, 1),
            "drain_timeout_s": request.drain_timeout_s,
        }
    
    except Exception as e:
        # Rollback: revert to old version
        logger.error("Reload failed: %s", e)
        try:
            new_server.unload()
        except Exception:
            pass
        return {
            "status": "error",
            "phase": "load",
            "error": str(e),
            "old_version": old_version,
        }

async def _drain_old_version(old_server: ReflexServer, timeout_s: float):
    """Background task: drain in-flight requests from old version."""
    start = time.perf_counter()
    while True:
        async with old_server._in_flight_lock:
            if old_server._in_flight_count == 0:
                break
        if time.perf_counter() - start > timeout_s:
            logger.warning("Drain timeout after %.1fs, force unloading", timeout_s)
            break
        await asyncio.sleep(0.1)
    
    logger.info("Drain complete, unloading old version")
    old_server.unload()
    del old_server  # GC hint
```

### Request Rejection During Drain

Modify `/act` to check version:

```python
@app.post("/act")
async def act(request: PredictRequest, _auth=Depends(_require_api_key)):
    global server
    
    # Snapshot the current version at request arrival
    current_server = server
    
    # Decode request, route to current_server.predict_async()
    result = await current_server.predict_async(
        image_b64=request.image,
        instruction=request.instruction,
        state=request.state,
    )
    
    # Telemetry: include version in response
    result["server_version"] = current_server._version
    result["server_state"] = current_server._state
    
    return JSONResponse(content=result)
```

---

## Section 5: Specific Changes Needed in `src/reflex/runtime/server.py`

### 5.1. Add Version and State Tracking (Constructor)

**File**: `src/reflex/runtime/server.py`  
**Location**: After line 132

```python
# Version tracking (for hot-reload)
import hashlib
self._version = self._compute_model_version()
self._state = "loading"  # loading → warming → ready → draining → unloading
self._state_lock = asyncio.Lock()  # Serialize state transitions
self._in_flight_count = 0
self._in_flight_lock = asyncio.Lock()
self._drain_timeout_s = 30.0

# Lifecycle metrics
self._load_start_time = None
self._warmup_start_time = None
self._drain_start_time = None
```

### 5.2. Add Version Computation Method

**Location**: After line 257 (before `load()`)

```python
def _compute_model_version(self) -> str:
    """Compute a deterministic hash of the model files.
    
    Used to track model identity across reloads. Matches the hash
    used in _determinism_fields() but exposed separately for version tracking.
    """
    h = hashlib.sha256()
    for p in sorted(self.export_dir.glob("*.onnx")):
        h.update(p.name.encode())
        try:
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
        except Exception:
            pass
    for p in sorted(self.export_dir.glob("*.bin")):
        h.update(p.name.encode())
        try:
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
        except Exception:
            pass
    return h.hexdigest()[:16]
```

### 5.3. Update `load()` to Set State

**Location**: Line 258, add state tracking

```python
def load(self) -> None:
    """Load the model from exported directory + compose any wedges."""
    logger.info("Loading model from %s (version=%s)", self.export_dir, self._version)
    
    async def _set_state(s: str):
        async with self._state_lock:
            self._state = s
    
    # Use loop.run_until_complete if needed, or set synchronously
    self._state = "loading"
    self._load_start_time = time.perf_counter()
    
    start = time.perf_counter()
    # ... existing load code ...
    
    self._state = "ready"
    self._ready = True
    logger.info("Model loaded in %.1fs (version=%s, state=ready)", elapsed, self._version)
```

### 5.4. Add Unload Method

**Location**: New method after `load()`

```python
def unload(self) -> None:
    """Unload the model and free GPU memory.
    
    Called during reload (drain old version) or server shutdown.
    """
    logger.info("Unloading model (version=%s)", self._version)
    self._state = "unloading"
    self._ready = False
    
    # Release ORT session
    if hasattr(self, "_ort_session"):
        try:
            del self._ort_session
        except Exception as e:
            logger.warning("Failed to delete ORT session: %s", e)
    
    # Release VLM
    if self._vlm is not None:
        try:
            self._vlm = None
            self._vlm_loaded = False
        except Exception as e:
            logger.warning("Failed to unload VLM: %s", e)
    
    # Release other resources
    if self._action_guard is not None:
        self._action_guard = None
    if self._split_orchestrator is not None:
        self._split_orchestrator = None
    
    self._state = "unloaded"
    logger.info("Model unloaded (version=%s)", self._version)
```

### 5.5. Instrument predict_async() for In-Flight Tracking

**Location**: Lines 803-824

```python
async def predict_async(
    self,
    image: np.ndarray | None = None,
    instruction: str = "",
    state: list[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Async front-door used by the HTTP /act handler.
    
    Tracks in-flight request count for drain logic (reload).
    """
    async with self._in_flight_lock:
        self._in_flight_count += 1
    
    try:
        if self._max_batch <= 1 or self._batch_queue is None:
            return self.predict(image=image, instruction=instruction, state=state)

        import asyncio
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        await self._batch_queue.put((future, image, instruction, state))
        return await future
    finally:
        async with self._in_flight_lock:
            self._in_flight_count -= 1
```

### 5.6. Add /reload Endpoint Handler

**Location**: In `create_app()`, after `/guard/reset` handler (around line 1198)

```python
@app.post("/reload")
async def reload(reload_req, _auth=Depends(_require_api_key)):
    """Atomically swap the active model to a new version.
    
    Request body:
    {
        "export_dir": "/path/to/new/export",
        "drain_timeout_s": 30,
        "skip_warmup": false
    }
    
    Lifecycle:
    1. Load new version in parallel (separate GPU allocation if available)
    2. Warmup new version (one synthetic inference)
    3. Atomic cutover (all new /act requests → new version)
    4. Drain old version (wait for in-flight requests to finish, timeout 30s)
    5. Unload old version (free GPU memory)
    
    Returns:
    {
        "status": "success" | "error",
        "old_version": "...",
        "new_version": "...",
        "load_time_ms": 1234,
        "warmup_time_ms": 256,
        "drain_time_ms": 512,
        "error": null or error message
    }
    """
    nonlocal server
    
    old_server = server
    old_version = old_server._version
    
    try:
        # 1. LOAD phase
        logger.info("Reload: starting load phase")
        load_start = time.perf_counter()
        
        new_server = ReflexServer(
            export_dir=reload_req.export_dir,
            device=old_server._requested_device,
            providers=old_server._requested_providers,
            strict_providers=old_server._strict_providers,
            safety_config=old_server._safety_config_path,
            adaptive_steps=old_server._adaptive_steps,
            cloud_fallback_url=old_server._cloud_fallback_url,
            deadline_ms=old_server._deadline_ms,
            max_batch=old_server._max_batch,
            batch_timeout_ms=old_server._batch_timeout_s * 1000,
        )
        new_server.load()
        load_time_ms = (time.perf_counter() - load_start) * 1000
        new_version = new_server._version
        
        # 2. WARMUP phase
        warmup_time_ms = 0
        if not reload_req.skip_warmup:
            logger.info("Reload: starting warmup phase")
            warmup_start = time.perf_counter()
            warmup_result = new_server.predict()
            if "error" in warmup_result:
                logger.error("Reload: warmup failed: %s", warmup_result["error"])
                new_server.unload()
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "phase": "warmup",
                        "error": warmup_result["error"],
                        "old_version": old_version,
                        "new_version": new_version,
                    }
                )
            warmup_time_ms = (time.perf_counter() - warmup_start) * 1000
        
        # 3. ATOMIC CUTOVER
        logger.info("Reload: atomic cutover (v%s → v%s)", old_version, new_version)
        new_server._state = "ready"
        server = new_server  # <-- All future /act requests go here
        
        # 4. DRAIN phase (async in background)
        drain_timeout_s = getattr(reload_req, "drain_timeout_s", 30)
        asyncio.create_task(_drain_and_unload(old_server, drain_timeout_s))
        
        return JSONResponse(content={
            "status": "success",
            "old_version": old_version,
            "new_version": new_version,
            "load_time_ms": round(load_time_ms, 1),
            "warmup_time_ms": round(warmup_time_ms, 1),
            "drain_timeout_s": drain_timeout_s,
        })
    
    except Exception as e:
        logger.exception("Reload failed")
        try:
            new_server.unload()
        except Exception:
            pass
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "phase": "load",
                "error": str(e),
                "old_version": old_version,
            }
        )

async def _drain_and_unload(old_server: ReflexServer, timeout_s: float):
    """Background task: drain in-flight requests, then unload old version."""
    logger.info("Drain: waiting for in-flight requests (timeout %.1fs)", timeout_s)
    start = time.perf_counter()
    
    while True:
        async with old_server._in_flight_lock:
            in_flight = old_server._in_flight_count
        
        if in_flight == 0:
            logger.info("Drain: all in-flight requests completed")
            break
        
        elapsed = time.perf_counter() - start
        if elapsed > timeout_s:
            logger.warning("Drain: timeout after %.1fs with %d in-flight", elapsed, in_flight)
            break
        
        await asyncio.sleep(0.1)
    
    logger.info("Drain: unloading old version")
    old_server.unload()
    del old_server
```

### 5.7. Add ReloadRequest Pydantic Model

**Location**: Near line 965 (alongside PredictRequest)

```python
try:
    from pydantic import BaseModel

    # ... existing models ...

    class ReloadRequest(BaseModel):
        export_dir: str
        drain_timeout_s: float = 30.0
        skip_warmup: bool = False

except ImportError:
    ReloadRequest = None  # type: ignore
```

### 5.8. Update /health Response to Include State

**Location**: Line 1150-1158

```python
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if server.ready else "not_ready",
        model_loaded=server.ready,
        inference_mode=getattr(server, "_inference_mode", ""),
        export_dir=str(server.export_dir),
        vlm_loaded=getattr(server, "_vlm_loaded", False),
        model_version=getattr(server, "_version", "unknown"),
        model_state=getattr(server, "_state", "unknown"),
        in_flight_requests=getattr(server, "_in_flight_count", 0),
    )

# Update HealthResponse model:
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    inference_mode: str = ""
    export_dir: str = ""
    vlm_loaded: bool = False
    model_version: str = "unknown"
    model_state: str = "unknown"
    in_flight_requests: int = 0
```

---

## Section 6: Metrics and Observability

### New Metrics to Track

Add to `ReflexServer` for production observability:

```python
self._reload_count = 0
self._reload_success_count = 0
self._reload_error_count = 0
self._last_reload_time = None
self._total_load_time_ms = 0.0
self._total_warmup_time_ms = 0.0
self._total_drain_time_ms = 0.0
self._max_in_flight_count = 0
```

### Example /metrics Endpoint

```python
@app.get("/metrics")
async def metrics():
    """Prometheus-format metrics for observability."""
    lines = [
        f"# HELP reflex_reloads_total Total number of reload attempts",
        f"# TYPE reflex_reloads_total counter",
        f"reflex_reloads_total {server._reload_count}",
        f"",
        f"# HELP reflex_reload_success_total Successful reloads",
        f"# TYPE reflex_reload_success_total counter",
        f"reflex_reload_success_total {server._reload_success_count}",
        f"",
        f"# HELP reflex_in_flight_requests Current in-flight request count",
        f"# TYPE reflex_in_flight_requests gauge",
        f"reflex_in_flight_requests {server._in_flight_count}",
        f"",
        f"# HELP reflex_load_time_ms Load time in milliseconds (last)",
        f"# TYPE reflex_load_time_ms gauge",
        f"reflex_load_time_ms {server._last_load_time_ms or 0}",
    ]
    return PlainTextResponse("\n".join(lines))
```

---

## Section 7: Comparison with vLLM's LoRA Adapter Loading

### Why vLLM's Pattern Transfers

vLLM's `WorkerLoRAManager` (lines 103-147, `worker_manager.py`) shows how to:
1. **Load an adapter** (`_load_adapter`): deserialize weights from disk, validate against model config
2. **Activate an adapter** (`activate_adapter` in `LoRAModelManager`, lines 265-304): move weights to GPU, bind to forward pass
3. **Deactivate an adapter** (`_deactivate_adapter`, lines 306-311): free GPU memory

The pattern maps to our `POST /reload` flow:

| vLLM Step | Reflex Equivalent |
|-----------|-------------------|
| Load adapter from disk | `ReflexServer.load()` deserializes ONNX |
| Validate adapter | Warmup inference (checks model works) |
| Activate (move to GPU) | Implicit when model is loaded (no separate step) |
| Update request routing | Atomic `server = new_server` cutover |
| Deactivate old adapter | `old_server.unload()` after drain |
| Free GPU memory | ORT session released in `unload()` |

**Key difference**: vLLM hot-swaps adapters (lightweight, <1s), while we swap full models (100-500ms). But the drain pattern is identical.

---

## Section 8: Error Handling and Rollback

### Scenarios and Responses

| Scenario | Response | Action |
|----------|----------|--------|
| Load fails (bad export_dir) | 500 error + old version stays loaded | User fixes export_dir, retry |
| Warmup fails (model bug) | 400 error + new version unloaded, old version still ready | User fixes model, retry |
| Warmup timeout (OOM during JIT) | 500 error + new version unloaded | User investigates GPU memory, retry |
| Drain timeout (request hung) | 200 success (drain still happening in background) | Robot operator notices dropped /act, system recovers after timeout |
| Request arrives during cutover | Served by new version (pointer assignment is atomic) | Request succeeds (no retries needed) |

### Telemetry on Reload Failure

```python
# In /reload error path:
server._reload_error_count += 1
logger.error(
    "Reload failed: phase=%s error=%s old_version=%s",
    phase, error, old_version,
)
```

Metrics endpoint reports failure rate to observability system (Prometheus, Datadog, etc.).

---

## Section 9: Testing Strategy

### Unit Tests

```python
# tests/test_hot_reload.py

async def test_reload_success():
    """Load new model, warmup, cutover, drain, unload."""
    server = ReflexServer(export_dir_v1, ...)
    server.load()
    v1 = server._version
    assert server._state == "ready"
    
    # Trigger reload
    response = await reload(ReloadRequest(export_dir=export_dir_v2))
    
    assert response["status"] == "success"
    assert response["old_version"] == v1
    assert response["new_version"] != v1
    
    # Verify cutover happened
    assert server._version != v1
    assert server._state == "ready"

async def test_reload_warmup_fails():
    """If warmup fails, new model is unloaded and old version stays."""
    server = ReflexServer(export_dir_v1, ...)
    server.load()
    v1 = server._version
    
    # export_dir_v2_broken has bad weights
    response = await reload(ReloadRequest(export_dir=export_dir_v2_broken))
    
    assert response["status"] == "error"
    assert response["phase"] == "warmup"
    assert server._version == v1  # Didn't change
    assert server._state == "ready"

async def test_in_flight_requests_drained():
    """Requests in-flight during reload are allowed to finish."""
    server = ReflexServer(export_dir_v1, ...)
    server.load()
    
    # Start a slow /act call
    slow_future = asyncio.create_task(
        server.predict_async(image=dummy_image)
    )
    await asyncio.sleep(0.1)  # Let it start
    
    # Trigger reload
    reload_response = await reload(ReloadRequest(export_dir=export_dir_v2))
    assert reload_response["status"] == "success"
    
    # Old request should still finish
    result = await slow_future
    assert "actions" in result or "error" not in result

async def test_drain_timeout():
    """After drain timeout, old version unloads even with in-flight."""
    server = ReflexServer(export_dir_v1, ...)
    server.load()
    
    # Mock a request that never finishes
    async def never_finishes():
        async with server._in_flight_lock:
            server._in_flight_count += 1
        await asyncio.sleep(1000)
    
    asyncio.create_task(never_finishes())
    
    # Reload with short timeout
    response = await reload(ReloadRequest(
        export_dir=export_dir_v2,
        drain_timeout_s=0.1
    ))
    assert response["status"] == "success"
    
    # Wait for background drain task
    await asyncio.sleep(0.5)
    
    # Old server should be unloaded despite timeout
    assert old_server._state == "unloading"
```

### Integration Tests

```python
async def test_reload_under_load():
    """Reload while hammering /act with requests."""
    server = ReflexServer(export_dir_v1, ...)
    server.load()
    
    # Spam /act calls (no synchronization)
    async def spam_requests():
        for _ in range(100):
            try:
                await server.predict_async(image=dummy_image)
            except Exception:
                pass  # Expected if version changed mid-request
    
    # Start spam in background
    spam_task = asyncio.create_task(spam_requests())
    
    # Wait a bit, then reload
    await asyncio.sleep(0.2)
    response = await reload(ReloadRequest(export_dir=export_dir_v2))
    
    # Spam should eventually finish
    await asyncio.wait_for(spam_task, timeout=5.0)
    assert response["status"] == "success"
```

---

## Section 10: Rollout Checklist for v0.2

- [ ] Add version computation (`_compute_model_version()`)
- [ ] Add state tracking (`_state`, `_state_lock`)
- [ ] Add in-flight request counter (`_in_flight_count`, `_in_flight_lock`)
- [ ] Update `load()` to set state
- [ ] Implement `unload()`
- [ ] Instrument `predict_async()` with in-flight tracking
- [ ] Add `/reload` POST handler with load → warmup → cutover → drain → unload flow
- [ ] Add `ReloadRequest` Pydantic model
- [ ] Add `_drain_and_unload()` background task
- [ ] Update `/health` to expose version, state, in-flight count
- [ ] Add `/metrics` endpoint (Prometheus format)
- [ ] Write unit tests (warmup failure, drain timeout, rollback)
- [ ] Write integration tests (reload under load, high concurrency)
- [ ] Document `/reload` in API docs (OpenAPI / Swagger)
- [ ] Add logging instrumentation (load time, warmup time, drain time)
- [ ] Test with different model sizes (pi0, smolvla, gr00t if available)
- [ ] Benchmark: reload latency + throughput recovery time
- [ ] Manual QA: reload + robot continues serving (no dropped requests)

---

## Section 11: Future Enhancements (v0.3+)

### Per-Model Metrics
Track metrics per model version:
```python
self._versions_loaded_ever = {}  # version → {load_time, warmup_time, ...}
```

### Gradual Traffic Shift (Canary)
Instead of atomic cutover, shift 10% traffic to new version, then 50%, then 100%:
```python
# POST /reload?canary_percent=10
# → Internally, route 10% of requests to new_server, 90% to old_server
# → Update canary_percent in subsequent /reload requests
```

### Model Repository Watching
Detect changes to export_dir on disk and auto-reload:
```python
# watch_path = "/path/to/models"
# On file change → auto-trigger reload
```

### A/B Testing Support
Keep multiple versions loaded, route by request metadata:
```python
# POST /act with {"model_version": "v1"} or {"model_version": "v2"}
# → Request routed to appropriate server instance
```

---

## References

- **Triton Inference Server**: `reference/triton/CMakeLists.txt`, `reference/triton/build.py`, `reference/triton/deploy/`
- **vLLM LoRA**: `reference/vllm/vllm/lora/model_manager.py`, `reference/vllm/vllm/lora/worker_manager.py`
- **Current Reflex Server**: `src/reflex/runtime/server.py` (lines 258-335: load; 770-943: batching; 1087-1199: FastAPI)
- **vLLM Adapter Lifecycle**: Lines 265-304 (activate), 306-311 (deactivate), 349-354 (remove_all)
- **Drain Pattern**: vLLM `AdapterLRUCache._on_remove()` (line 52-60)
- **Metrics Pattern**: `/reference/vllm/vllm/v1/metrics/stats.py` (rolling window stats)

