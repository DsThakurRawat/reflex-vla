# Deep Dive: vLLM Prefix Caching Implementation

## Executive Summary

vLLM's prefix caching is a token-granular, hash-based system that caches KV blocks by computing cryptographic hashes of token sequences and their ancestry. When multiple requests share identical token prefixes (even when they differ in later tokens), vLLM reuses cached KV blocks. The system is designed for LLM generative workloads where a batch of requests can have shared prompt tokens; it does NOT directly apply to VLA episodes where shared vision+language need per-timestep proprioceptive updates.

---

## Section 1: How It Works (The Algorithm)

### 1.1 Block Hashing: The Core Mechanism

**File reference:** `vllm/v1/core/kv_cache_utils.py:535-562`

vLLM hashes KV blocks at a fixed granularity (typically 16 tokens per block). Each block hash is computed as:

```
BlockHash = hash_function((parent_block_hash, token_ids, extra_keys))
```

Key details:
- **Parent dependency:** Each block hash depends on the previous block's hash, forming a chain. The first block uses `NONE_HASH` as parent (file: line 104-106).
- **Token content:** All token IDs in the block are hashed together as a tuple.
- **Extra keys:** Optional metadata (LoRA adapter names, multimodal feature IDs, cache salt) are included to distinguish requests that differ only in non-token attributes (file: lines 497-532).

**Hash function selection** (file: lines 88, 94-106):
- Default: `sha256_cbor` or `xxhash_cbor` (CBOR serialization for determinism)
- PYTHONHASHSEED env var controls reproducibility for Python's built-in `hash()` fallback
- NONE_HASH is initialized once per process and cached to avoid recomputation

### 1.2 Block Hash Computation Lifecycle

**File reference:** `vllm/v1/core/kv_cache_utils.py:565-616` (request_block_hasher)
**Also:** `vllm/v1/request.py:213-216`

Hashes are computed **incrementally** as tokens are added:

1. **Initial computation:** When a Request is created, `update_block_hashes()` scans all_token_ids and computes hashes for complete blocks (line 213-216, request.py).
2. **Per-token appends:** When new output tokens are appended via `append_output_token_ids()`, the hasher checks if a new full block has been completed (line 211, request.py).
3. **Lazy hashing:** Only **full blocks** are hashed; partial blocks remain unhashed (file: kv_cache_utils.py:577-579).

This means `request.block_hashes` grows incrementally and is queryable at any time.

### 1.3 Cache Lookup: Finding Hits

**File reference:** `vllm/v1/core/block_pool.py:184-209` and `vllm/v1/core/single_type_kv_cache_manager.py:420-468` (FullAttentionManager.find_longest_cache_hit)

The lookup is a **linear forward scan** that finds the longest matching prefix:

```python
for block_hash in block_hashes:
    if cached_block := block_pool.get_cached_block(block_hash):
        append(cached_block)
    else:
        break  # Cache miss: stop searching
```

**Key properties:**
- **Prefix property:** vLLM only reuses a contiguous block sequence from the start. Partial hits in the middle are discarded.
- **Early exit:** The moment any block hash is not found in the cache, the search stops.
- **Group-aware:** For models with multiple KV cache groups (e.g., full attention + sliding window), all groups must hit together, or the prefix is rejected (file: block_pool.py:197-209).

### 1.4 Block Storage: BlockHashToBlockMap

**File reference:** `vllm/v1/core/block_pool.py:34-128`

The cache maps block hashes to physical block IDs:

```python
self.cached_block_hash_to_block: dict[BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]]
```

**Design detail:** Hashes are packed with group IDs to distinguish identical token sequences in different attention layers (file: lines 49-68, kv_cache_utils.py).

**Deduplication:** Multiple requests can contribute identical blocks to the same hash key. When blocks collide (same hash), they are stored in a dict indexed by block_id. This allows one hash to point to many blocks (file: block_pool.py:75-91), reducing deduplication work but allowing block reuse.

---

## Section 2: Edge Cases and Handling

### 2.1 Variable Prefix Length

**File reference:** `vllm/v1/core/kv_cache_manager.py:195-206`

When a request is first scheduled, `get_computed_blocks()` performs cache lookup:
- Accepts cache hit for `min(prompt_length - 1, max_cache_hit_length)` tokens
- The `-1` ensures the last prompt token is recomputed to obtain logits
- This can force recomputation of an entire block if the hit length forces block-size alignment

**Problem it solves:** If all tokens were cached, the model would have no logits to return on the first forward pass.

### 2.2 Concurrent Requests Sharing Prefix

**File reference:** `vllm/v1/core/block_pool.py:391-406` (touch method)

When multiple requests hit the same cached block:

1. Block's `ref_cnt` is incremented for each request (line 404)
2. If `ref_cnt > 0`, the block is removed from the free queue (line 403)
3. Block is safe from eviction while any request holds it

**Serialization:** Only when ref_cnt drops to 0 does the block re-enter the free queue and become an eviction candidate.

### 2.3 GPU Memory Pressure: LRU Eviction

**File reference:** `vllm/v1/core/kv_cache_utils.py:158-178` (FreeKVCacheBlockQueue)
**Also:** `vllm/v1/core/block_pool.py:322-352` (get_new_blocks)

Eviction is **LRU-based**:
- Free blocks are maintained as a doubly-linked list ordered by eviction priority (line 169-174, kv_cache_utils.py)
- Blocks are appended to the tail when freed, establishing eviction order
- When new blocks are needed, the head of the free list is popped first (popleft, line 210-245)
- If a block is cached (has a block_hash), it is evicted before reallocation (line 354-389, block_pool.py)

**Two-phase eviction:**
1. Block loses all references (ref_cnt=0) → added to free list tail
2. Block needed for new request → removed from free list head, evicted from hash table (file: line 374)

### 2.4 Multimodal and LoRA Requests

**File reference:** `vllm/v1/core/kv_cache_utils.py:369-532`

Requests with multimodal features or LoRA adapters include **extra keys** in the block hash:

- **Multimodal:** MM feature identifier + relative offset within block (lines 389-453)
- **LoRA:** Adapter name (lines 456-468)
- **Cache salt:** A per-request nonce to force different cache entries (lines 518-520)

**Why needed:** Two requests with identical tokens but different MM features or LoRA adapters must produce different hashes, so they don't incorrectly share KV cache.

### 2.5 Prompt Embeddings (Alternative to Token IDs)

**File reference:** `vllm/v1/core/kv_cache_utils.py:471-494` and `vllm/v1/request.py:115-122`

Requests can supply pre-computed embeddings instead of token IDs. To ensure consistent hashing:
- Embeddings are SHA256-hashed per block (line 492, kv_cache_utils.py)
- The hash is cached in `request._prompt_embeds_per_block_hashes` to avoid recomputation (line 119, request.py)
- This hash becomes part of the extra_keys tuple

### 2.6 Sliding Window and Sparse Attention

**File reference:** `vllm/v1/core/single_type_kv_cache_manager.py:481-580` (SlidingWindowManager.find_longest_cache_hit)

For sliding window attention, cache hits must respect the window constraint:
- Computed blocks are right-aligned within the window (line 523-526)
- Leftmost tokens outside the window are replaced with null blocks (line 524)
- No cache hits are accepted if they break alignment with `alignment_tokens` (lines 537-541)

**Why:** Sliding window layers only attend to recent tokens; caching old tokens would produce incorrect attention outputs.

### 2.7 EAGLE Speculative Decoding

**File reference:** `vllm/v1/core/single_type_kv_cache_manager.py:422-468` (FullAttentionManager)

When EAGLE (a draft head) is enabled:
- The last matched block is always dropped (line 458-461)
- This forces recomputation of the last block to generate the hidden state needed by EAGLE's draft head

**Rationale:** EAGLE decoding requires fresh hidden states from the main model; using stale cached KV is unsafe.

### 2.8 Preemption and Cache Invalidation

**File reference:** `vllm/v1/core/kv_cache_manager.py:460-474` (reset_prefix_cache)

When `reset_prefix_cache()` is called (e.g., after model weight updates in RLHF):
- All block hashes are cleared from `BlockHashToBlockMap` (line 462)
- All blocks' `block_hash` metadata is reset (lines 465-466)
- Existing requests' cache hits become invalid on next lookup
- Hit rate metrics are recorded and reset (lines 472-473)

---

## Section 3: Patterns That DON'T Transfer to VLA

### 3.1 Token-Granular Hashing vs. Episode Granularity

**Problem:** vLLM hashes at the **token level** (16 tokens = 1 block). VLA episodes have **stable vision+language** that should be cached as a unit, but **per-timestep proprioceptive state changes**. Token-level hashing will NOT match across episodes if proprioception differs.

**Why it fails:** If episode E1 has tokens [vision, language] + props_t1, and episode E2 has [vision, language] + props_t2, the block hash will differ even though vision+language are identical.

### 3.2 Autoregressive Generation Assumptions

**Problem:** vLLM's cache lookup assumes **sequential token generation**. The cache hit lookup scans block hashes in order and stops at the first miss (file: single_type_kv_cache_manager.py:447-457). This works for LLMs because output tokens are generated one by one.

**Why it fails for VLA:** VLA timesteps are **NOT generated sequentially**. Each timestep receives fresh proprioceptive input and must re-infer the action. There's no "parent block hash" relationship between timesteps—each denoising step stands alone.

### 3.3 Request Lifecycle Binding

**Problem:** vLLM ties cache hits to a **Request object** that persists for the entire generation (from prompt to completion). Cache stats are recorded per request (file: kv_cache_manager.py:208-214). A request's blocks are freed when finished.

**Why it fails for VLA:** A robot episode is **not a single request**. It's a sequence of **independent denoising steps**, each consuming fresh state. Binding cache to a "request" means invalidating the entire cache when the episode ends—exactly what we want to avoid.

### 3.4 Hitting Requires Contiguous Prefix Match

**Problem:** vLLM only accepts cache hits if all blocks match from the start. A single miss breaks the prefix (file: single_type_kv_cache_manager.py:456, break statement). Partial hits in the middle of the token sequence are not reused.

**Why it fails for VLA:** If vision+language are fully cached but proprioception changes slightly, the entire block hash fails. There's no way to reuse the vision+language KV without re-hashing proprioception as part of the same block.

### 3.5 Reference Counting for Deallocation

**Problem:** vLLM uses ref_cnt to track which requests hold a block. When all requests release a block (ref_cnt=0), it becomes evictable. This is tied to request lifecycle (file: block_pool.py:391-406).

**Why it fails for VLA:** Episodes don't have a "lifecycle" where cache ownership transfers between requests. Each timestep is independent. You need **episodic reference counting** (episode_id → how many timesteps currently depend on this cache), not request-based.

### 3.6 Hash Collisions and Deduplication Are Not Handles Gracefully

**Problem:** vLLM assumes hash collisions are rare (SHA256 is cryptographically strong). When collisions occur, multiple blocks map to the same hash, and the cache returns any one of them (file: block_pool.py:62-73). For most real workloads, collisions are impossible, but the design doesn't gracefully handle them.

**Why it matters for VLA:** If you're caching hundreds of episodes × thousands of timesteps, and you want to reuse cache across similar episodes, **hash collisions become a real concern**. You need explicit metadata matching (episode_id, timestep_range, proprioception_hash), not just token content hash.

---

## Section 4: Concrete Proposal for Pi05PrefixCache

### 4.1 Core Data Structures

```python
@dataclass
class CachedEpisodePrefix:
    """A cached KV prefix tied to an episode, not a request."""
    episode_id: str                      # Episode identifier
    vision_language_tokens: list[int]    # Immutable token sequence
    vision_language_hash: BlockHash      # SHA256 of token content
    kv_blocks: list[KVCacheBlock]        # Physical KV blocks on GPU
    ref_count: int = 0                   # How many timesteps depend on this
    last_accessed_ns: int = 0            # For LRU eviction
    birth_time_ns: int = 0               # For age-based metrics


class Pi05PrefixCache:
    """Episode-aware KV cache manager for VLA denoising.
    
    Core design:
    - Cache is keyed by (episode_id, vision_language_hash)
    - Each cached prefix is referenced by active timesteps
    - Timesteps are independent; each can have different proprioception
    - Eviction is LRU on episodes, not requests
    - Hit rate is episodic (% of timesteps that reuse the VL cache)
    """
    
    def __init__(
        self,
        num_gpu_blocks: int,
        block_size: int,
        hash_fn: Callable = sha256_cbor,
    ):
        self.num_gpu_blocks = num_gpu_blocks
        self.block_size = block_size
        self.hash_fn = hash_fn
        
        # Mapping: (episode_id, vl_hash) → CachedEpisodePrefix
        self.episode_cache: dict[tuple[str, BlockHash], CachedEpisodePrefix] = {}
        
        # Free block queue (same as vLLM)
        self.free_block_queue = FreeKVCacheBlockQueue(...)
        
        # Metrics
        self.hit_count = 0        # Timesteps that hit the cache
        self.miss_count = 0       # Timesteps that missed
        self.episode_count = 0    # Unique episodes processed


# Per-timestep denoising context
@dataclass
class TimestepContext:
    """Runtime state for a single denoising step."""
    episode_id: str
    timestep: int
    vision_language_tokens: list[int]    # Immutable for this episode
    proprio_tokens: list[int]             # Changes per timestep
    cached_vl_prefix: CachedEpisodePrefix | None  # Will be set on cache hit
```

### 4.2 Key Methods

#### 4.2.1 Lookup (Cache Hit/Miss)

```python
def get_cached_prefix(
    self,
    episode_id: str,
    vision_language_tokens: list[int],
) -> CachedEpisodePrefix | None:
    """
    Lookup cached VL prefix for an episode.
    
    Returns:
        The cached prefix if found and not evicted, None otherwise.
    
    File analogs:
      - vllm/v1/core/kv_cache_manager.py:176-216 (get_computed_blocks)
      - vllm/v1/core/single_type_kv_cache_manager.py:420-468 (find_longest_cache_hit)
    """
    vl_hash = BlockHash(
        self.hash_fn(tuple(vision_language_tokens))
    )
    key = (episode_id, vl_hash)
    
    if prefix := self.episode_cache.get(key):
        # Hit: refresh LRU timestamp
        prefix.last_accessed_ns = time.monotonic_ns()
        self.hit_count += 1
        return prefix
    
    self.miss_count += 1
    return None


def allocate_or_reuse_prefix(
    self,
    episode_id: str,
    vision_language_tokens: list[int],
    num_vl_blocks: int,
) -> CachedEpisodePrefix:
    """
    Allocate KV cache blocks for a new VL prefix, or return cached if exists.
    
    File analogs:
      - vllm/v1/core/block_pool.py:322-352 (get_new_blocks)
      - vllm/v1/core/single_type_kv_cache_manager.py:143-215 (allocate_new_computed_blocks)
    """
    vl_hash = BlockHash(
        self.hash_fn(tuple(vision_language_tokens))
    )
    key = (episode_id, vl_hash)
    
    if prefix := self.episode_cache.get(key):
        prefix.ref_count += 1
        return prefix
    
    # Allocate new blocks
    if num_vl_blocks > self.free_block_queue.num_free_blocks:
        self._evict_lru_episode()
    
    blocks = self.free_block_queue.popleft_n(num_vl_blocks)
    prefix = CachedEpisodePrefix(
        episode_id=episode_id,
        vision_language_tokens=vision_language_tokens,
        vision_language_hash=vl_hash,
        kv_blocks=blocks,
        ref_count=1,
        birth_time_ns=time.monotonic_ns(),
        last_accessed_ns=time.monotonic_ns(),
    )
    
    self.episode_cache[key] = prefix
    self.episode_count += 1
    return prefix
```

#### 4.2.2 Deallocation

```python
def release_prefix(
    self,
    prefix: CachedEpisodePrefix,
) -> None:
    """
    Release a timestep's hold on a VL prefix. If ref_count reaches 0,
    mark for eviction.
    
    File analog:
      - vllm/v1/core/block_pool.py:408-422 (free_blocks)
    """
    prefix.ref_count -= 1
    assert prefix.ref_count >= 0


def _evict_lru_episode(self) -> None:
    """
    Evict the least-recently-used episode's cached prefix.
    
    File analog:
      - vllm/v1/core/block_pool.py:354-389 (_maybe_evict_cached_block)
    """
    # Find the prefix with min last_accessed_ns that has ref_count == 0
    evictable = [
        (key, prefix) for key, prefix in self.episode_cache.items()
        if prefix.ref_count == 0
    ]
    
    if not evictable:
        # No free prefixes; force eviction of the LRU one
        evictable = list(self.episode_cache.items())
    
    if evictable:
        key, prefix = min(
            evictable,
            key=lambda x: x[1].last_accessed_ns,
        )
        del self.episode_cache[key]
        self.free_block_queue.append_n(prefix.kv_blocks)
```

#### 4.2.3 Hit Rate Metric

```python
@property
def cache_hit_rate(self) -> float:
    """
    Return the hit rate for episodic timesteps.
    
    File analog:
      - vllm/v1/metrics/stats.py:101-111 (CachingMetrics.hit_rate)
    """
    total = self.hit_count + self.miss_count
    if total == 0:
        return 0.0
    return self.hit_count / total


def make_stats(self) -> dict[str, float]:
    """
    Return current cache statistics (for monitoring/logging).
    
    File analog:
      - vllm/v1/core/kv_cache_manager.py:164-174 (make_prefix_cache_stats)
    """
    return {
        "hit_rate": self.cache_hit_rate,
        "total_hits": self.hit_count,
        "total_misses": self.miss_count,
        "unique_episodes": self.episode_count,
        "num_cached_prefixes": len(self.episode_cache),
        "num_free_blocks": self.free_block_queue.num_free_blocks,
    }
```

### 4.3 Design Rationale

| vLLM Concept | VLA Pi05 Equivalent | Reason |
|---|---|---|
| `Request` (request lifecycle) | `CachedEpisodePrefix` (episode lifetime) | Episodes are stable; timesteps are ephemeral |
| `block_hashes` (token chain) | `vision_language_hash` (single hash) | Vision+language are immutable per episode; no chaining needed |
| `ref_cnt` per request | `ref_count` per episode | Timesteps within an episode share the same prefix |
| `FreeKVCacheBlockQueue` LRU | Same LRU queue | Eviction policy is identical |
| `find_longest_cache_hit()` scan | `get_cached_prefix()` direct lookup | No sequential token dependency; O(1) lookup by hash |
| `PrefixCacheStats.hit_rate` (tokens) | `cache_hit_rate` (timesteps) | Metrics are per-timestep, not per-token |
| `skip_reading_prefix_cache` flag | N/A | VLA always wants to cache (no prompt logprobs needed) |
| Multimodal extra keys | Episode metadata (proprioception not in hash) | Proprioception changes per timestep, not part of the prefix |

### 4.4 Integration with Denoising Loop

```python
class VLADenoisingStep:
    def __init__(self, cache: Pi05PrefixCache):
        self.cache = cache
    
    def denoise_timestep(
        self,
        episode_id: str,
        timestep: int,
        vision_language_tokens: list[int],
        proprio_tokens: list[int],
    ) -> list[int]:
        """
        Denoise a single timestep. Reuse cached VL KV if available.
        """
        # Try to hit the cache
        prefix = self.cache.get_cached_prefix(episode_id, vision_language_tokens)
        
        if prefix is None:
            # Cache miss: allocate new blocks and compute VL
            num_vl_blocks = (len(vision_language_tokens) + self.cache.block_size - 1) // self.cache.block_size
            prefix = self.cache.allocate_or_reuse_prefix(
                episode_id, vision_language_tokens, num_vl_blocks
            )
            # [Compute VL KV, store in prefix.kv_blocks]
        else:
            # Cache hit: reuse blocks
            prefix.ref_count += 1
        
        # Denoise with combined VL + proprioception
        all_tokens = vision_language_tokens + proprio_tokens
        # [Run model forward with prefix.kv_blocks as cached KV]
        
        # Release the prefix after denoising
        self.cache.release_prefix(prefix)
        
        return action_tokens
```

---

## Section 5: File References Summary

### Core Files

| File | Lines | Purpose |
|---|---|---|
| `kv_cache_utils.py` | 109-156 | `KVCacheBlock` and metadata |
| `kv_cache_utils.py` | 158-367 | `FreeKVCacheBlockQueue` (LRU linked list) |
| `kv_cache_utils.py` | 535-562 | `hash_block_tokens()` (hashing algorithm) |
| `kv_cache_utils.py` | 565-616 | `get_request_block_hasher()` (incremental hashing) |
| `block_pool.py` | 34-128 | `BlockHashToBlockMap` (hash → block lookup) |
| `block_pool.py` | 130-510 | `BlockPool` (cache management) |
| `kv_cache_manager.py` | 176-216 | `get_computed_blocks()` (cache lookup entry point) |
| `single_type_kv_cache_manager.py` | 420-468 | `FullAttentionManager.find_longest_cache_hit()` (cache scan) |
| `request.py` | 162-216 | Request's `block_hashes` and update logic |
| `metrics/stats.py` | 18-112 | `PrefixCacheStats` and `CachingMetrics` |
| `kv_cache_metrics.py` | 46-96 | `KVCacheMetricsCollector` (residency tracking) |

### Test Files

| File | Purpose |
|---|---|
| `tests/v1/core/test_kv_cache_utils.py:180-200` | Hash reproducibility with PYTHONHASHSEED |
| `tests/v1/core/test_kv_cache_utils.py:237-350` | FreeKVCacheBlockQueue operations |
| `tests/v1/core/test_kv_cache_metrics.py:86-225` | Block lifecycle metrics (alloc, access, evict) |

---

## Appendix: Collision Detection Strategy (Future)

Since vLLM doesn't handle collisions gracefully, Pi05 should add:

```python
@dataclass
class VerifiedEpisodePrefix(CachedEpisodePrefix):
    """Extends CachedEpisodePrefix with collision detection."""
    vision_language_tokens_backup: list[int]  # Verify on hit
    
    def verify_match(self, other_tokens: list[int]) -> bool:
        """Verify that cache entry truly matches the tokens."""
        return other_tokens == self.vision_language_tokens_backup
```

On cache hit, verify the tokens match before reusing to ensure hash collisions don't corrupt KV cache.

