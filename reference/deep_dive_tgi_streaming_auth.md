# Deep Dive: TGI Streaming Response + Auth/Middleware Patterns
## Design Research for Reflex VLA Runtime Upgrade

**Objective**: Research TGI and vLLM's production-grade streaming + auth patterns to design `/act/stream` endpoint and middleware for Reflex serving runtime.

**Context**: Reflex VLA returns 50 actions per inference. Currently sent atomically. Proposal: stream actions 1-10 while computing 11-50, + upgrade basic API-key auth to TGI's level (proper validation, rate limiting, tracing).

---

## Section 1: Streaming Response Patterns

### 1.1 TGI: Server-Sent Events (SSE) over HTTP

**File**: `/reference/tgi/router/src/server.rs` (lines 441-506)

**Transport**: HTTP/1.1 with `text/event-stream` content type
- Endpoint: `POST /generate_stream`
- Response framing: Server-Sent Events (SSE) standard
- Header: `Content-Type: text/event-stream`
- Back-pressure buffer header: `X-Accel-Buffering: no` (line 530)

**Code Flow**:
```
generate_stream() [L476]
  → generate_stream_internal() [L508]
  → infer.generate_stream() [L555] — returns (permit, input_length, response_stream)
  → async_stream::stream! {} block [L532]
    → Loop: while let Some(response) = response_stream.next().await [L561]
      → Match on InferStreamResponse enum:
        - Prefill(_): ignored [L567]
        - Intermediate{token, top_tokens}: yield StreamResponse [L569-583]
        - End{token, top_tokens, generated_text, ...}: yield final + timings [L586-647]
```

**Flushing Strategy**:
- **Per-token flushing** (line 583, 647): Each token response immediately yields to Axum SSE wrapper
- No batching delay; no time-based coalescing
- `KeepAlive::default()` (line 504) sends ping frames every 30s if no token generated
- Semaphore permit (`permit`) kept alive for stream lifetime (L557 comment) — ensures concurrency bound

**Back-Pressure Handling**:
- `X-Accel-Buffering: no` disables nginx/proxy buffering (L530)
- Axum's `Sse<Stream>` wrapper yields to HTTP/1.1 chunked encoding automatically
- If client is slow, the generator async task yields on `response_stream.next().await` (L561)
- Slow client → backpressure propagates to backend batch scheduler (implicit via tokio select)

**Response Frame Format** (`StreamResponse` struct, inferred from lines 576, 639):
```json
{
  "index": 1,
  "token": {"id": 42, "text": "Hello", "logprob": -0.5},
  "generated_text": null,  // Only on final frame
  "details": null,         // Only on final frame
  "top_tokens": [{"id": 43, "text": "World"}]
}
```

**Metrics + Tracing**:
- Span recording (lines 612-617): total_time, validation_time, queue_time, inference_time, time_per_token recorded on final frame
- Histogram metrics (lines 620-626): request_duration, validation_duration, etc. recorded per token stream completion
- Request context propagated via OpenTelemetry span (line 485, 555)

---

### 1.2 vLLM: SSE + Streaming Options

**Files**: 
- `/reference/vllm/vllm/entrypoints/openai/api_server.py` (lines 1-200)
- `/reference/vllm/vllm/entrypoints/openai/server_utils.py` (SSE decoding, lines 167-317)
- `/reference/vllm/vllm/entrypoints/openai/chat_completion/protocol.py` (lines 1-135)

**Transport**: Same as TGI — HTTP/1.1 SSE, `text/event-stream`

**Key Differences from TGI**:
1. **Streaming Options** (protocol.py, lines 29): Client can request `stream_options={"include_usage": true}` to get token counts mid-stream
2. **Delta Messages**: Streaming uses `ChatCompletionStreamResponse` with `delta` field (incremental text), not full token object
3. **SSE Decoder** (server_utils.py, lines 197-291): Robust parsing with fallback for malformed chunks
   - Handles `data:` prefix + JSON blob per event
   - Multi-line event reassembly
   - Graceful failure on parse errors

**Streaming Decision**:
- Client specifies `"stream": true` in request (optional, default false)
- Server checks at request validation, routes to streaming handler or non-streaming

**Flushing**: Per-token, identical to TGI (inherent to SSE pattern)

---

## Section 2: Auth + Middleware Patterns

### 2.1 TGI: Bearer Token Middleware

**File**: `/reference/tgi/router/src/server.rs` (lines 2213-2236)

**Auth Pattern**: Simple Bearer token validation

```rust
if let Some(api_key) = api_key {
    let mut prefix = "Bearer ".to_string();
    prefix.push_str(&api_key);
    let api_key: &'static str = prefix.leak();  // ← STATIC lifetime for FnMut closure
    
    let auth = move |headers: HeaderMap,
                     request: axum::extract::Request,
                     next: axum::middleware::Next| async move {
        match headers.get(AUTHORIZATION) {  // ← Raw HTTP header access
            Some(token) => match token.to_str() {
                Ok(token_str) if token_str.to_lowercase() == api_key.to_lowercase() => {
                    let response = next.run(request).await;
                    Ok(response)  // ← 200 pass-through
                }
                _ => Err(StatusCode::UNAUTHORIZED),  // ← 401
            },
            None => Err(StatusCode::UNAUTHORIZED),  // ← 401
        }
    };
    
    base_routes = base_routes.layer(axum::middleware::from_fn(auth))
}
```

**Key Design Choices**:
1. **Conditional Layer**: Auth middleware only applied if `--api-key` flag provided (line 2213 condition)
2. **No Rate Limiting**: Simple presence check, no per-client quota or request counting
3. **No Request ID Propagation**: Authorization header checked but not correlated to trace spans
4. **Error Response**: Plain `StatusCode::UNAUTHORIZED` (no JSON body, no retry-after header)

**Where Applied**:
- Applied to `base_routes` (lines 2235) which includes `/generate`, `/generate_stream`, `/chat/completions`, etc.
- **NOT** applied to info routes: `/health`, `/ping`, `/info`, `/metrics` (lines 2237-2244) are unauthenticated
- Applied via Axum layer stacking (L2235) before OpenTelemetry layer (L2297)

---

### 2.2 Logging + Tracing Middleware

**File**: `/reference/tgi/router/src/logging.rs` (full file, 137 lines)

**Trace Context Injection** (lines 45-64):
```rust
pub async fn trace_context_middleware(mut request: Request, next: Next) -> Response {
    let context = request
        .headers()
        .get("traceparent")  // ← W3C Trace Context standard
        .and_then(|v| v.to_str().ok())
        .and_then(parse_traceparent)
        .map(|traceparent| {
            Context::new().with_remote_span_context(SpanContext::new(
                traceparent.trace_id,
                traceparent.parent_id,
                traceparent.trace_flags,
                true,
                Default::default(),
            ))
        });
    
    request.extensions_mut().insert(context);
    next.run(request).await
}
```

**Stack**:
- Parses `traceparent` header (format: `00-<trace_id>-<span_id>-<flags>`)
- Inserts OpenTelemetry `Context` into request extensions
- Handlers extract via `Extension(context)` and call `span.set_parent(context)` (server.rs line 280)

**OTLP Export** (lines 88-113):
- If `OTLP_ENDPOINT` env var set, creates OpenTelemetry SDK exporter
- Service name from `OTLP_SERVICE_NAME`
- Sampler: `Sampler::AlwaysOn` (no probabilistic sampling)
- Batch exporter (async, buffered)

**Log Levels** (lines 116-130):
- Env var `LOG_LEVEL`: TRACE, DEBUG, INFO, WARN, ERROR
- Module-scoped filter: `text_generation_launcher=info,text_generation_router=info`
- Output format: TEXT (colored) or JSON (env `LOG_FORMAT`)

**Applied In Stack** (server.rs line 2299):
```
.layer(axum::middleware::from_fn(trace_context_middleware))
```
Applied after OtelAxumLayer (L2297) for consistent request ID propagation

---

### 2.3 Request Validation Pattern

**File**: `/reference/tgi/router/src/validation.rs` (lines 1-150+)

**Validation Struct** (lines 28-39):
```rust
pub struct Validation {
    max_best_of: usize,
    max_stop_sequences: usize,
    max_top_n_tokens: u32,
    max_input_length: usize,        // ← Max input tokens
    max_total_tokens: usize,         // ← Max output tokens
    disable_grammar_support: bool,
    sender: mpsc::UnboundedSender<TokenizerRequest>,  // ← Validation worker channel
}
```

**Request Lifecycle**:
1. **Tokenization** (async, lines 104-129): Offload to background worker pool
   - Round-robin dispatch across N tokenizer workers (L88)
   - Spawn blocking task per worker (L76)
   - One-shot channel for response (L113)

2. **Validation** (lines 131-150):
   - `validate_input()` calls `tokenize()`, then checks constraints:
     - Input length vs `max_input_length`
     - Truncate to `max_input_length` if needed (L145-147)
     - Return input_length for tracing

3. **Error Returns**:
   - `ValidationError` enum (imported, used at server.rs L255)
   - HTTP 422 status on validation error (server.rs line 255)

**Integration** (server.rs lines 122-128):
```rust
let valid_request = self.validation.validate(request).await.map_err(|err| {
    metrics::counter!("tgi_request_failure", "err" => "validation").increment(1);
    tracing::error!("{err}");
    err
})?;
```
- Counter tagged with `err=validation`
- Error logged to structured logs (tracing)
- Returns to caller with 422 status

---

## Section 3: Request Lifecycle + Concurrency Control

### 3.1 Semaphore-Based Concurrency Limiting

**File**: `/reference/tgi/router/src/infer/mod.rs` (lines 48-96)

```rust
pub struct Infer {
    validation: Validation,
    backend: Arc<dyn Backend + Send + Sync>,
    chat_template: Option<ChatTemplate>,
    limit_concurrent_requests: Arc<Semaphore>,  // ← Semaphore for backpressure
    backend_health: Arc<AtomicBool>,
}
```

**Usage** (lines 112-120):
```rust
let permit = self
    .clone()
    .limit_concurrent_requests
    .try_acquire_owned()
    .map_err(|err| {
        metrics::counter!("tgi_request_failure", "err" => "overloaded").increment(1);
        tracing::error!("{err}");
        err
    })?;
```

**Key Points**:
- Semaphore created with `Arc::new(Semaphore::new(max_concurrent_requests))` (L84)
- Each request calls `try_acquire_owned()` (non-blocking)
- If all permits exhausted: HTTP 429 "overloaded" error
- Permit held for lifetime of request (returned from `generate_stream` as part of tuple, L105-107)
- Release on drop (tokio::sync::Semaphore semantics)

**No Explicit Rate Limiting**: No per-second quota, per-user bucket, or token consumption tracking — only concurrent slot limiting.

---

## Section 4: Concrete Proposal for Reflex `/act/stream` + Auth Upgrade

### 4.1 `/act/stream` Endpoint Design

**Endpoint**: `POST /act/stream`

**Request Body** (same as `/act`):
```json
{
  "image": "<base64 or URL>",
  "instruction": "pick up the cup",
  "state": [0.1, 0.2, ...],
  "options": {
    "chunk_size": 50,
    "flush_interval_ms": null,  // Optional: override per-request
    "include_metrics": true     // Optional: include timings in stream
  }
}
```

**Response**: HTTP/1.1 SSE, `Content-Type: text/event-stream`, `X-Accel-Buffering: no`

**Event Frame** (JSON per SSE event):
```json
{
  "chunk_index": 0,
  "action_index": 3,
  "action": [0.1, 0.05, -0.02, ...],
  "action_dim": 32,
  "generated_text": null,      // null until end
  "details": null,
  "finish_reason": null,       // null until end: "length", "stop", "error"
  "metrics": {
    "latency_ms": 15.2,        // Cumulative from start
    "hz": 6.5                  // Estimate
  }
}
```

**Flushing Strategy**:
- **Per-action flushing** (not per-token): Each action from `predict()` immediately yielded
- Async loop: `while let Some(action) = action_buffer.pop_or_wait().await { yield action }`
- No coalescing; no time-based batching
- If client slow: backpressure to model inference loop (implicit via tokio task yield)

**Completion Frame**:
```json
{
  "chunk_index": 0,
  "action_index": 49,
  "action": [...],
  "generated_text": "success",
  "details": {
    "finish_reason": "length",
    "total_actions": 50,
    "total_latency_ms": 253.1,
    "model_hash": "abc123...",
    "config_hash": "def456...",
    "reflex_version": "0.1.0"
  },
  "metrics": {...}
}
```

---

### 4.2 Auth Middleware Upgrade

**Pattern**: TGI-style Bearer token + instrumentation

**Implementation Location**: New file `src/reflex/runtime/auth.py`

**Middleware Function**:
```python
# Pseudo-code (not full implementation)

async def auth_middleware(request, call_next):
    """
    Validate Bearer token in Authorization header.
    Propagate trace context. Return 401 on invalid token.
    """
    auth_header = request.headers.get("authorization", "")
    
    # Extract token: "Bearer <token>" format
    if not auth_header.lower().startswith("bearer "):
        return JSONResponse(
            {"error": "missing_auth", "error_type": "authentication"},
            status_code=401
        )
    
    token = auth_header[7:]  # Skip "Bearer "
    
    # Validate against config (env var or init param)
    if token != config.api_key:
        return JSONResponse(
            {"error": "invalid_token", "error_type": "authentication"},
            status_code=401,
            headers={"Retry-After": "10"}  # ← TGI pattern
        )
    
    # Extract trace context (W3C traceparent header)
    traceparent = request.headers.get("traceparent")
    # ... parse and inject into telemetry context
    
    # Track request in metrics (per-endpoint, per-token)
    # No explicit rate limiting yet (Phase 2)
    
    response = await call_next(request)
    return response
```

**Applied To**:
- All routes except `/health`, `/ping`, `/info`, `/metrics` (unauthenticated)
- Applied early in middleware stack (before OpenTelemetry layer)

**Error Response Format** (JSON, not plain 401):
```json
{
  "error": "invalid_token",
  "error_type": "authentication",
  "request_id": "req-abc123",
  "timestamp": "2026-04-22T13:00:00Z"
}
```

**Rate Limiting** (Phase 2, not Phase 1):
- Token bucket per API key: X requests per minute
- Counter per endpoint (e.g., `/act`, `/act/stream`)
- Return 429 with `Retry-After` header

---

## Section 5: What to Keep vs Replace in `src/reflex/runtime/server.py`

### 5.1 Current State (Line Overview)

**File Size**: 1199 lines

**Major Components**:
- Lines 37-112: `ReflexServer.__init__()` — initialization + config
- Lines 135-175: `configure_replan()` — buffer-based streaming setup
- Lines 176-206: `_latency_percentiles()` — rolling window stats
- Lines 258-335: `load()` — ONNX + wedges loading
- Lines 337-500+: `_load_onnx()` — Provider selection + TRT EP
- Lines 600+: `predict()` — Core inference
- HTTP endpoint handlers (FastAPI or similar) — not shown in first 200 lines, presumably later

### 5.2 Keep (No Changes Required)

**✓ KEEP**: 
- `ReflexServer.__init__()` initialization structure
- `configure_replan()` buffer semantics — directly applicable to action streaming
- `_latency_percentiles()` — excellent for per-stream metrics
- `load()` and `_load_onnx()` — model loading + provider logic untouched
- `predict()` core inference loop — no changes needed
- Latency history rolling window (`_latency_history`)
- Safety guard + split orchestrator composition

**Rationale**: These are inference-layer components. Streaming is HTTP/protocol-layer, orthogonal.

---

### 5.3 Replace (HTTP/Middleware-Layer)

**✗ REPLACE**:
- Basic API-key auth (if present) → Adopt TGI's Axum middleware pattern
  - Move auth to standalone middleware file
  - Add trace context propagation (traceparent header parsing)
  - Add structured error responses with request_id

- Request validation → Adopt TGI's pattern
  - Offload tokenization to background workers (if text input)
  - Validate input size constraints
  - Return 422 with ValidationError on constraint violation

- Concurrency control → Adopt semaphore-based backpressure
  - Replace any queue-based batching with semaphore permit acquire/release
  - Return 429 "overloaded" on semaphore exhaustion

- Logging/telemetry → Adopt TGI's stack
  - Replace any custom logging with `tracing` crate (if using Rust/Axum)
  - Or use Python `structlog` + OpenTelemetry
  - Parse `traceparent` header, inject into span context

---

### 5.4 New (Streaming Endpoint)

**⊕ NEW**:
- `POST /act/stream` endpoint
  - Accept same body as `/act`
  - Return SSE stream of action frames
  - Per-action flushing (not per-token)
  - Include metrics in completion frame

- Async action generator
  - `async def generate_action_stream(request) -> AsyncIterator[ActionFrame]:`
  - Loop over actions from buffer or inference, yield immediately
  - Handle backpressure via async yield

- Optional: Action buffer streaming mode
  - If `configure_replan()` called, `/act/stream` yields from buffer
  - Otherwise, yields directly from `predict()`

---

## Section 6: Implementation Roadmap

### Phase 1: Streaming Foundation (No Code, Design Only)
- [x] Research TGI/vLLM patterns ← This document
- [ ] Adapt HTTP handler to support SSE (use FastAPI `StreamingResponse`)
- [ ] Add `/act/stream` endpoint with per-action flushing
- [ ] Integrate with existing `predict()` loop

### Phase 2: Auth Upgrade
- [ ] Extract Bearer token validation to standalone middleware
- [ ] Add traceparent header parsing (W3C Trace Context)
- [ ] Structured error responses with request_id + timestamp
- [ ] Unit tests for auth failure modes (missing, invalid, malformed)

### Phase 3: Request Validation + Concurrency
- [ ] Implement semaphore-based concurrency limiting
- [ ] Add input constraint validation (max image size, etc.)
- [ ] Return 422 on validation error
- [ ] Return 429 on semaphore exhaustion

### Phase 4: Observability (OTLP Integration)
- [ ] Add OpenTelemetry span recording (latency, action index, finish reason)
- [ ] Propagate trace context through streaming loop
- [ ] Histogram metrics (per-action generation latency)
- [ ] Export to OTLP collector if endpoint configured

### Phase 5: Rate Limiting (Optional)
- [ ] Token bucket per API key
- [ ] Per-endpoint quotas (e.g., `/act/stream` vs `/act`)
- [ ] Return 429 with `Retry-After` header

---

## Appendix: Key File References

| Pattern | File | Lines | Language |
|---------|------|-------|----------|
| SSE Streaming | `/reference/tgi/router/src/server.rs` | 441-506 | Rust |
| Streaming Response Handler | `/reference/tgi/router/src/infer/mod.rs` | 98-199 | Rust |
| Bearer Token Auth | `/reference/tgi/router/src/server.rs` | 2213-2236 | Rust |
| Trace Context Middleware | `/reference/tgi/router/src/logging.rs` | 45-64 | Rust |
| Logging Init | `/reference/tgi/router/src/logging.rs` | 66-136 | Rust |
| Validation Worker Pool | `/reference/tgi/router/src/validation.rs` | 28-150 | Rust |
| Concurrency Semaphore | `/reference/tgi/router/src/infer/mod.rs` | 48-120 | Rust |
| vLLM SSE Decoder | `/reference/vllm/vllm/entrypoints/openai/server_utils.py` | 167-317 | Python |
| vLLM Protocol | `/reference/vllm/vllm/entrypoints/openai/chat_completion/protocol.py` | 1-135 | Python |
| Current Reflex Server | `/src/reflex/runtime/server.py` | 1-1199 | Python |
| Reflex Config | `/src/reflex/runtime/server.py` | 135-175 | Python |

---

## Summary

**TGI's Streaming**:
- SSE over HTTP/1.1, per-token flushing
- `X-Accel-Buffering: no` to disable proxy buffering
- Keep-alive pings every 30s
- Semaphore-based concurrency control

**TGI's Auth**:
- Simple Bearer token validation in middleware
- 401 on invalid, 429 on overload
- No explicit rate limiting (Phase 2 feature)
- Trace context propagation via traceparent header

**Reflex Proposal**:
- `/act/stream` endpoint: SSE + per-action flushing
- Upgrade auth to TGI pattern (Bearer + trace context)
- Adopt semaphore-based concurrency limiting
- Keep existing inference loop + model loading untouched
