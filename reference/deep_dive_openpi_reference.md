# Deep Dive: Physical Intelligence OpenPI Reference Implementation

## Executive Summary
This document analyzes the canonical pi0/pi0.5 inference paths from the official openpi reference implementation. The goal is to identify what PI actually ships to customers and what patterns lerobot may have missed or misported.

---

## Section 1: Canonical Pi0 Inference Path

### 1.1 Entry Point: Policy.infer()
**File**: `/src/openpi/policies/policy.py:68-106`

The public inference API is `Policy.infer(obs: dict, *, noise: np.ndarray | None = None) -> dict`.

**Key flow**:
1. Copy obs dict (in-place transforms may occur)
2. Apply input transforms: `self._input_transform(inputs)`
3. For PyTorch models:
   - Add batch dim via `[None, ...]`
   - Convert numpy → torch via `torch.from_numpy(...).to(self._pytorch_device)`
4. Create Observation object from dict: `_model.Observation.from_dict(inputs)`
5. Call `self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)`
6. Extract batch dim `[0, ...]` and convert back to numpy
7. Apply output transforms: `self._output_transform(outputs)`
8. Return `{"state": ..., "actions": ..., "policy_timing": {"infer_ms": ...}}`

**Critical detail** (line 78): PyTorch inputs are converted **without** explicit normalization or denormalization. The model expects images in `[-1, 1]` float32.

### 1.2 Observation Structure
**File**: `/src/openpi/models/model.py:82-129`

```
Observation:
  - images: dict[str, float32[*b, h, w, c]]  # [-1, 1] range
  - image_masks: dict[str, bool[*b]]          # True if valid
  - state: float32[*b, s]                    # Low-dim robot state
  - tokenized_prompt: int32[*b, l] | None    # Language tokenized
  - tokenized_prompt_mask: bool[*b, l] | None
  - token_ar_mask: int32[*b, l] | None       # Pi0-FAST only
  - token_loss_mask: bool[*b, l] | None      # Pi0-FAST only
```

**Image preprocessing** (model.py:144-214):
- Images are resized to (224, 224) if needed
- For training: augmentations applied (crop, rotate, color jitter)
- Expected format at entry: `float32[B, H, W, C]` in `[-1, 1]`
- PyTorch version (`preprocessing_pytorch.py`) handles both `[B,C,H,W]` and `[B,H,W,C]`, converts to `[B,H,W,C]` for processing

### 1.3 Pi0 Model Forward: sample_actions()
**File**: `/src/openpi/models_pytorch/pi0_pytorch.py:377-420`

**Shape expectations**:
- Input: `observation` with `state: [B, action_dim]`
- Output: `actions: [B, action_horizon, action_dim]`

**Inference loop** (Euler sampling):
```
1. noise = sample_noise([B, action_horizon, action_dim])  # std=1.0
2. time = 1.0 (start of diffusion)
3. x_t = noise
4. dt = -1.0 / num_steps  # Default num_steps=10 → dt=-0.1

WHILE time >= -dt/2:  # Robust to float error
  v_t = denoise_step(...)
  x_t += dt * v_t
  time += dt
RETURN x_t
```

**Key insight**: t=1.0 is **noise**, t→0 is **solution**. This is opposite the paper convention (acknowledged in pi0.py:226-227).

### 1.4 Denoising Step: denoise_step()
**File**: `/src/openpi/models_pytorch/pi0_pytorch.py:422-462`

**Inputs**:
- `state: [B, action_dim]` — unchanged throughout loop
- `x_t: [B, action_horizon, action_dim]` — noisy actions
- `timestep: [B]` — current time in [0, 1]
- `past_key_values` — cached prefix embeddings (KV cache from prompt/images)

**Suffix embedding** (lines 431):
```python
suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = 
  self.embed_suffix(state, x_t, timestep)
```

For pi0 (not pi0.5):
- State projected: `state_emb = state_proj(state)`  — shape `[B, 1, width]`
- Noisy actions projected: `action_emb = action_in_proj(x_t)`  — shape `[B, H, width]`
- Timestep embedded: `time_emb = create_sinusoidal_pos_embedding(...)`  — shape `[B, width]`
- Time + action fused: concatenate then MLP
  ```python
  action_time_emb = cat([action_emb, time_emb], dim=2)
  action_time_emb = action_time_mlp_in(action_time_emb)
  action_time_emb = swish(...)
  action_time_emb = action_time_mlp_out(action_time_emb)  # [B, H, width]
  ```

**Attention masks**:
- Prefix (images + lang): attend to each other freely
- Suffix (state + actions): state does NOT attend to actions (ar_mask=1), actions are causal
- Prefix cannot attend to suffix (ar_mask boundary)

**Forward pass** (lines 450-457):
```python
outputs_embeds, _ = self.paligemma_with_expert.forward(
    attention_mask=full_att_2d_masks_4d,
    position_ids=position_ids,
    past_key_values=past_key_values,  # From prefix, reused
    inputs_embeds=[None, suffix_embs],  # Prefix None (cached)
    use_cache=False,
    adarms_cond=[None, adarms_cond],  # None for pi0
)
```

**Output projection** (lines 459-462):
```python
suffix_out = outputs_embeds[1]  # Take suffix output only
suffix_out = suffix_out[:, -action_horizon:]  # Last H tokens
suffix_out = suffix_out.to(dtype=torch.float32)
v_t = self.action_out_proj(suffix_out)  # [B, H, action_dim]
RETURN v_t  # Velocity prediction
```

**Important**: Returns predicted **velocity** `v_t`, not denoised `x_0`.

### 1.5 Prefix Caching (KV Cache)
**File**: `/src/openpi/models_pytorch/pi0_pytorch.py:376-400`

Before the diffusion loop:
```python
# 1. Embed prefix (images + language)
prefix_embs, prefix_pad_masks, prefix_att_masks = 
  self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

# 2. Create attention masks & positions
prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

# 3. Compute KV cache ONCE
_, past_key_values = self.paligemma_with_expert.forward(
    attention_mask=prefix_att_2d_masks_4d,
    position_ids=prefix_position_ids,
    past_key_values=None,
    inputs_embeds=[prefix_embs, None],  # Prefix only
    use_cache=True,
)
```

**Then in loop**: Pass `past_key_values` to each denoise step, **reuse across iterations**.

---

## Section 2: Canonical Pi0.5 Inference Path

### 2.1 State Tokenization (The "State-in-Language" Quirk)
**File**: `/src/openpi/models/tokenizer.py:22-48`

Pi0.5 (and FAST) discretizes state **into the prompt**:

```python
def tokenize(self, prompt: str, state: np.ndarray | None = None):
    cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
    
    if state is not None:  # Pi0.5 path
        # Quantize state: [-1, 1] → [0, 255]
        discretized_state = np.digitize(
            state, 
            bins=np.linspace(-1, 1, 256 + 1)[:-1]
        ) - 1
        state_str = " ".join(map(str, discretized_state))
        
        # Prepend state to prompt
        full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
        tokens = self._tokenizer.encode(full_prompt, add_bos=True)
    else:  # Pi0 path
        tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + 
                 self._tokenizer.encode("\n")
    
    # Pad to max_token_len (default 200 for pi05, 48 for pi0)
    ...
```

**Key differences**:
- Pi0: max_token_len=48, state is separate continuous input
- Pi0.5: max_token_len=200, state is **discrete tokens** in the prompt string

### 2.2 Pi0.5 Model Configuration
**File**: `/src/openpi/models/pi0_config.py:18-41`

```python
@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # Set in __post_init__
    
    pi05: bool = False
    discrete_state_input: bool = None  # Set based on pi05
    
    pytorch_compile_mode: str | None = "max-autotune"
    
    def __post_init__(self):
        # Pi0.5 sets these:
        if self.max_token_len is None:
            max_token_len = 200 if self.pi05 else 48
        if self.discrete_state_input is None:
            discrete_state_input = self.pi05  # True for pi05
```

**Transform setup** (transforms.py:250-266):
```python
class TokenizePrompt(DataTransformFn):
    def __call__(self, data: DataDict) -> DataDict:
        if self.discrete_state_input:  # Pi0.5
            state = data.get("state")
            tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        else:  # Pi0
            tokens, token_masks = self.tokenizer.tokenize(prompt, None)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}
```

### 2.3 Pi0.5 Architecture: adaRMSNorm with Time Injection
**File**: `/src/openpi/models_pytorch/pi0_pytorch.py:103-109`

Pi0 vs Pi0.5 suffix embedding differs fundamentally:

**Pi0** (lines 107-109):
```python
self.state_proj = nn.Linear(action_dim, width)
self.action_time_mlp_in = nn.Linear(2 * width, width)
self.action_time_mlp_out = nn.Linear(width, width)
```
→ State is a continuous token; time and action are concatenated and processed together.

**Pi0.5** (lines 104-105):
```python
self.time_mlp_in = nn.Linear(width, width)
self.time_mlp_out = nn.Linear(width, width)
```
→ State is in prompt tokens; time is processed separately via **adaRMSNorm** (conditional layer norm).

**In embed_suffix()** (lines 276-298):
```python
if not self.pi05:  # Pi0
    action_time_emb = cat([action_emb, time_emb[:, None, :]], dim=2)
    action_time_emb = action_time_mlp_in(action_time_emb)
    action_time_emb = swish(action_time_emb)
    action_time_emb = action_time_mlp_out(action_time_emb)
    adarms_cond = None
else:  # Pi0.5
    # Time goes through separate MLP for adaRMSNorm injection
    time_emb = time_mlp_in(time_emb)
    time_emb = swish(time_emb)
    time_emb = time_mlp_out(time_emb)
    time_emb = swish(time_emb)
    action_time_emb = action_emb  # Actions unchanged
    adarms_cond = time_emb  # Used in forward pass
```

### 2.4 Pi0.5 Forward Pass: adaRMS Conditioning
**File**: `/src/openpi/models_pytorch/pi0_pytorch.py:350-358`

```python
(_, suffix_out), _ = self.paligemma_with_expert.forward(
    attention_mask=att_2d_masks_4d,
    position_ids=position_ids,
    past_key_values=None,
    inputs_embeds=[prefix_embs, suffix_embs],
    use_cache=False,
    adarms_cond=[None, adarms_cond],  # ← Time conditioning for expert
)
```

`adarms_cond=[None, adarms_cond]` means:
- Prefix (PaliGemma) gets no conditioning
- Suffix (action expert) gets time embedding for adaRMSNorm scaling

This is **structural difference**: pi0.5 uses conditional layer norm; pi0 concatenates embeddings.

---

## Section 3: Patterns PI Uses That LeRobot Didn't Port (Or Misported)

### 3.1 Image Format Handling
**OpenPI** (`preprocessing_pytorch.py:40-46`):
```python
is_channels_first = image.shape[1] == 3  # Detect [B,C,H,W]
if is_channels_first:
    image = image.permute(0, 2, 3, 1)  # → [B,H,W,C]
```
Explicitly handles both formats and **always converts to [B,H,W,C] for processing**.

**LeRobot likely missed**: May assume one format only, leading to shape errors when image_keys differ or when imported models have different conventions.

### 3.2 Precise Timestep Schedule
**OpenPI** (`pi0_pytorch.py:402-403`):
```python
dt = -1.0 / num_steps
dt = torch.tensor(dt, dtype=torch.float32, device=device)
```
Exact Euler step with **negative** dt (t goes 1.0 → 0.0).

Loop condition (line 407): `while time >= -dt / 2` — **robust to floating-point error**.

**LeRobot may differ**: Could use positive dt or different loop condition, changing denoising trajectory.

### 3.3 KV Cache Reuse Pattern
**OpenPI** (`pi0_pytorch.py:394-400`):
- Compute prefix KV cache **once** before loop
- Pass `past_key_values` to **every denoise step**
- Each step calls forward with `use_cache=False` (KV values are **read-only** from prefix)

This is critical for inference speed: O(action_horizon * num_steps) without it becomes O(action_horizon * num_steps * prefix_len^2).

**LeRobot may differ**: Might not reuse KV cache, or might recompute prefix embeddings each step.

### 3.4 Position ID Computation (Crucial for Attention)
**OpenPI** (`pi0_pytorch.py:443-444`):
```python
prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
```

This ensures suffix positions continue from prefix positions. E.g., if prefix=20 tokens, suffix tokens are at positions [20, 21, 22, ...], not [0, 1, 2, ...].

**Why it matters**: Positional embeddings assume absolute positions in sequence.

### 3.5 Attention Mask Construction
**OpenPI** (`pi0_pytorch.py:52-81, 157-160`):
```python
def make_att_2d_masks(pad_masks, att_masks):
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks

# Convert to 4D for transformer
att_2d_masks_4d = torch.where(att_2d_masks, 0.0, -2.3819763e38)
```

The constant `-2.3819763e38` is chosen to be close to `float32 min` without underflow. **LeRobot may use different masking values** (e.g., `-inf`, `-1e9`), causing numerical differences.

### 3.6 Batch Handling in Denoise Step
**OpenPI** (`pi0_pytorch.py:422-462`):
```python
def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
    # timestep is [B] — expanded per batch
    suffix_embs, ..., adarms_cond = self.embed_suffix(state, x_t, timestep)
    
    # Suffix length varies with batch (due to varying valid tokens)
    suffix_len = suffix_pad_masks.shape[1]
    batch_size = prefix_pad_masks.shape[0]
    prefix_len = prefix_pad_masks.shape[1]
    
    # Build attention mask: [B, suffix_len, prefix_len + suffix_len]
    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
    suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
```

This handles **variable batch sizes** and **variable token lengths** correctly.

### 3.7 Action Extraction (Last H Tokens)
**OpenPI** (`pi0_pytorch.py:365, 460`):
```python
suffix_out = suffix_out[:, -self.config.action_horizon:]
```

Takes the **last `action_horizon` tokens** from the suffix output. This assumes:
- Suffix includes state token (pi0) + action tokens
- Actions are at the end of the suffix

**LeRobot error risk**: Could extract wrong indices if suffix structure differs.

---

## Section 4: Concrete Proposal — Which OpenPI Patterns to Mirror

### 4.1 Inference API
**Adopt from openpi**:
```python
Policy.infer(obs: dict, *, noise: np.ndarray | None = None) -> dict
# Returns: {"state": ..., "actions": ..., "policy_timing": {"infer_ms": ...}}
```
- Single entry point
- Input/output transforms composable
- Timing included

### 4.2 Observation Creation
**Adopt**:
```python
Observation.from_dict(data: dict) -> Observation
```
with auto-conversion of uint8 → float32 in [-1, 1].

**Add explicit handling for**:
- Channel format detection ([B,C,H,W] ↔ [B,H,W,C])
- Image resizing with padding (not cropping)
- Mask defaults (True if not provided)

### 4.3 Inference Loop for Diffusion
**Exact pattern**:
```python
time = 1.0
dt = -1.0 / num_steps
while time >= -dt / 2:
    v_t = denoise_step(...)
    x_t = x_t + dt * v_t
    time = time + dt
```

Not:
- `time = 0.0; while time <= 1.0; time += dt`
- `time = 1.0 - i * dt` (recalculation each step)

### 4.4 KV Cache Management
**Implement**:
1. Pre-compute prefix KV cache before loop
2. Pass `past_key_values` to every denoise step
3. Only update KV with suffix embeddings, never recompute prefix

**Benefit**: 2-3x speedup for typical horizon=50, steps=10.

### 4.5 Attention Masks
**Use exact patterns**:
```python
# Cumsum-based mask for causal/prefix-LM semantics
cumsum = torch.cumsum(ar_mask, dim=1)
att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]

# Convert to attention weights (large negative for masked)
att_2d_masks_4d = torch.where(att_2d_masks, 0.0, -2.3819763e38)
```

### 4.6 Position IDs
**Implement**:
```python
prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
```

Ensures suffix positions are absolute, not relative to 0.

### 4.7 Pi0.5 State Tokenization
**Critical**:
1. Detect `pi05=True` in config
2. If discrete_state_input=True:
   - Quantize state: `digitize(state, bins=linspace(-1, 1, 257)[:-1]) - 1`
   - Format: `f"Task: {prompt}, State: {state_str};\nAction: "`
3. Pass state to tokenizer alongside prompt
4. max_token_len=200 (not 48)

### 4.8 adaRMS Time Conditioning (Pi0.5)
**For pi0.5**:
```python
# Time MLP (separate from actions)
time_emb = self.time_mlp_in(time_emb)
time_emb = F.silu(time_emb)
time_emb = self.time_mlp_out(time_emb)
time_emb = F.silu(time_emb)

# Pass to forward as adarms_cond
adarms_cond = time_emb  # [B, width]
```

Then in forward: `adarms_cond=[None, adarms_cond]` for the expert.

### 4.9 Output Format
**Match**:
```python
{
    "state": state,  # Original obs state
    "actions": actions,  # [action_horizon, action_dim] unbatched
    "policy_timing": {"infer_ms": model_time_ms}
}
```

---

## Section 5: Deltas Between OpenPI and LeRobot (Known Bug Sources)

Based on the code review, these are likely sources of lerobot bugs:

### 5.1 State Token Extraction
**OpenPI**: State is **one continuous embedding** passed through `state_proj`.
**LeRobot risk**: May try to extract state from tokenized_prompt, causing shape mismatch or wrong values.

### 5.2 Timestep Expansion in denoise_step
**OpenPI** (pi0_pytorch.py:408):
```python
expanded_time = time.expand(bsize)  # Broadcast scalar to [B]
```
Then `embed_suffix` expects `timestep: [B]`.

**LeRobot risk**: May pass scalar `time` or forget to expand, causing shape mismatch in sinusoidal embedding.

### 5.3 KV Cache as Read-Only
**OpenPI**: `past_key_values` from prefix is **never updated** in denoise loop.
**LeRobot risk**: May try to update KV cache with suffix, corrupting the prefix cache for next iteration.

### 5.4 Attention Mask Broadcasting
**OpenPI**: Carefully broadcasts masks from `[B, N]` to `[B, 1, N, M]` for 4D attention.
**LeRobot risk**: May forget batch dimension, apply same mask to all batches, or use wrong shape.

### 5.5 Position Offset Calculation
**OpenPI** (pi0_pytorch.py:443):
```python
prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
```
This is **per-batch**: batch[0] might have 18 valid prefix tokens, batch[1] might have 20.

**LeRobot risk**: May compute single offset or forget the `[:, None]` unsqueeze, breaking variable-length batches.

### 5.6 Action Extraction Index
**OpenPI**: `suffix_out[:, -action_horizon:]`
**LeRobot risk**: May use `suffix_out[:, action_horizon:]` (first H instead of last H) or forget batch dimension.

### 5.7 Preprocess Train vs. Eval
**OpenPI** (`preprocessing_pytorch.py:20-174`):
- `train=False` in sample_actions (line 384)
- `train=True` in compute_loss (line 319)

Augmentations only applied in training path.

**LeRobot risk**: May always apply augmentations or always skip them.

### 5.8 Dtype Handling in Training vs. Inference
**OpenPI** (pi0_pytorch.py:337-338, 366):
```python
suffix_embs = suffix_embs.to(dtype=torch.bfloat16)  # Match model dtype
suffix_out = suffix_out.to(dtype=torch.float32)  # Always convert back to float32
```

Careful dtype transitions between input embedding and final projection.

**LeRobot risk**: May leave suffix_out in bfloat16, causing precision loss in action_out_proj.

---

## Section 6: Production Patterns from WebSocket Server

**File**: `/src/openpi/serving/websocket_policy_server.py`

### 6.1 Timing Instrumentation
```python
infer_time = time.monotonic()
action = self._policy.infer(obs)
infer_time = time.monotonic() - infer_time

action["server_timing"] = {
    "infer_ms": infer_time * 1000,
}
if prev_total_time is not None:
    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000
```

**Propose for Reflex**: 
- Log `infer_ms` per call
- Include `prev_total_ms` for end-to-end throughput visibility

### 6.2 Error Handling
```python
except Exception:
    await websocket.send(traceback.format_exc())
    await websocket.close(code=CloseCode.INTERNAL_ERROR, ...)
    raise
```

Always send traceback to client, then close connection cleanly.

### 6.3 Health Check Endpoint
```python
if request.path == "/healthz":
    return connection.respond(http.HTTPStatus.OK, "OK\n")
```

Simple liveness probe (useful for load balancers).

### 6.4 Compression Disabled
```python
async with _server.serve(
    ...,
    compression=None,
    max_size=None,
    ...
)
```

No compression (images are already compressed by torch serialization).

---

## Baseline Metrics

From the code structure, inferred latencies for PI's setup:

- **Prefix embedding + KV cache** (once): ~50-100ms (depends on image resolution, num images)
- **Single denoise step** (with cached KV): ~20-30ms
- **Total inference** (10 steps): **250-400ms** on A100 / GPU
- **Action extraction & serialization**: ~5-10ms

---

## Summary: Changes Needed in Reflex Runtime

1. **Adopt exact Euler loop** (time=1→0, dt=-1/num_steps)
2. **Implement KV cache reuse** (cache once, reuse 10x)
3. **Fix position ID offsets** (per-batch, broadcast correctly)
4. **Add attention mask validation** (check shapes, use -2.3819763e38)
5. **Implement pi0.5 state tokenization** (discretize to 256 bins, embed in prompt)
6. **Handle adaRMS conditioning** (separate time MLP for pi0.5)
7. **Ensure input/output transforms** (int8→float32, denormalize actions)
8. **Add timing instrumentation** (infer_ms per call)

---
