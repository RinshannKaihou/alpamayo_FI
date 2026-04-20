# Alpamayo-v1.5 BFA Experiments — Running Summary

Gradient-guided Bit-Flip Attack (BFA) on the Alpamayo-v1.5 VLA model in BF16.
Complements `pytorch_sdc_experiments_summary.md` (prior CV-model work).

---

## 1. Scope (Phase 1)

**In scope:**
- BF16 single-bit flips on weight tensors, dtype `"auto"` (i.e., whatever `from_pretrained(dtype=torch.bfloat16)` loads).
- Gradient-guided ranking via a **one-step flow-matching (FM) training loss** as surrogate.
- Target modules: `nn.Linear` inside `model.expert` + `model.action_in_proj` + `model.action_out_proj`.
- Measurement loop: flip → re-run FM one-step loss → restore.
- Output: per-trial JSON + per-bit loss-delta distribution plot.

**Explicitly out of scope (future phases):**
- VLM backbone (Qwen3-VL-8B) weights — would require FSDP for backward.
- Random baseline (trivial drop-in once Phase 1 is validated).
- Multi-bit byte-aligned flips (Sullivan et al.).
- Full-trajectory ADE/FDE measurement — the FM one-step loss is the fast proxy.
- Fake-quant / INT8 — user excluded; we target native BF16 only.
- Training-time injection.

---

## 2. Why Alpamayo is architecturally different from prior CV targets

| Axis | ResNet-50 / ViT-base (prior work) | Alpamayo-v1.5 |
|---|---|---|
| Params | 25M–86M | ~10B (VLM + expert + projections) |
| Forward pass | Single pass | VLM autoregressive → Expert iterative (10 Euler steps) |
| Native dtype | FP32 / INT8 fake-quant | **BF16** |
| Output | 1000-class logits → argmax absorbs small perturbations | 64 waypoints × (accel, curvature), integrated twice — no absorbing op |
| Safety criticality | Misclassification | Vehicle trajectory deviation |

Key architectural notes (from reading `alpamayo1.5/src/alpamayo1_5/`):
- Expert uses **non-causal attention** (`expert_non_causal_attention=True`) — bidirectional, so a corrupted weight affects all diffusion tokens symmetrically.
- VLM's KV-cache is shared with expert. VLM-side corruption silently propagates.
- Action space has 4 `register_buffer` scalars (`accel_mean/std`, `curvature_mean/std`) that denormalize every waypoint — direct analogue to INT8 per-channel scale vulnerability.
- `action_to_traj` uses **double `torch.cumsum`** (velocity + heading). Single-waypoint errors compound through kinematic integration on top of Euler integration.

---

## 3. Methodology (Phase 1)

### 3.1 Surrogate loss

One-step flow matching, mirroring the model's training objective:

```
t = 0.5  (fixed)
x_t      = (1 - t) * noise + t * gt_action
target_v = gt_action - noise
pred_v   = action_out_proj( expert( action_in_proj(x_t, t), past_kv=VLM_prefill_cache ) )
loss     = MSE(pred_v, target_v)
```

- VLM prefill is run **once** under `@torch.no_grad()` to build an `FMContext` holding `(prompt_cache, position_ids, attention_mask, gt_action)`. Cost ~600 ms. Reused across all trials.
- Only expert + action projections are in the gradient graph — the ~10× smaller parameter count vs. VLM makes the backward tractable on a single H20.

### 3.2 Gradient computation

- `loss_fn()` called once against clean weights → `.backward()` → `.grad.detach().clone().float()` snapshotted per target module.
- Gradients are **not recomputed per bit** (unlike the CV-model workflow). Same clean grads used across all 16 bit positions. Saves 15× backward passes with no loss of correctness (grads are always taken at clean weights).

### 3.3 Bit-gradient and ranking

Per coordinate, per bit position `b`:

```
delta_w = bf16_xor_bit(w, b) - w       # fp32 after view(bf16 -> int16) XOR then cast
bit_grad = grad * delta_w              # fp32; non-finite entries replaced with sign(grad)·1e30
```

Top-K largest bit_grad values → coordinates most predicted to **increase** loss when flipped.

### 3.4 Injection-measure-restore

Single model copy (no fresh reload per bit — too expensive at 10B params). Per trial:

1. `bf16_flip_one(module.weight.data, flat_idx, bit)` — in-place XOR on a single element.
2. `loss_fn()` under `@torch.no_grad()` — ~30 ms on H20.
3. `restore_one(module.weight.data, flat_idx, orig)` — writes back original value.

`measure_flipped_loss` wraps this in try/finally so restore always runs.

### 3.5 BF16-specific implementation note

`torch.bitwise_xor` is **not implemented for `uint16` on CUDA** (uint16 is a minimal-op dtype in PyTorch). We view BF16 as `torch.int16` instead — same bit pattern, fully supported. The mask for bit 15 (`1 << 15 = 0x8000`) is converted to signed `-32768` via two's-complement arithmetic.

### 3.6 Determinism for comparable losses

- `fixed_noise = torch.randn_like(ctx.gt_action)` seeded via `torch.cuda.manual_seed_all(42)` before `ctx.gt_action` is drawn.
- `t_val` fixed at 0.5.
- Every `loss_fn()` call uses the same `(x_t, target_v)` → only the weight flip varies.
- The determinism smoke check wraps the two `loss_fn()` baseline calls in `torch.no_grad()` to prevent `DynamicCache.crop` retaining autograd graph state across calls.

---

## 4. Deliverables

Both files live in `alpamayo1.5/notebooks/` so the notebook can `import bfa_utils as bfa` without packaging.

| File | Purpose |
|---|---|
| `alpamayo1.5/notebooks/bfa_utils.py` | Flat utilities: 3 BF16 bit primitives, `bit_grad_per_bit`, `collect_target_linears`, `FMContext` + `build_fm_context` + `fm_one_step_loss`, `compute_clean_grads`, `topk_bitflip_coords`, `measure_flipped_loss`. No classes except `FMContext` dataclass. ~340 LOC. |
| `alpamayo1.5/notebooks/bfa_demo.ipynb` | 12-cell demo. Order: header → imports → load model+data → build FM context + baseline → collect targets + grads → **smoke tests (3 cells)** → main BFA loop → analysis → per-bit boxplot. Smoke tests deliberately placed **before** the ~65 min main loop. |

Plan document: `docs/plans/2026-04-17-alpamayo-bfa.md`.

---

## 5. Key design decisions (and rationale)

| Decision | Rationale |
|---|---|
| Surrogate = FM one-step loss, not full-trajectory ADE | `flow_matching.sample()` is `@torch.no_grad()`. FM one-step is the model's actual training loss, is differentiable, and is ~40× faster to measure. |
| Single model copy, not fresh-reload per bit | 10B-param reload costs ~60 s; fresh grads per bit would cost 16× ~1 s backward. Single copy + clean-grad snapshot + in-place flip/restore is 16-20× faster end-to-end. |
| Compute clean grads once at start | Grads are always w.r.t. clean weights; flips are never accumulated (always restored). |
| Cache crop in `try/finally` | A single trial exception would otherwise leave `ctx.prompt_cache` extended, silently corrupting all subsequent trials. |
| BF16 viewed as `int16`, not `uint16` | CUDA kernel coverage. See §3.5. |
| `"action_out_proj"` without trailing dot, `"action_in_proj."` with | `action_out_proj` is a plain `nn.Linear` at the root; `action_in_proj` is a container with sub-linears. Asymmetry is intentional — do not "normalize". |
| Smoke tests BEFORE main loop | Broken bit-flip primitive wastes ~65 min of H20 time if discovered post-hoc. Smoke validates in <5 s. |
| VLM excluded from Phase 1 targets | Backward through an 8B VLM on one H20 is memory-tight; FSDP is a Phase 3 concern. |

---

## 6. Status

- ✅ Plan drafted (`docs/plans/2026-04-17-alpamayo-bfa.md`).
- ✅ `bfa_utils.py` implemented and reviewed (3 rounds: spec + quality + final).
- ✅ `bfa_demo.ipynb` implemented with 12 cells, smoke tests front-loaded.
- ✅ BF16 uint16-on-CUDA issue patched (switch to int16 view).
- 🔜 Run the full sweep on 4×H20.
- 🔜 Expected output: `bfa_results.json` (~64K trials), `bfa_per_bit.png`.

## 7. Planned extensions (not implemented)

- **Random baseline** — same loop, replace `topk_bitflip_coords` with uniform random coord sampling.
- **Diffusion-step sensitivity** — corrupt at specific Euler step `i ∈ [0, 10)`; measure trajectory error.
- **Action-space scalar attack** — exhaustive sweep over the 4 `register_buffer` denormalization scalars (only 64 trials total).
- **Cross-stage amplification** — inject into VLM weights, measure expert loss drift via the shared KV-cache.
- **VLM-target BFA** — same pipeline but `include_prefixes=("vlm.",)` + FSDP-sharded backward.
- **Full-trajectory ADE follow-up** — for the top-N highest-delta_loss positions, run `sample_trajectories_from_data_with_vlm_rollout` and compute ADE vs ground-truth.

---

## 8. Runtime budget (reference)

Single H20 (96 GB HBM3, ~296 BF16 TFLOPS):

| Phase | Count | Per-trial | Total |
|---|---|---|---|
| Model load | 1 | ~60 s | 60 s |
| `build_fm_context` | 1 | ~600 ms | <1 s |
| `compute_clean_grads` | 1 | ~1 s | 1 s |
| Smoke tests | 3 cells | ~1–3 s | ~5 s |
| BFA main loop | 16 bits × ~200 modules × k=20 | ~60 ms/trial | ~65 min |

With 4× H20, run 4 independent clip scenes in parallel → 4 complete sweeps in ~65 min.

---

## 9. Reference: prior CV-model methodology

See `pytorch_sdc_experiments_summary.md` (same folder) for the BFA/random/multi-bit-byte-aligned methodology originally developed for ResNet-50, ViT-base, and Qwen3-30B-A3B. The Alpamayo adaptation inherits the bit-gradient idea and the inject-measure-restore loop structure, but changes: loss function (FM one-step vs cross-entropy), ranking scope (no module-reload-per-bit), dtype (BF16 native), and measurement metric (trajectory-regression loss vs classification loss).

---

## 10. Phase 2a: KV-cache fault injection (addendum — 2026-04-20)

Activation-level variant targeting the **VLM prefill K/V cache** instead of weights. Complements, does not replace, Phase 1. Added because (i) the Phase 1 write-up already identifies the cache as the sole propagation medium between VLM and expert (§2, §7) but never targets it directly, and (ii) the gradient path is cheaper than §7's VLM-weight attack — no FSDP required.

### 10.1 Scope

**In scope:**
- BF16 single-bit flips on K and V tensors inside `ctx.prompt_cache` (HF `DynamicCache`) after the VLM prefill.
- Gradient-guided ranking via the same FM one-step loss, with `∂loss/∂K_l` and `∂loss/∂V_l` obtained by toggling `requires_grad_(True)` on the cache tensors.
- Same injection-measure-restore loop structure as §3.4. Per-layer, per-K/V, per-bit aggregation.

**Out of scope (deferred):**
- Multi-bit byte-aligned flips on cache entries (mechanical port of the multi-bit code once Phase 2a is validated).
- Time-extended corruption across Euler steps (corrupt-once-read-many is the default).
- Quantized KV caches (not enabled in the release model).

### 10.2 Target tensors

```
target = {
  f"layer{l}.key":   ctx.prompt_cache.key_cache[l],    # (B, H_kv, S_prefix, D_head), bf16
  f"layer{l}.value": ctx.prompt_cache.value_cache[l],  # (B, H_kv, S_prefix, D_head), bf16
  for l in range(num_text_layers)
}
```

- Qwen3-VL-8B text stack: **~36 layers × 2 = ~72 tensors**.
- Per-tensor shape with GQA: `(B, 8, S, 128)`. For `S ≈ 2000` prefix tokens, ~2–4 MB in BF16 each. Aggregate cache ≈ 300 MB.
- Access is direct (no `nn.Linear` wrapper, no FQN matching).

### 10.3 Gradient-through-expert ranking

The cache tensors are not leaves by default — they are the no-grad outputs of the VLM prefill. Toggle `requires_grad_` before the backward so grads land on them **without re-running the VLM**:

```python
for t in kv_targets.values():
    t.requires_grad_(True)

model.zero_grad(set_to_none=True)
loss = fm_one_step_loss(model, ctx)   # unchanged
loss.backward()

kv_grads = {n: t.grad.detach().clone().float() for n, t in kv_targets.items()}

for t in kv_targets.values():
    t.requires_grad_(False)
    t.grad = None
```

Backward stops at the expert boundary. Cost ≈ single expert backward (~1–2 s on H20), independent of cache size. `bit_grad_per_bit` is reused verbatim (same BF16 int16-view primitive — §3.5).

### 10.4 Injection-measure-restore

Mechanics identical to §3.4; target is a cache tensor, not `module.weight.data`:

```python
def measure_kv_flipped_loss(model, ctx, kv_tensor, flat_idx, bit, loss_fn):
    orig, flipped = bf16_flip_one(kv_tensor, flat_idx, bit)
    try:
        loss = loss_fn().item()
    finally:
        restore_one(kv_tensor, flat_idx, orig)
    return {...}   # same dict shape as measure_flipped_loss
```

**Crop invariant (load-bearing).** `DynamicCache.crop(prefill_seq_len)` truncates along the sequence dim only; it does **not** overwrite the prefix region (positions `0..prefill_seq_len-1`). A bit flipped at prefix position `t` therefore stays flipped across the next expert forward, and try/finally restore returns the cache to exactly its pre-trial state. Smoke test 10.5(c) verifies this before the main loop runs.

### 10.5 Smoke tests (before the main sweep)

| # | Check | Expected |
|---|---|---|
| (a) | Flip bit 0 (LSB mantissa) of a random coord in `layer0.key` → measure → restore | Small finite Δloss; baseline recovered to 1e-6 |
| (b) | Flip bit 14 (exponent MSB) of a random coord in `layer0.key` | `loss` is inf/nan; restore recovers baseline |
| (c) | Flip at prefix position `t = S_prefix // 2`, run `fm_one_step_loss` twice, restore → baseline | Baseline recovered — confirms `crop` doesn't leak prefix state across trials |
| (d) | `requires_grad_(True)` → `loss.backward()` → assert every `kv_tensor.grad is not None` | All ~72 tensors get grads; none `None` |
| (e) | Disable grad; run 100 no-grad main-loop trials | Memory steady — no graph accumulation |

Smoke tests go **before** the main loop (same convention as Phase 1 — see §5's rationale).

### 10.6 Runtime budget (reference)

Single H20, one scene:

| Phase | Count | Per-trial | Total |
|---|---|---|---|
| `build_fm_context` (Phase-1 reuse) | 1 | ~600 ms | <1 s |
| `compute_clean_kv_grads` | 1 | ~1–2 s | ~2 s |
| KV BFA main loop | 16 bits × 72 tensors × k=20 | ~30 ms | **~12 min** |

Across 4×H20 in parallel scenes: 4 complete sweeps in ~15 min — shorter than Phase 1's 65 min, because target count drops from ~200 Linears to ~72 cache tensors.

### 10.7 Code drop (~55 LOC added to `alpamayo1.5/notebooks/bfa_utils.py`)

```python
def collect_kv_cache_targets(ctx: FMContext) -> Dict[str, torch.Tensor]:
    """Return {name: tensor} for every K and V tensor in the VLM prefill cache."""
    out: Dict[str, torch.Tensor] = {}
    for l, (k, v) in enumerate(zip(ctx.prompt_cache.key_cache,
                                   ctx.prompt_cache.value_cache)):
        out[f"layer{l}.key"]   = k.contiguous()
        out[f"layer{l}.value"] = v.contiguous()
    return out


def compute_clean_kv_grads(
    model,
    ctx: FMContext,
    targets: Dict[str, torch.Tensor],
    loss_fn: Callable[[], torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """∂loss/∂K_l and ∂loss/∂V_l via backward-through-expert only.

    Sets requires_grad_(True) on each cache tensor, runs one FM backward,
    snapshots grads, then un-sets requires_grad. VLM is never re-run.
    """
    for t in targets.values():
        t.requires_grad_(True)
    try:
        model.zero_grad(set_to_none=True)
        loss_fn().backward()
        grads = {}
        for n, t in targets.items():
            assert t.grad is not None, f"no grad on {n}"
            grads[n] = t.grad.detach().clone().float()
    finally:
        for t in targets.values():
            t.requires_grad_(False)
            t.grad = None
        model.zero_grad(set_to_none=True)
    return grads


@torch.no_grad()
def measure_kv_flipped_loss(
    model,
    ctx: FMContext,
    kv_tensor: torch.Tensor,
    flat_idx: int,
    bit: int,
    loss_fn: Callable[[], torch.Tensor],
) -> Dict[str, float]:
    """Flip one bit in a cache tensor, run loss_fn, restore. Mirrors
    measure_flipped_loss but operates on an activation tensor."""
    orig, flipped = bf16_flip_one(kv_tensor, flat_idx, bit)
    try:
        loss = loss_fn().item()
        finite = float(torch.isfinite(torch.tensor(loss)).item())
    finally:
        restore_one(kv_tensor, flat_idx, orig)
    return {
        "flat_idx": int(flat_idx),
        "bit": int(bit),
        "orig_value": float(orig.float().item()),
        "flipped_value": float(flipped.float().item())
            if torch.isfinite(flipped.float()) else float("nan"),
        "post_loss": float(loss),
        "post_loss_finite": finite,
    }
```

Reused verbatim from Phase 1 (§3): `bf16_flip_one`, `restore_one`, `bit_grad_per_bit`, `topk_bitflip_coords`, `FMContext`, `build_fm_context`, `fm_one_step_loss`. `.contiguous()` in `collect_kv_cache_targets` is defensive (HF returns contiguous tensors today, but the bit-flip primitive requires it).

### 10.8 Notebook cells to add (to `bfa_demo.ipynb`)

Inserted between Phase-1 cell 7 (grad collection) and cell 8 (main BFA loop):

- **Cell 7b — KV targets + grads.**
  `kv_targets = bfa.collect_kv_cache_targets(ctx)` →
  `kv_grads = bfa.compute_clean_kv_grads(model, ctx, kv_targets, loss_fn=lambda: bfa.fm_one_step_loss(model, ctx, noise=fixed_noise))`.
- **Cell 8b — KV main loop.** Outer over `(name, tensor)` in `kv_targets`; middle over `bit in range(16)`; inner over `topk_bitflip_coords(tensor, kv_grads[name], bit, k=20)`. Writes `bfa_kv_results.json` alongside `bfa_results.json`.

### 10.9 Relationship to §7 cross-stage amplification

§7 proposes "inject into VLM weights, measure expert loss drift via the shared KV-cache". Phase 2a is the **activation-level twin**: inject into the cache *directly* — the exact medium through which §7's VLM-weight corruption would propagate. Running both decomposes sensitivity into (i) VLM weight → cache amplification and (ii) cache → expert reading, which matters for error-containment arguments (e.g., selective re-prefill vs weight ECC).

### 10.10 Status

- ✅ Feasibility analysis (this section).
- 🔜 Implement `collect_kv_cache_targets`, `compute_clean_kv_grads`, `measure_kv_flipped_loss` in `bfa_utils.py`.
- 🔜 Smoke tests (a)–(e) in notebook before committing to a full sweep.
- 🔜 First sweep on 4×H20 (4 scenes in parallel, ~15 min).
- 🔜 Expected output: `bfa_kv_results.json`, `bfa_kv_per_bit.png`, `bfa_kv_per_layer.png`.
