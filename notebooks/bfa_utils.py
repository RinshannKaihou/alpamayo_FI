"""BFA (Bit-Flip Attack) utilities for Alpamayo-v1.5.

Experimental code — functions are intentionally flat, small, and easy
to modify inline from a notebook. No classes, no registration.

Target dtype: BF16 (16-bit: 1 sign + 8 exponent + 7 mantissa).
"""

from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _torch_ckpt


def bf16_xor_bit_all(weight: torch.Tensor, bit: int) -> torch.Tensor:
    """Return a NEW bf16 tensor with `bit` XORed at every coordinate.

    Args:
        weight: bfloat16 tensor, any shape.
        bit: bit index in [0, 15]. 0 = LSB of mantissa, 15 = sign.
            Bits 7..14 = exponent, bit 14 = exponent MSB (catastrophic bit).

    Notes:
        - For values in (-2, 2) the exponent MSB is 0; flipping it
          can produce +/-inf, so callers must handle non-finite results.
        - We view the BF16 tensor as **int16** (not uint16) because
          ``bitwise_xor`` is not implemented for uint16 on CUDA. The
          bit pattern is identical; only the signed interpretation of
          the mask differs. The mask for bit 15 becomes -32768 (the
          signed representation of 0x8000).
    """
    assert weight.dtype == torch.bfloat16, f"expected bf16, got {weight.dtype}"
    assert 0 <= bit < 16
    w_i16 = weight.contiguous().view(torch.int16)
    # Python int -> signed int16: values >= 0x8000 wrap to negative via two's complement.
    raw = 1 << bit
    mask_val = raw if raw < 0x8000 else raw - 0x10000
    mask = torch.tensor(mask_val, dtype=torch.int16, device=weight.device)
    return (w_i16 ^ mask).view(torch.bfloat16).clone()


def bf16_flip_one(
    weight: torch.Tensor, flat_idx: int, bit: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flip a single bit at flat index `flat_idx` IN PLACE.

    Returns (orig_scalar, flipped_scalar) as 0-dim bf16 tensors so the
    caller can log them and restore.

    The caller is responsible for calling `restore_one` afterwards.
    """
    assert weight.dtype == torch.bfloat16
    assert 0 <= bit < 16
    assert weight.is_contiguous(), "bf16_flip_one requires a contiguous weight tensor"
    flat = weight.view(-1)
    orig = flat[flat_idx].detach().clone()
    # See bf16_xor_bit_all for the int16-view rationale.
    raw = 1 << bit
    mask_val = raw if raw < 0x8000 else raw - 0x10000
    mask = torch.tensor(mask_val, dtype=torch.int16, device=weight.device)
    flipped_i16 = orig.view(torch.int16) ^ mask
    flipped = flipped_i16.view(torch.bfloat16).clone()
    with torch.no_grad():
        flat[flat_idx] = flipped
    return orig, flipped


def restore_one(weight: torch.Tensor, flat_idx: int, orig: torch.Tensor) -> None:
    """Undo a previous `bf16_flip_one` by writing `orig` back."""
    with torch.no_grad():
        weight.view(-1)[flat_idx] = orig


def bit_grad_per_bit(
    weight_bf16: torch.Tensor,
    grad: torch.Tensor,
    bit: int,
    inf_sentinel: float = 1e30,
) -> torch.Tensor:
    """First-order estimate of loss change if `bit` is flipped at each coord.

    Returns a float32 tensor of the same shape as weight. Positive values =
    coordinates where flipping this bit is predicted to increase loss.

    Args:
        weight_bf16: the clean weight tensor (bf16).
        grad: the loss gradient w.r.t. the weight. Must be broadcastable
            to weight; computed in fp32 for numerical stability.
        bit: bit index in [0, 15].
        inf_sentinel: magnitude used to replace inf/nan deltas so the
            ranking still works. These are the catastrophic exponent flips.
    """
    flipped = bf16_xor_bit_all(weight_bf16, bit).to(torch.float32)
    clean_fp32 = weight_bf16.to(torch.float32)
    delta = flipped - clean_fp32                  # (*shape,), fp32
    grad_fp32 = grad.to(torch.float32)
    bg = grad_fp32 * delta
    # Replace non-finite (exponent-flip infinities) with a large finite
    # value whose sign matches sign(grad): if grad>0, flipping a bit to
    # +inf makes loss grow (positive bit_grad), and vice versa.
    nonfinite = ~torch.isfinite(bg)
    if nonfinite.any():
        bg = torch.where(
            nonfinite,
            torch.sign(grad_fp32) * inf_sentinel,
            bg,
        )
    return bg


def collect_target_linears(
    model: nn.Module,
    include_prefixes: Tuple[str, ...] = ("expert.", "action_in_proj.", "action_out_proj"),
) -> Dict[str, nn.Linear]:
    """Return {qualified_name: module} for Linear layers matching prefixes.

    Default targets the Alpamayo expert denoiser + action projections.
    The VLM is deliberately excluded (its backward needs FSDP).

    Edit `include_prefixes` to target different sub-trees.

    Note: ``"expert."`` and ``"action_in_proj."`` use trailing dots because
    they are container modules with sub-linears. ``"action_out_proj"`` has
    no dot because it is itself a plain ``nn.Linear`` at the root — matching
    ``name == p.rstrip(".")``. Do not "normalize" the asymmetry.
    """
    out: Dict[str, nn.Linear] = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if any(name.startswith(p) or name == p.rstrip(".") for p in include_prefixes):
            out[name] = mod
    return out


@dataclass
class FMContext:
    """Pre-computed VLM state reusable across many FM loss calls.

    Building this is the expensive part (~800 ms). After that, each
    fm_one_step_loss call is just one expert forward (~30 ms) + a small
    backward.
    """
    prompt_cache: object          # transformers DynamicCache
    prefill_seq_len: int
    position_ids: torch.Tensor    # (3, b_star, n_diffusion_tokens)
    attention_mask: torch.Tensor  # (b_star, 1, n_diff, KV)
    gt_action: torch.Tensor       # (b_star, n_waypoints, 2) bf16
    n_diffusion_tokens: int
    device: torch.device


@torch.no_grad()
def build_fm_context(model, model_inputs: Dict, gt_action: torch.Tensor) -> FMContext:
    """Run VLM prefill once; build cache + pos_ids + attn_mask for the expert.

    Mirrors the first half of
    `Alpamayo1_5.sample_trajectories_from_data_with_vlm_rollout` but
    WITHOUT autoregressive generation — we just want a KV-cache the
    expert can consume. We treat the input sequence itself as the
    prefix.

    Args:
        model: loaded Alpamayo1_5 in eval mode.
        model_inputs: dict with `tokenized_data`, `ego_history_xyz`, etc.
        gt_action: (B, n_waypoints, 2) bf16, typically from
            action_space.traj_to_action(ego_future_xyz, ...).
    """
    tokenized = model_inputs["tokenized_data"]
    input_ids = tokenized["input_ids"]
    # fuse history trajectory tokens (same as in sample_trajectories...)
    input_ids = model.fuse_traj_tokens(
        input_ids,
        {
            "ego_history_xyz": model_inputs["ego_history_xyz"],
            "ego_history_rot": model_inputs["ego_history_rot"],
        },
    )
    device = input_ids.device

    # VLM prefill — no generation, no logits needed
    vlm_out = model.vlm(
        input_ids=input_ids,
        attention_mask=tokenized.get("attention_mask"),
        image_grid_thw=tokenized.get("image_grid_thw"),
        pixel_values=tokenized.get("pixel_values"),
        use_cache=True,
        logits_to_keep=1,
    )
    prompt_cache = vlm_out.past_key_values
    prefill_seq_len = prompt_cache.get_seq_length()
    rope_deltas = model.vlm.model.rope_deltas

    b_star = input_ids.shape[0]
    n_diff = model.action_space.get_action_space_dims()[0]

    # offset = end of prefix (no EOS since we didn't generate)
    offset = torch.full((b_star,), prefill_seq_len, device=device, dtype=torch.long)
    prefix_mask = tokenized.get("attention_mask")
    position_ids, attention_mask = model._build_expert_pos_ids_and_attn_mask(
        offset=offset,
        rope_deltas=rope_deltas,
        kv_cache_seq_len=prefill_seq_len,
        n_diffusion_tokens=n_diff,
        b_star=b_star,
        device=device,
        prefix_mask=prefix_mask,
    )
    return FMContext(
        prompt_cache=prompt_cache,
        prefill_seq_len=prefill_seq_len,
        position_ids=position_ids,
        attention_mask=attention_mask,
        gt_action=gt_action.to(device=device).to(torch.bfloat16),
        n_diffusion_tokens=n_diff,
        device=device,
    )


def fm_one_step_loss(
    model,
    ctx: FMContext,
    t_val: float = 0.5,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Single-step flow-matching training loss. Differentiable w.r.t.
    expert + action_in_proj + action_out_proj weights.

    x_t = (1 - t) * noise + t * gt_action     (flow-matching interpolation)
    target_v = gt_action - noise              (straight-line vector field)
    pred_v = model.expert( action_in_proj(x_t, t) ) → action_out_proj
    loss = MSE(pred_v, target_v)

    Args:
        model: Alpamayo1_5 instance.
        ctx: result of build_fm_context.
        t_val: scalar in (0, 1). Defaults to 0.5.
        noise: optional pre-sampled noise for determinism across trials.
            Shape = gt_action.shape. If None, samples fresh.
    """
    gt = ctx.gt_action
    if noise is None:
        noise = torch.randn_like(gt)
    t = torch.full((gt.shape[0], *[1] * (gt.ndim - 1)), t_val,
                   device=gt.device, dtype=gt.dtype)

    x_t = (1.0 - t) * noise + t * gt          # bf16
    target_v = gt - noise                      # bf16

    # Project noisy action → expert embedding
    future_token_embeds = model.action_in_proj(x_t, t)
    if future_token_embeds.dim() == 2:
        future_token_embeds = future_token_embeds.view(
            gt.shape[0], ctx.n_diffusion_tokens, -1
        )

    forward_kwargs = {}
    if model.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False

    expert_out = model.expert(
        inputs_embeds=future_token_embeds,
        position_ids=ctx.position_ids,
        past_key_values=ctx.prompt_cache,
        attention_mask=ctx.attention_mask,
        use_cache=True,
        **forward_kwargs,
    )
    try:
        last_hidden = expert_out.last_hidden_state[:, -ctx.n_diffusion_tokens:]
        pred_v = model.action_out_proj(last_hidden).view_as(target_v)
        return F.mse_loss(pred_v.float(), target_v.float())
    finally:
        # Always crop — even on exception — so ctx is safe for the next trial.
        ctx.prompt_cache.crop(ctx.prefill_seq_len)


def differentiable_rollout(
    model,
    ctx: FMContext,
    fixed_noise: torch.Tensor,
    n_ode_steps: int = 4,
) -> torch.Tensor:
    """Differentiable multi-step Euler rollout reusing a pre-built FMContext.

    Mirrors the Euler loop in src/alpamayo1_5/diffusion/flow_matching.py:171-196
    but (a) does NOT use @torch.no_grad, (b) reuses ctx.prompt_cache instead
    of re-running the VLM, (c) takes fixed_noise so the rollout is a smooth
    deterministic function of weights.

    Args:
        ctx: from build_fm_context (no-grad version is fine; we only backprop
            through the expert + action projections, not the VLM).
        fixed_noise: (B, n_waypoints, action_dim) bf16 tensor — same shape as
            ctx.gt_action. Held constant across (clean, flipped) trials.
        n_ode_steps: number of Euler steps. Inference default is 10; 4 is a
            speed/fidelity tradeoff for the BFA gradient computation.

    Returns:
        Final action state (B, n_waypoints, action_dim), bf16. Decode with
        model.action_space.action_to_traj to get xyz waypoints.

    Cache discipline: each step does
        expert(... past_key_values=ctx.prompt_cache, use_cache=True)
        ctx.prompt_cache.crop(ctx.prefill_seq_len)
    so the cache is restored to prefix length after each step. The autograd
    graph keeps the saved K/V tensors alive even after crop drops the Python
    refs. Wrapped in try/finally so an exception still leaves the cache safe.
    """
    x = fixed_noise
    device = ctx.device
    n_diff = ctx.n_diffusion_tokens
    b_star = x.shape[0]
    time_steps = torch.linspace(0.0, 1.0, n_ode_steps + 1, device=device)

    forward_kwargs = {}
    if model.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False

    try:
        for i in range(n_ode_steps):
            dt = time_steps[i + 1] - time_steps[i]
            t_start = time_steps[i].view(1, *([1] * (x.ndim - 1))).expand(
                b_star, *([1] * (x.ndim - 1))
            ).to(x.dtype)

            future_token_embeds = model.action_in_proj(x, t_start)
            if future_token_embeds.dim() == 2:
                future_token_embeds = future_token_embeds.view(b_star, n_diff, -1)

            expert_out = model.expert(
                inputs_embeds=future_token_embeds,
                position_ids=ctx.position_ids,
                past_key_values=ctx.prompt_cache,
                attention_mask=ctx.attention_mask,
                use_cache=True,
                **forward_kwargs,
            )
            ctx.prompt_cache.crop(ctx.prefill_seq_len)

            last_hidden = expert_out.last_hidden_state[:, -n_diff:]
            v = model.action_out_proj(last_hidden).view_as(x)
            x = x + dt.to(x.dtype) * v
    finally:
        # Defensive: ensure cache length is the prefix length on exit even on error.
        if ctx.prompt_cache.get_seq_length() != ctx.prefill_seq_len:
            ctx.prompt_cache.crop(ctx.prefill_seq_len)

    return x


def proj_target_loss(
    model,
    ctx: FMContext,
    fixed_noise: torch.Tensor,
    u: torch.Tensor,
    history_xyz_last: torch.Tensor,
    history_rot_last: torch.Tensor,
    n_ode_steps: int = 4,
) -> torch.Tensor:
    """Scalar linear projection of the rolled-out trajectory.

    L = (u * pred_xy.float()).sum()

    pred_xy = action_to_traj(differentiable_rollout(...))[..., :2]

    Args:
        u: tensor broadcastable to pred_xy shape (B, n_wp, 2). For Hutchinson
            ranking, draw n_probe random unit vectors and call this fn n_probe
            times, averaging squared bit-grads downstream.
        history_xyz_last: (B, T_hist, 3). Pre-sliced last-step history.
            Mirror the pattern at src/alpamayo1_5/models/alpamayo1_5.py:373-378:
            `model_inputs["ego_history_xyz"][:, -1]` (no einops.repeat needed
            since we use B=1, n_samples_total=1 in the BFA setup).
        history_rot_last: (B, T_hist, 3, 3). Pre-sliced last-step rotation.
    """
    final_action = differentiable_rollout(model, ctx, fixed_noise, n_ode_steps=n_ode_steps)
    pred_xyz, _pred_rot = model.action_space.action_to_traj(
        final_action, history_xyz_last, history_rot_last
    )
    pred_xy = pred_xyz[..., :2]
    return (u.to(pred_xy.dtype) * pred_xy).float().sum()


def compute_clean_grads_proj(
    model: nn.Module,
    targets: Dict[str, nn.Linear],
    ctx: FMContext,
    fixed_noise: torch.Tensor,
    u: torch.Tensor,
    history_xyz_last: torch.Tensor,
    history_rot_last: torch.Tensor,
    n_ode_steps: int = 4,
) -> Dict[str, torch.Tensor]:
    """Backward through proj_target_loss, snapshot per-target fp32 grads.

    Drop-in shape-compatible with compute_clean_grads (bfa_utils.py:287),
    so downstream topk_bitflip_coords works unchanged.
    """
    model.zero_grad(set_to_none=True)
    loss = proj_target_loss(
        model, ctx, fixed_noise, u, history_xyz_last, history_rot_last, n_ode_steps=n_ode_steps
    )
    loss.backward()
    grads: Dict[str, torch.Tensor] = {}
    for name, mod in targets.items():
        assert mod.weight.grad is not None, f"no grad reached {name}"
        grads[name] = mod.weight.grad.detach().clone().float()
    model.zero_grad(set_to_none=True)
    return grads


def compute_clean_grads(
    model: nn.Module,
    targets: Dict[str, nn.Linear],
    loss_fn: Callable[[], torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Run `loss_fn()`, backward, clone grads for every target module.

    The model is expected to already have `requires_grad=True` on
    target weights (which is the default after `from_pretrained`).
    """
    model.zero_grad(set_to_none=True)
    loss = loss_fn()
    loss.backward()
    grads: Dict[str, torch.Tensor] = {}
    for name, mod in targets.items():
        assert mod.weight.grad is not None, (
            f"no grad reached {name}. Is it outside the expert subgraph?"
        )
        grads[name] = mod.weight.grad.detach().clone().float()
    model.zero_grad(set_to_none=True)
    return grads


def topk_bitflip_coords(
    weight_bf16: torch.Tensor,
    grad_fp32: torch.Tensor,
    bit: int,
    k: int,
) -> Tuple[torch.LongTensor, torch.Tensor]:
    """Return (flat_indices, bit_grad_values) for top-k LOSS-INCREASING
    coordinates at this bit position.

    Top-k is taken over `bit_grad_per_bit` descending — i.e., the
    coordinates where flipping `bit` is most predicted to raise loss.
    """
    bg = bit_grad_per_bit(weight_bf16, grad_fp32, bit)
    flat = bg.reshape(-1)
    k_eff = min(k, flat.numel())
    vals, idx = torch.topk(flat, k=k_eff, largest=True)
    return idx.cpu(), vals.cpu()


@torch.no_grad()
def measure_flipped_loss(
    model,
    ctx: FMContext,
    module: nn.Linear,
    flat_idx: int,
    bit: int,
    loss_fn: Callable[[], torch.Tensor],
) -> Dict[str, float]:
    """Flip one bit, run `loss_fn`, restore. Returns scalars.

    `loss_fn` is typically `lambda: fm_one_step_loss(model, ctx, noise=<fixed>)`
    with a FIXED noise so measurements are comparable across trials.
    """
    orig, flipped = bf16_flip_one(module.weight.data, flat_idx, bit)
    try:
        loss = loss_fn().item()
        is_finite = float(torch.isfinite(torch.tensor(loss)).item())
    finally:
        restore_one(module.weight.data, flat_idx, orig)
    return {
        "flat_idx": int(flat_idx),
        "bit": int(bit),
        "orig_value": float(orig.float().item()),
        "flipped_value": float(flipped.float().item())
            if torch.isfinite(flipped.float()) else float("nan"),
        "post_loss": float(loss),
        "post_loss_finite": is_finite,
    }


@torch.no_grad()
def output_deflection_norm(
    model,
    ctx: FMContext,
    pred_v_clean: torch.Tensor,
    t_val: float = 0.5,
    noise: torch.Tensor | None = None,
) -> float:
    """L2 norm of pred_v difference vs a precomputed clean reference.

    Reuses the same forward path as fm_one_step_loss but returns
    ‖pred_v_current − pred_v_clean‖ instead of MSE-against-gt. Use
    inside measure_*_flipped_loss to log a baseline-deflection column.

    pred_v_clean: snapshot from a clean run with the same fixed (t_val, noise).
    """
    gt = ctx.gt_action
    if noise is None:
        noise = torch.randn_like(gt)
    t = torch.full((gt.shape[0], *[1] * (gt.ndim - 1)), t_val, device=gt.device, dtype=gt.dtype)
    x_t = (1.0 - t) * noise + t * gt

    future_token_embeds = model.action_in_proj(x_t, t)
    if future_token_embeds.dim() == 2:
        future_token_embeds = future_token_embeds.view(gt.shape[0], ctx.n_diffusion_tokens, -1)

    forward_kwargs = {}
    if model.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False

    expert_out = model.expert(
        inputs_embeds=future_token_embeds,
        position_ids=ctx.position_ids,
        past_key_values=ctx.prompt_cache,
        attention_mask=ctx.attention_mask,
        use_cache=True,
        **forward_kwargs,
    )
    try:
        last_hidden = expert_out.last_hidden_state[:, -ctx.n_diffusion_tokens:]
        pred_v = model.action_out_proj(last_hidden).view_as(gt)
        return (pred_v.float() - pred_v_clean.float()).norm().item()
    finally:
        ctx.prompt_cache.crop(ctx.prefill_seq_len)


# --------------------------------------------------------------------------- #
# Phase 2a: KV-cache fault injection
# --------------------------------------------------------------------------- #
#
# Injection target is the VLM prefill K/V cache (HF DynamicCache) rather than
# weights. See docs/sdc/alpamayo_bfa_experiments.md §10 for the spec.
#
# Why we store (layer_idx, kind) specs instead of tensor references:
# the expert forward calls `torch.cat(old, new)` to extend the cache, which
# REPLACES ctx.prompt_cache.key_cache[l] with a new tensor backed by new
# storage. The subsequent `crop(prefill_seq_len)` slices that new storage
# back to prefix length — values are preserved, but object identity is not.
# Any Python reference saved before the forward becomes stale. Re-fetching
# inside each trial via layer_idx keeps us pointed at the live tensor.


def get_kv_tensor(ctx: FMContext, layer_idx: int, kind: str) -> torch.Tensor:
    """Look up the current K or V tensor for `layer_idx` in the prefill cache."""
    c = ctx.prompt_cache
    if kind == "key":
        return c.key_cache[layer_idx]
    if kind == "value":
        return c.value_cache[layer_idx]
    raise ValueError(f"kind must be 'key' or 'value', got {kind!r}")


def collect_kv_cache_targets(ctx: FMContext) -> Dict[str, Tuple[int, str]]:
    """Enumerate every K and V tensor in the prefill cache as a (layer_idx, kind)
    spec. For Qwen3-VL-8B text stack that's ~36 layers × 2 = ~72 targets.
    """
    n_layers = len(ctx.prompt_cache.key_cache)
    out: Dict[str, Tuple[int, str]] = {}
    for l in range(n_layers):
        out[f"layer{l}.key"]   = (l, "key")
        out[f"layer{l}.value"] = (l, "value")
    return out


def compute_clean_kv_grads(
    model,
    ctx: FMContext,
    targets: Dict[str, Tuple[int, str]],
    loss_fn: Callable[[], torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """∂loss/∂K_l and ∂loss/∂V_l via one expert-only backward pass.

    Toggles requires_grad_(True) on the *current* cache tensors, calls
    loss_fn().backward(), snapshots grads into an fp32 dict, then restores
    requires_grad_(False) and clears .grad. The VLM is not re-run — backward
    stops at the expert boundary where the cache is consumed.

    Returned grads are indexed by the same names as `targets`. Shape matches
    the cache tensor: (B, H_kv, S_prefix, D_head).
    """
    # Snapshot CURRENT tensor references; these are what the expert reads.
    live = {name: get_kv_tensor(ctx, l, k) for name, (l, k) in targets.items()}
    for t in live.values():
        t.requires_grad_(True)
    try:
        model.zero_grad(set_to_none=True)
        loss_fn().backward()
        grads: Dict[str, torch.Tensor] = {}
        for n, t in live.items():
            assert t.grad is not None, (
                f"no grad on {n}. Did the expert actually read this cache entry? "
                f"Check that loss_fn uses ctx.prompt_cache."
            )
            grads[n] = t.grad.detach().clone().float()
    finally:
        for t in live.values():
            t.requires_grad_(False)
            t.grad = None
        model.zero_grad(set_to_none=True)
    return grads


@torch.no_grad()
def measure_kv_flipped_loss(
    model,
    ctx: FMContext,
    layer_idx: int,
    kind: str,
    flat_idx: int,
    bit: int,
    loss_fn: Callable[[], torch.Tensor],
) -> Dict[str, float]:
    """Flip one bit in ctx.prompt_cache[layer_idx][kind] at flat_idx, run
    loss_fn, restore. Mirrors `measure_flipped_loss` but operates on a
    KV-cache activation tensor.

    Re-acquires the tensor reference both for flip and restore — see the
    module-level comment for why object identity is not stable across
    loss_fn calls.
    """
    t = get_kv_tensor(ctx, layer_idx, kind)
    orig, flipped = bf16_flip_one(t, flat_idx, bit)
    try:
        loss = loss_fn().item()
        finite = float(torch.isfinite(torch.tensor(loss)).item())
    finally:
        # Re-fetch: crop may have swapped the list entry during loss_fn.
        restore_one(get_kv_tensor(ctx, layer_idx, kind), flat_idx, orig)
    return {
        "flat_idx": int(flat_idx),
        "bit": int(bit),
        "layer_idx": int(layer_idx),
        "kind": kind,
        "orig_value": float(orig.float().item()),
        "flipped_value": float(flipped.float().item())
            if torch.isfinite(flipped.float()) else float("nan"),
        "post_loss": float(loss),
        "post_loss_finite": finite,
    }


# --------------------------------------------------------------------------- #
# Rollout + metrics helpers (shared by demo notebooks)
# --------------------------------------------------------------------------- #
#
# Thin wrappers around `sample_trajectories_from_data_with_vlm_rollout` and
# ADE/FDE, factored out so the single-flip CoT/trajectory demo notebook stays
# short. Seeding convention matches inference_cam_num.ipynb: seed on CUDA
# before every rollout so sampling noise is identical across conditions.


def run_rollout(
    model,
    model_inputs: Dict[str, Any],
    device: torch.device | str | int,
    seed: int = 42,
    **roll_kwargs: Any,
) -> Dict[str, Any]:
    """Seeded VLM + expert rollout. Returns {pred_xyz (cpu), pred_rot (cpu), cot}.

    `roll_kwargs` are forwarded to
    `model.sample_trajectories_from_data_with_vlm_rollout` — typically
    `top_p`, `temperature`, `num_traj_samples`, `max_generation_length`.
    `return_extra=True` is always set so CoT text is captured.

    Args:
        device: REQUIRED — GPU to scope the CUDA RNG seed to. On a shared
            multi-GPU server, do not rely on `cuda` defaults; pass an
            explicit device id or ``torch.device``. The autocast dtype is
            derived from this device's type so ``cuda:N`` correctly selects
            CUDA autocast policy.
        seed: CUDA RNG seed for this rollout. Set inside the function so
            back-to-back calls are independent of caller-side RNG state.
    """
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    # Scope the seed to the chosen device only — manual_seed_all would touch
    # every visible CUDA device, which we want to avoid on shared hardware.
    with torch.cuda.device(dev):
        torch.cuda.manual_seed(seed)
    # autocast device_type ("cuda" / "cpu") is policy-only; the specific
    # GPU is still determined by tensor placement on `dev`.
    with torch.autocast(dev.type, dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            return_extra=True,
            **roll_kwargs,
        )
    if extra is not None and "cot" in extra:
        arr = np.asarray(extra["cot"])
        cot = str(arr.reshape(-1)[0]) if arr.size > 0 else ""
    else:
        cot = ""
    return {
        "pred_xyz": pred_xyz.detach().cpu(),
        "pred_rot": pred_rot.detach().cpu() if pred_rot is not None else None,
        "cot": cot,
    }


def compute_traj_metrics(
    pred_xy_all: np.ndarray,
    gt_xy: np.ndarray,
) -> Dict[str, float]:
    """Standard minADE / meanADE / minFDE / meanFDE from (K, T, 2) predictions
    against a (T, 2) ground-truth path.

    Samples containing any NaN waypoint are filtered before computing the
    metrics — a catastrophic-bit flip can push some rollout samples into
    non-finite territory while leaving others intact. `n_finite` reports
    how many usable samples remained.
    """
    mask = ~np.isnan(pred_xy_all.reshape(pred_xy_all.shape[0], -1)).any(axis=1)
    pred_xy = pred_xy_all[mask]
    if pred_xy.size == 0:
        return {
            "n_finite": 0,
            "minADE_m": float("nan"),
            "meanADE_m": float("nan"),
            "minFDE_m": float("nan"),
            "meanFDE_m": float("nan"),
        }
    ade = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=-1).mean(axis=-1)
    fde = np.linalg.norm(pred_xy[:, -1, :] - gt_xy[-1, :][None, :], axis=-1)
    return {
        "n_finite": int(pred_xy.shape[0]),
        "minADE_m": float(ade.min()),
        "meanADE_m": float(ade.mean()),
        "minFDE_m": float(fde.min()),
        "meanFDE_m": float(fde.mean()),
    }


# --------------------------------------------------------------------------- #
# Phase 3: VLM-target weight BFA + ADE cascade
# --------------------------------------------------------------------------- #
#
# Two ranking variants:
#   (3a) gradient-guided — needs backward through the 8B VLM. We run a single
#        VLM backward under activation checkpointing so peak memory stays
#        under ~80 GB on a 96 GB H20. After the clean-grads snapshot the
#        with-grad context is dropped and the bit-flip loop reuses the
#        cheap no-grad FMContext, restoring weights between trials so the
#        cached K/V never goes stale.
#   (3b) random + re-prefill — no gradients. Each trial flips, REBUILDS the
#        FMContext (so the cached K/V reflects the flipped weight), measures,
#        then restores. ~22× slower per trial than (3a), so the demo
#        subsamples modules + coords.
#
# Plus a cross-cutting cascade that replays the top-N FM-loss flips through
# full-trajectory rollout to get ADE/FDE — the actual safety metric that
# fm_one_step_loss only proxies.


def collect_target_linears_vlm(
    model: nn.Module,
    include_prefixes: Tuple[str, ...] = ("vlm.model.language_model.layers.",),
) -> Dict[str, nn.Linear]:
    """Linear collector restricted to the VLM text decoder by default.

    Skips ``vlm.model.visual.*`` (image encoder) and ``vlm.lm_head`` —
    those don't flow into action regression through the cached K/V.
    Pass a wider tuple to include them. ~36 layers × 7 linears ≈ 250
    modules for Qwen3-VL-8B.

    Path note: in current HF transformers the Qwen3-VL text decoder lives at
    ``vlm.model.language_model.layers.*`` (the outer ``Qwen3VLModel`` was
    split into ``visual`` + ``language_model`` submodules). Older snapshots
    used ``vlm.model.layers.*`` directly.
    """
    out: Dict[str, nn.Linear] = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if any(name.startswith(p) or name == p.rstrip(".") for p in include_prefixes):
            out[name] = mod
    return out


@contextlib.contextmanager
def _checkpoint_vlm_layers(model):
    """Wrap each ``model.vlm.model.language_model.layers[i].forward`` in
    ``torch.utils.checkpoint.checkpoint`` for the duration of the with-block.

    We monkey-patch the bound method on each layer instance instead of using
    HF's ``gradient_checkpointing_enable()`` because the latter is gated on
    ``self.training`` in some transformers versions, and we are in eval mode.
    Restored on exit even if the block raises.

    Path note: see ``collect_target_linears_vlm`` for why this is
    ``language_model.layers`` rather than just ``layers``.
    """
    layers = model.vlm.model.language_model.layers
    originals: List[Callable] = []
    for layer in layers:
        orig = layer.forward
        originals.append(orig)

        def make(o):
            def wrapper(*args, **kwargs):
                return _torch_ckpt.checkpoint(o, *args, use_reentrant=False, **kwargs)
            return wrapper

        layer.forward = make(orig)
    try:
        yield
    finally:
        for layer, orig in zip(layers, originals):
            layer.forward = orig


def build_fm_context_with_grad(
    model,
    model_inputs: Dict,
    gt_action: torch.Tensor,
    use_checkpoint: bool = True,
) -> FMContext:
    """Like ``build_fm_context`` (no-grad version) but lets gradients flow
    back to VLM weights. Used once at the start of Phase 3a to compute
    ``∂loss/∂vlm_weights``; afterwards the cheap no-grad context is rebuilt
    and reused for the per-trial measurement loop.

    With ``use_checkpoint=True``, every VLM transformer layer is wrapped in
    ``torch.utils.checkpoint`` so peak activation memory stays under ~40 GB
    even at S≈2000 prefix tokens. Without it the backward can OOM on a
    busy 96 GB H20.
    """
    tokenized = model_inputs["tokenized_data"]
    input_ids = tokenized["input_ids"]
    input_ids = model.fuse_traj_tokens(
        input_ids,
        {
            "ego_history_xyz": model_inputs["ego_history_xyz"],
            "ego_history_rot": model_inputs["ego_history_rot"],
        },
    )
    device = input_ids.device

    ckpt_ctx = _checkpoint_vlm_layers(model) if use_checkpoint else contextlib.nullcontext()
    with ckpt_ctx:
        vlm_out = model.vlm(
            input_ids=input_ids,
            attention_mask=tokenized.get("attention_mask"),
            image_grid_thw=tokenized.get("image_grid_thw"),
            pixel_values=tokenized.get("pixel_values"),
            use_cache=True,
            logits_to_keep=1,
        )
    prompt_cache = vlm_out.past_key_values
    prefill_seq_len = prompt_cache.get_seq_length()
    rope_deltas = model.vlm.model.rope_deltas

    b_star = input_ids.shape[0]
    n_diff = model.action_space.get_action_space_dims()[0]

    offset = torch.full((b_star,), prefill_seq_len, device=device, dtype=torch.long)
    prefix_mask = tokenized.get("attention_mask")
    position_ids, attention_mask = model._build_expert_pos_ids_and_attn_mask(
        offset=offset,
        rope_deltas=rope_deltas,
        kv_cache_seq_len=prefill_seq_len,
        n_diffusion_tokens=n_diff,
        b_star=b_star,
        device=device,
        prefix_mask=prefix_mask,
    )
    return FMContext(
        prompt_cache=prompt_cache,
        prefill_seq_len=prefill_seq_len,
        position_ids=position_ids,
        attention_mask=attention_mask,
        gt_action=gt_action.to(device=device).to(torch.bfloat16),
        n_diffusion_tokens=n_diff,
        device=device,
    )


def compute_clean_grads_vlm(
    model: nn.Module,
    targets: Dict[str, nn.Linear],
    model_inputs: Dict,
    gt_action: torch.Tensor,
    device: torch.device | str | int,
    t_val: float = 0.5,
    noise: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """End-to-end VLM-grad pipeline. Builds a with-grad FMContext, runs one
    FM one-step backward, snapshots fp32 grads for every target Linear, then
    drops the with-grad context so its ~25-40 GB activation footprint is
    GC'd before the bit-flip loop starts.

    Args:
        device: REQUIRED — GPU on which the autocast policy is selected and
            ``empty_cache`` is scoped. Avoids relying on ``cuda`` defaults
            on shared multi-GPU servers.

    Returns the same dict shape as ``compute_clean_grads`` so downstream
    ``topk_bitflip_coords`` works unchanged.
    """
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    model.zero_grad(set_to_none=True)
    with torch.autocast(dev.type, dtype=torch.bfloat16):
        ctx_grad = build_fm_context_with_grad(model, model_inputs, gt_action)
        loss = fm_one_step_loss(model, ctx_grad, t_val=t_val, noise=noise)
    # Reset the prompt cache to length 0 before backward so per-layer
    # checkpoint recomputation sees an empty cache (matching the state the
    # checkpointed VLM forward originally saw). Without this, each layer's
    # `past_key_values.update()` does cat([cached_prefix, new_K]) on recompute
    # but cat([empty, new_K]) on the original forward — different shapes flow
    # into attention_interface, autograd's saved-tensor signature diverges,
    # and use_reentrant=False raises CheckpointError. The autograd-saved K/V
    # tensors that the action expert read are kept alive by the engine itself;
    # crop(0) only drops the cache's external Python references.
    ctx_grad.prompt_cache.crop(0)
    loss.backward()

    grads: Dict[str, torch.Tensor] = {}
    unreached: List[str] = []
    for name, mod in targets.items():
        if mod.weight.grad is None:
            unreached.append(name)
            continue
        grads[name] = mod.weight.grad.detach().clone().float()
    if unreached:
        # Gradient enters the VLM only through the K/V cache (the only thing the
        # action expert reads). Weights whose output doesn't reach a cache slot
        # have None grad — most commonly the LAST decoder layer's
        # mlp.{gate,up,down}_proj and self_attn.{q,o}_proj, since their outputs
        # flow to model.norm → lm_head, which fm_one_step_loss never touches.
        # First-order bit-flip impact is zero for these, so we drop them rather
        # than waste trial budget; pair them with the random baseline instead.
        # Caller should iterate `grads.keys()`, not `targets.keys()`.
        import warnings
        head = ", ".join(unreached[:5]) + (" …" if len(unreached) > 5 else "")
        warnings.warn(
            f"compute_clean_grads_vlm: {len(unreached)}/{len(targets)} target "
            f"Linear(s) had no gradient (graph dead-ends w.r.t. the FM loss). "
            f"Dropped from returned dict. Examples: {head}",
            stacklevel=2,
        )
    model.zero_grad(set_to_none=True)
    # Drop the with-grad context so its activation buffers free. empty_cache
    # is process-scoped but we wrap it in torch.cuda.device(dev) anyway so
    # caller intent is explicit and visible in any future profiling.
    del ctx_grad
    with torch.cuda.device(dev):
        torch.cuda.empty_cache()
    return grads


def random_topk_coords(
    weight: torch.Tensor,
    bit: int,
    k: int,
    rng: int | np.random.Generator = 0,
) -> Tuple[torch.LongTensor, torch.Tensor]:
    """Random-coord baseline. Drop-in replacement for ``topk_bitflip_coords``
    when gradients aren't available (Phase 3b).

    Returns (flat_indices, NaN-filled values) — the NaN channel lets
    downstream code distinguish ranked vs. random rows in the saved results.
    The ``bit`` parameter is unused (signature parity with the gradient-guided
    function); coords are sampled uniformly per call.
    """
    del bit  # signature parity only
    g = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    n = weight.numel()
    k_eff = min(k, n)
    idx = torch.from_numpy(g.choice(n, size=k_eff, replace=False).astype(np.int64))
    vals = torch.full((k_eff,), float("nan"), dtype=torch.float32)
    return idx, vals


@torch.no_grad()
def measure_vlm_flipped_loss_reprefill(
    model,
    model_inputs: Dict,
    gt_action: torch.Tensor,
    module: nn.Linear,
    flat_idx: int,
    bit: int,
    noise: torch.Tensor,
    device: torch.device | str | int,
    t_val: float = 0.5,
) -> Dict[str, float]:
    """Phase 3b per-trial measurement.

    The cached K/V from a prior ``build_fm_context`` does NOT reflect a flip
    on a VLM weight, so we rebuild the FMContext inside the trial. Cost
    breakdown (single H20, S≈2000): ~600 ms VLM prefill + ~30 ms FM loss
    ≈ 650 ms/trial — ~22× the cache-stable Phase 1 cost.

    Args:
        device: REQUIRED — autocast policy is selected from this device's
            type. The actual GPU used is wherever ``model`` and
            ``model_inputs`` already live; passing ``device`` ensures the
            autocast device_type is consistent with that placement on shared
            multi-GPU servers.

    Mirrors ``measure_flipped_loss`` (line 326) for return-dict shape.
    """
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    orig, flipped = bf16_flip_one(module.weight.data, flat_idx, bit)
    try:
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            ctx = build_fm_context(model, model_inputs, gt_action)
            loss = fm_one_step_loss(model, ctx, t_val=t_val, noise=noise).item()
        is_finite = float(torch.isfinite(torch.tensor(loss)).item())
    finally:
        restore_one(module.weight.data, flat_idx, orig)
    return {
        "flat_idx": int(flat_idx),
        "bit": int(bit),
        "orig_value": float(orig.float().item()),
        "flipped_value": float(flipped.float().item())
            if torch.isfinite(flipped.float()) else float("nan"),
        "post_loss": float(loss),
        "post_loss_finite": is_finite,
    }


def cascade_top_n_to_rollout(
    results_df: pd.DataFrame,
    model,
    model_inputs: Dict[str, Any],
    gt_xy_traj: np.ndarray,
    target_lookup: Dict[str, nn.Linear],
    device: torch.device | str | int,
    top_n: int = 50,
    seed: int = 42,
    rollout_kwargs: Dict[str, Any] | None = None,
    name_col: str = "module",
) -> pd.DataFrame:
    """Replay top-N highest-Δloss WEIGHT flips through full-trajectory rollout.

    For each row (sorted by ``post_loss`` desc, finite only):
      1. ``bf16_flip_one(module.weight.data, flat_idx, bit)``
      2. ``run_rollout(model, model_inputs, seed)``
      3. ``compute_traj_metrics(pred_xy, gt_xy_traj)``
      4. ``restore_one(...)`` in finally

    Cache flips (Phase 2a) are filtered out: their FM-loss-via-cache attack
    doesn't naturally compose with a closed-loop rollout that rebuilds cache
    per inference. A cache-aware cascade would patch the inference loop and
    is intentionally out of scope.

    Args:
        results_df: parquet/csv-loaded BFA results with at minimum columns
            [name_col, flat_idx, bit, post_loss, post_loss_finite].
        target_lookup: maps qualified module name -> nn.Linear. Build with
            ``{name: model.get_submodule(name) for name in df[name_col].unique()}``.
        gt_xy_traj: (T, 2) ground-truth waypoints in the same frame as
            the rollout's pred_xyz output.
        device: REQUIRED — forwarded to ``run_rollout`` so the per-flip
            rollout's CUDA RNG seeding and autocast policy are scoped to
            the user's allocated GPU rather than ``cuda:0``.
        rollout_kwargs: forwarded to ``run_rollout`` (top_p, temperature,
            num_traj_samples, ...).
        name_col: column name holding the module qualified name. Existing
            Phase 1 / 3a / 3b notebooks use ``"module"``; future schemas may
            use ``"target_name"``.

    Returns:
        DataFrame with all original columns + [n_finite, minADE_m, meanADE_m,
        minFDE_m, meanFDE_m] appended for cascaded rows. Skipped rows
        (e.g. cache flips) are not included.
    """
    rollout_kwargs = rollout_kwargs or {}
    finite = results_df[results_df["post_loss_finite"] == 1.0]
    top = finite.nlargest(top_n, "post_loss")

    out_rows: List[Dict[str, Any]] = []
    for _, row in top.iterrows():
        name = row[name_col]
        target = target_lookup.get(name)
        if not isinstance(target, nn.Linear):
            continue  # cache flips deferred

        flat_idx = int(row["flat_idx"])
        bit = int(row["bit"])
        orig, _ = bf16_flip_one(target.weight.data, flat_idx, bit)
        try:
            roll = run_rollout(model, model_inputs, device=device, seed=seed, **rollout_kwargs)
            pred_xyz = roll["pred_xyz"].numpy()
            # Squeeze leading batch dim if present: rollout may return (B,K,T,3)
            # or (K,T,3) depending on whether the data was already batched.
            if pred_xyz.ndim == 4:
                pred_xyz = pred_xyz[0]
            pred_xy = pred_xyz[..., :2]
            metrics = compute_traj_metrics(pred_xy, gt_xy_traj)
        finally:
            restore_one(target.weight.data, flat_idx, orig)

        merged = row.to_dict()
        merged.update(metrics)
        out_rows.append(merged)

    return pd.DataFrame(out_rows)
