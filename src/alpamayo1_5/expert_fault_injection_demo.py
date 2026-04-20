# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone expert weight fault-injection demo for Alpamayo 1.5.

This script mirrors ``test_inference.py`` but injects a single bit flip into a
parameter inside ``model.expert`` between a clean baseline rollout and a faulty
rollout. The two runs use the same random seed so the reported difference is
primarily due to the injected fault rather than sampling variance.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5


DEFAULT_CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
DEFAULT_MODEL_ID = "nvidia/Alpamayo-1.5-10B"


@dataclass
class FaultSpec:
    """Concrete description of one injected fault."""

    param_name: str
    flat_index: int
    coord: tuple[int, ...]
    bit: int
    dtype: torch.dtype
    original_bits: int
    corrupted_bits: int
    original_value: float
    corrupted_value: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-id", default=DEFAULT_CLIP_ID, help="Clip ID to evaluate.")
    parser.add_argument(
        "--t0-us",
        type=int,
        default=5_100_000,
        help="Timestamp in microseconds used for dataset sampling.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="HuggingFace model id or local checkpoint path.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for inference, e.g. cuda or cpu.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Model dtype passed to from_pretrained.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed reused for both clean and faulty rollouts.",
    )
    parser.add_argument(
        "--bit",
        type=int,
        default=14,
        help="Bit position to flip in the selected parameter element.",
    )
    parser.add_argument(
        "--param-name",
        default=None,
        help=(
            "Exact expert parameter name to target, e.g. "
            "'model.layers.0.self_attn.q_proj.weight'. If omitted, choose randomly."
        ),
    )
    parser.add_argument(
        "--flat-index",
        type=int,
        default=None,
        help="Flat element index within the selected parameter. If omitted, choose randomly.",
    )
    parser.add_argument(
        "--module-substring",
        default=None,
        help="Optional substring filter applied to expert parameter names before selection.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.98,
        help="Top-p sampling parameter for VLM rollout.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature shared by clean and faulty runs.",
    )
    parser.add_argument(
        "--num-traj-samples",
        type=int,
        default=1,
        help="Number of trajectory samples to draw per rollout.",
    )
    parser.add_argument(
        "--max-generation-length",
        type=int,
        default=256,
        help="Maximum VLM generation length.",
    )
    return parser.parse_args()


def parse_torch_dtype(dtype_name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model_inputs(
    model: Alpamayo1_5,
    data: dict[str, Any],
    device: str | torch.device,
) -> dict[str, Any]:
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
    )
    processor = helper.get_processor(model.tokenizer)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    return helper.to_device(model_inputs, device)


def compute_metrics(pred_xyz: torch.Tensor, gt_future_xyz: torch.Tensor) -> dict[str, float]:
    gt_xy = gt_future_xyz.cpu()[0, 0, :, :2].numpy()  # (T, 2)
    pred_xy = pred_xyz.detach().cpu().numpy()[0, 0, :, :, :2]  # (K, T, 2)
    ade_all = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=-1).mean(axis=-1)
    fde_all = np.linalg.norm(pred_xy[:, -1, :] - gt_xy[-1, :][None, :], axis=-1)
    return {
        "minADE_m": float(ade_all.min()),
        "meanADE_m": float(ade_all.mean()),
        "minFDE_m": float(fde_all.min()),
        "meanFDE_m": float(fde_all.mean()),
    }


def flatten_reasoning(extra: dict[str, Any] | None) -> str:
    if not extra or "cot" not in extra:
        return ""
    value = extra["cot"]
    if isinstance(value, np.ndarray) and value.size > 0:
        return str(value.reshape(-1)[0])
    return ""


def run_rollout(
    model: Alpamayo1_5,
    model_inputs: dict[str, Any],
    gt_future_xyz: torch.Tensor,
    seed: int,
    dtype: torch.dtype,
    top_p: float,
    temperature: float,
    num_traj_samples: int,
    max_generation_length: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], dict[str, float]]:
    set_all_seeds(seed)
    model_device = next(model.parameters()).device
    if model_device.type == "cuda" and dtype in (torch.bfloat16, torch.float16):
        autocast_ctx = torch.autocast(device_type="cuda", dtype=dtype)
    else:
        autocast_ctx = nullcontext()
    with torch.no_grad():
        with autocast_ctx:
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=top_p,
                temperature=temperature,
                num_traj_samples=num_traj_samples,
                max_generation_length=max_generation_length,
                return_extra=True,
            )
    metrics = compute_metrics(pred_xyz, gt_future_xyz)
    return pred_xyz, pred_rot, extra, metrics


def get_storage_view(param: torch.nn.Parameter) -> tuple[torch.Tensor, int]:
    if param.dtype in (torch.bfloat16, torch.float16):
        return param.data.view(torch.uint16).reshape(-1), 16
    if param.dtype == torch.float32:
        return param.data.view(torch.uint32).reshape(-1), 32
    raise TypeError(f"Unsupported dtype for bit flip: {param.dtype}")


def validate_bit_index(bit: int, n_bits: int) -> None:
    if bit < 0 or bit >= n_bits:
        raise ValueError(f"Bit index {bit} is invalid for {n_bits}-bit storage.")


def select_expert_parameter(
    model: Alpamayo1_5,
    rng: random.Random,
    param_name: str | None = None,
    module_substring: str | None = None,
) -> tuple[str, torch.nn.Parameter]:
    candidates = []
    for name, param in model.expert.named_parameters():
        if not param.is_floating_point():
            continue
        if module_substring is not None and module_substring not in name:
            continue
        candidates.append((name, param))

    if not candidates:
        raise ValueError("No expert parameters matched the requested filters.")

    if param_name is None:
        return candidates[rng.randrange(len(candidates))]

    for name, param in candidates:
        if name == param_name:
            return name, param
    raise ValueError(f"Expert parameter '{param_name}' was not found.")


def unravel_index(flat_index: int, shape: torch.Size) -> tuple[int, ...]:
    return tuple(np.unravel_index(flat_index, tuple(shape)))


def inject_single_bit_flip(
    param_name: str,
    param: torch.nn.Parameter,
    bit: int,
    rng: random.Random,
    flat_index: int | None = None,
) -> FaultSpec:
    storage, n_bits = get_storage_view(param)
    validate_bit_index(bit, n_bits)

    if storage.numel() == 0:
        raise ValueError(f"Expert parameter '{param_name}' is empty.")

    if flat_index is None:
        flat_index = rng.randrange(storage.numel())
    if flat_index < 0 or flat_index >= storage.numel():
        raise ValueError(
            f"Flat index {flat_index} is out of range for parameter '{param_name}' "
            f"with {storage.numel()} elements."
        )

    mask = 1 << bit
    with torch.no_grad():
        original_bits = int(storage[flat_index].item())
        corrupted_bits = original_bits ^ mask

        flat_values = param.data.reshape(-1)
        original_value = float(flat_values[flat_index].float().item())
        storage[flat_index] = corrupted_bits
        corrupted_value = float(flat_values[flat_index].float().item())

    return FaultSpec(
        param_name=param_name,
        flat_index=flat_index,
        coord=unravel_index(flat_index, param.shape),
        bit=bit,
        dtype=param.dtype,
        original_bits=original_bits,
        corrupted_bits=corrupted_bits,
        original_value=original_value,
        corrupted_value=corrupted_value,
    )


def restore_fault(param: torch.nn.Parameter, spec: FaultSpec) -> None:
    storage, _ = get_storage_view(param)
    with torch.no_grad():
        storage[spec.flat_index] = spec.original_bits


def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.7g}"


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(title)
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")


def main() -> None:
    args = parse_args()
    dtype = parse_torch_dtype(args.dtype)
    device = torch.device(args.device)
    rng = random.Random(args.seed)

    print(f"Loading dataset for clip_id: {args.clip_id} at t0_us={args.t0_us}...")
    data = load_physical_aiavdataset(args.clip_id, t0_us=args.t0_us)
    print("Dataset loaded.")

    print(f"Loading model from {args.model_id} with dtype={args.dtype} on {device}...")
    model = Alpamayo1_5.from_pretrained(args.model_id, dtype=dtype).to(device)
    model.eval()

    model_inputs = make_model_inputs(model, data, device)

    print("\nRunning clean baseline rollout...")
    clean_pred_xyz, _, clean_extra, clean_metrics = run_rollout(
        model=model,
        model_inputs=model_inputs,
        gt_future_xyz=data["ego_future_xyz"],
        seed=args.seed,
        dtype=dtype,
        top_p=args.top_p,
        temperature=args.temperature,
        num_traj_samples=args.num_traj_samples,
        max_generation_length=args.max_generation_length,
    )
    print_metrics("Clean metrics:", clean_metrics)
    clean_cot = flatten_reasoning(clean_extra)
    if clean_cot:
        print("Clean CoT:")
        print(clean_cot)

    param_name, param = select_expert_parameter(
        model=model,
        rng=rng,
        param_name=args.param_name,
        module_substring=args.module_substring,
    )
    print("\nInjecting a single bit flip into model.expert...")
    spec = inject_single_bit_flip(
        param_name=param_name,
        param=param,
        bit=args.bit,
        rng=rng,
        flat_index=args.flat_index,
    )
    print(f"  parameter: {spec.param_name}")
    print(f"  coord: {spec.coord}")
    print(f"  flat_index: {spec.flat_index}")
    print(f"  dtype: {spec.dtype}")
    print(f"  flipped_bit: {spec.bit}")
    print(f"  original_bits: 0x{spec.original_bits:x}")
    print(f"  corrupted_bits: 0x{spec.corrupted_bits:x}")
    print(f"  original_value: {format_float(spec.original_value)}")
    print(f"  corrupted_value: {format_float(spec.corrupted_value)}")

    try:
        print("\nRunning faulty rollout with the same seed...")
        faulty_pred_xyz, _, faulty_extra, faulty_metrics = run_rollout(
            model=model,
            model_inputs=model_inputs,
            gt_future_xyz=data["ego_future_xyz"],
            seed=args.seed,
            dtype=dtype,
            top_p=args.top_p,
            temperature=args.temperature,
            num_traj_samples=args.num_traj_samples,
            max_generation_length=args.max_generation_length,
        )
    finally:
        restore_fault(param, spec)

    print_metrics("Faulty metrics:", faulty_metrics)

    print("Metric deltas:")
    for key in clean_metrics:
        delta = faulty_metrics[key] - clean_metrics[key]
        print(f"  d{key}: {delta:+.6f}")

    clean_xy = clean_pred_xyz.detach().cpu().numpy()[0, 0, 0, :, :2]
    faulty_xy = faulty_pred_xyz.detach().cpu().numpy()[0, 0, 0, :, :2]
    traj_delta = np.linalg.norm(faulty_xy - clean_xy, axis=-1)
    print(f"  max waypoint shift: {traj_delta.max():.6f} m")
    print(f"  mean waypoint shift: {traj_delta.mean():.6f} m")

    faulty_cot = flatten_reasoning(faulty_extra)
    if faulty_cot:
        print("Faulty CoT:")
        print(faulty_cot)
    if clean_cot or faulty_cot:
        print(f"CoT changed: {clean_cot != faulty_cot}")


if __name__ == "__main__":
    main()
