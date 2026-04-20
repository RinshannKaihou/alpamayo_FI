# PyTorch SDC Weight Fault Injection — Methodology Summary

*Context from online Claude.ai project conversations. Reference for local Claude Code agents.*

---

## 1. Theoretical Background

### BF16 Bit-Flip Vulnerability (1 sign + 8 exponent + 7 mantissa, bias=127)

| Bits | Field | Single-Flip Impact |
|---|---|---|
| 15 | Sign | Negates value |
| 14 | Exp MSB | ~10³⁸ magnitude change — **only catastrophic bit for \|x\| < 2** |
| 13–7 | Exp (non-MSB) | For \|x\| < 2, result stays in [-2,2] or becomes tiny — harmless |
| 6–0 | Mantissa | Bounded relative error ≤50% (bit 6) to ≤0.78% (bit 0) |

For values in [-2,2], biased exponent ≤ 127 = `01111111` so bit 14 is always 0. Non-MSB exponent flips cannot push E' beyond 127. **93.75% of single-bit flips on \|x\| < 2 are harmless.**

### FP32: Same exponent structure. [1,2) is the unique danger zone — biased exp = `01111111`, one MSB flip → 0xFF → inf/NaN. Values \|x\| < 1 have exp ≤ `01111110` (two zero bits) — inf/NaN structurally impossible from single flip.

### INT8 Two's Complement: Max perturbation bounded at ±128 (MSB). No inf/NaN. But per-channel FP32 scale factors are single points of failure — exponent flip on scale broadcasts ~10³⁸ error across entire channel.

### ViT vs CNN

- **Attention V path** is the catastrophic failure source (linear propagation, no normalization). Softmax clips Q/K corruption to [0,1] in forward inference.
- **MLP/FFN** has 0% inf/NaN rate (intermediates \|x\| < 1) but produces silent extreme-finite errors through two-stage FC amplification. More dangerous for SDC because it bypasses simple inf/NaN detection.
- First encoder layer is most sensitive. GELU lacks ReLU's negative-value truncation.

---

## 2. Bit-Gradient Computation

### INT8 (Two's Complement)

```python
def weight_grad_to_bit_grad_int8(weight_int8, weight_grad):
    # Contributions: [1, 2, 4, 8, 16, 32, 64, -128]
    # bits = ((weight.view(uint8) >> bit_indices) & 1).float()
    # delta_w = (1 - 2*bits) * contributions
    # bit_grad = weight_grad.unsqueeze(-1) * delta_w
    # Returns: (*weight.shape, 8)
```

### IEEE 754 Float (FP32/FP16)

```python
def weight_grad_to_bit_grad_fp(weight, weight_grad):
    # For each bit i: delta_w = (value_with_bit_i_flipped - original) via int-view XOR
    # bit_grad = weight_grad.unsqueeze(-1) * delta_w
    # Returns: (*weight.shape, 32) for FP32, (*weight.shape, 16) for FP16
```

---

## 3. Experiment Types

### 3.1 Gradient-Guided BFA (Weight)

**Flow (per bit position):**

1. Load fresh **target model** + fresh **surrogate model**.
2. Surrogate: `forward → loss.backward()` → obtain `weight.grad` for all target modules.
3. Transform: `weight.grad` → `bit_grad` via the functions above.
4. Rank: `bit_grad.view(-1, n_bits)[:, bit].argsort(descending=True)[:top_k]` → top-k most sensitive coordinates.
5. For each coordinate on target model: inject flip → measure loss → restore clean weight.

**FP32 ResNet-50:**
```python
for bit in range(32):
    resnet = resnet50(weights=...).to(device).eval()
    resnet_sur = resnet50(weights=...).to(device).eval()
    resnet_sur.zero_grad(); loss.backward()

    for name, module in resnet.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            bit_grad, _ = weight_grad_to_bit_grad_fp(module_sur.weight, module_sur.weight.grad)
            indices = bit_grad.view(-1,32)[:,bit].argsort(descending=True)[:100]
            # For each index: flip_bit_in_float(val, bit), measure loss, restore
```

**INT8 Fake-Quant ResNet-50:**
```python
# Quantization: scale = module.weight_fake_quant.scale
# w_int8 = round(W / scale), clamp [-128, 127]
# Gradient transform: grad_int8 = grad_fp32 * scale
# Injection: XOR bit in INT8 → dequantize → substitute FP32 weight
# Filter: isinstance(module, (Conv2d, Linear)) and hasattr(module, 'weight_fake_quant')
```

**INT8 BnB-style ViT:**
```python
# BnB symmetric quantization: scale[i] = max(|W[i,:]|) / 127, zero_point = 0
# grad_int8 = grad_fp32 * scale
# Module filter: nn.Linear only (ViT has no Conv2d in encoder)
# Includes validate_fake_quant_vs_bnb() correctness check
```

### 3.2 Random Baseline (Weight)

Same injection/measurement loop but replace gradient-ranked selection with uniform random:

```python
# No surrogate model, no backward pass
k = min(1000, num_elements)
flat_indices = torch.randperm(num_elements)[:k]
indices = torch.unravel_index(flat_indices, weight_int8.shape)
# For each index: inject, measure, restore
```

### 3.3 Multi-Bit Byte-Aligned (Weight)

Models Sullivan et al. finding that ~75% of HBM2 multi-bit errors are byte-aligned.

```python
def make_byte_aligned_flip_mask(shape, coord, target_byte, num_flips, dtype, device, rng=None):
    # Picks num_flips random bits within target_byte (0-indexed from LSB)
    # byte 0: bits [0..7] (mantissa for BF16)
    # byte 1: bits [8..15] (sign + exponent for BF16)
    # Builds int16/int32 XOR mask with value_mask at coord, zeros elsewhere
```

BFA ranking for byte targets: sum `abs(bit_grad)` over the 8 bits in `target_byte` instead of indexing a single bit column.

---

## 4. Injection Mechanics

```python
# FP32 single-bit flip
def flip_bit_in_float(value, bit):
    # struct.pack('f', value) → XOR byte → struct.unpack

# INT8 fake-quant injection
def inject_int8_bitflip_to_fake_quant(module, coord, bit):
    # Get int8 weight, XOR the bit, dequantize, write back to module.weight
    # Returns (orig_i8, corr_i8, orig_fp, corr_fp)

# Weight restore after each trial
module.weight = nn.Parameter(clean_weight.detach().clone())
# or: module.weight.data.copy_(clean_weight)
```

---

## 5. Experiment Infrastructure

- **Fresh model per bit position** (outer loop) ensures accumulated corruption doesn't confound results.
- **Clean weight clone + restore** per coordinate (inner loop) ensures single-fault isolation.
- **JSON logging** per trial: `{coord: {loss, int8_change, fp32_change}}` or `{coord: loss}`.
- **GPU scheduling**: Explicit job queues across multiple CUDA devices, load-balanced by model × experiment-type runtime.
- **Models tested**: torchvision ResNet-50, HuggingFace ViT-base-patch16-224, Qwen3-30B-A3B (MoE).

---

## 6. Key References

| Paper | Venue | Used For |
|---|---|---|
| Rakin et al. "Bit-Flip Attack" | ICCV 2019 | BFA/PBS methodology |
| Sullivan et al. | MICRO 2021 | Byte-aligned multi-bit error distribution |
| Roquet et al. "MaxiMals" | DAC 2024 | ViT kernel-level SDC FIT rates |
| Xue et al. | TVLSI 2023 | ViT layer/module vulnerability profiling |
| Burel et al. | 2021 | Quantized vs FP32 robustness comparison |
