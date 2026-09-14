# SWIR-AdaIR-SR — Implementation Specification v2.1 (Fixed)

**Supersedes:** `SWIR_AdaIR_SR_SPEC_v2.md` (v2.0), which itself superseded the v1 coding handoff.
**Audience:** autonomous coding agent + human reviewer.
**Status of numbers:** every numeric degradation parameter is a *provisional engineering default*, not a measured sensor constant. Every claim marked `[VERIFIED]` was checked against source; `[UNVERIFIED]` must be checked by the agent before it is relied on.

**v2.1 correction summary.** This revision fixes scale-mixed batching, the D-2 gradient test, unintended dead parameters, degradation notation, Soft-FMiM method status, FFT precision scope, gated-base initialization, backscatter/exposure coupling, quantization semantics, tiled-inference criteria, frequency-loss wording, baseline grouping, and detection-control naming.

---

## 0. How to use this document

1. Read §1 (invariants) and §3 (conventions) before writing any code. They override every later section if a conflict appears.
2. §5 lists four defects in the upstream repository. Implement the fix *and* the assertion for each before training anything.
3. Do not proceed past a milestone gate (§16) until every checkbox in it passes with evidence pasted into the run log.
4. If a required input is missing (§18), stop and ask. Do not invent a value and do not silently pick a default.

---

## 1. Hard invariants

| ID | Invariant |
|---|---|
| INV-1 | One architecture, one checkpoint, all degradations. No per-degradation checkpoints. |
| INV-2 | The model never receives a degradation label, at training or inference. `forward(x)` and `forward(x, out_scale=s)` only. |
| INV-3 | `out_scale` is a **deployment setting**, not a degradation label. Both `s=1` and `s=2` are trained in the same checkpoint, but **each optimizer batch is scale-homogeneous** (one `out_scale` per batch). INV-2 is not violated by INV-3. |
| INV-4 | No spatial size is hard-coded anywhere. All shapes derive from the runtime tensor. |
| INV-5 | Nothing is reported as a result unless it was produced by an executed run whose config, seed and commit hash are in the log. |
| INV-6 | No architectural addition (Mamba, diffusion, detector head, degradation classifier) before the Milestone-2 gate passes. |
| INV-7 | Every table states the evaluation resolution, the degradation distribution that produced the inputs, and the number of seeds. |
| INV-8 | The upstream AdaIR baseline is preserved on a tagged branch and remains runnable. |

---

## 2. Requirement traceability (proposal → implementation)

The funded proposal specifies three separate model figures plus a detection task. This spec implements a **single unified backbone** and must therefore justify, per bullet, where the requirement is satisfied or why it is deferred. **This table is a deliverable for the PI, not an internal note.** Rows marked `GAP` require a decision (§18).

| Proposal requirement (Korean doc) | Where satisfied in this spec | Status |
|---|---|---|
| Denoising: 주파수-공간 하이브리드 특징 추출 (frequency–spatial hybrid) | AdaIR FreModule, **after** the D-2 repair (§5) | Satisfied only post-repair |
| Denoising: CNN 국소 + Transformer 전역, 다중 스케일-전역 융합 | AdaIR overlap-conv patch embed + Restormer blocks + 4-level U-hierarchy | Satisfied |
| Denoising: 복합 손실 (pixel + structural/perceptual), PSNR/SSIM/LPIPS | §10.2 composite loss, §11 metrics | Satisfied |
| Denoising / Low-light: **경량 (lightweight) 모델** | Backbone is 28.8 M params `[VERIFIED]` — **not lightweight** | **GAP-1** |
| SR: Dual-Path Encoder | Not implemented (single encoder) | **GAP-2** |
| SR: Tokenization of scene context injected at bottleneck + Cross-Attention fusion | Not implemented | **GAP-3** |
| SR: Spatial Modulation of decoder features | Not implemented | **GAP-4** |
| SR: suppression of diffusion-specific hallucination patterns | Not applicable — this model is not diffusion-based | **GAP-5** (proposal wording assumes a diffusion SR model) |
| Low-light: Spatial Path / Channel Path split + Cross-Fusion | Not implemented | **GAP-6** |
| Low-light: 채널별 조도 적응 (per-channel illumination) | Ill-defined for single-channel SWIR; would have to mean *feature* channels, not image channels | **GAP-7** (needs PI clarification) |
| Detection: YOLO mAP before/after restoration on NSLSR | §12 detection protocol | Satisfied, Milestone 3 |
| Detection: MOS subjective evaluation | Not implemented | **GAP-8** (deferred, needs human raters) |
| Detection: adversarial robustness training | Not implemented | **GAP-9** (deferred) |
| 1× SCI journal paper, 3× non-SCI presentations (stage 1) | §17 venue note | Planning item |

**GAP-1 handling.** Add a parameter/latency budget as a first-class constraint: report params, MACs @ 320×256 and @ 640×512, peak memory and ms/frame for every arm. If the proposal's "lightweight" is contractual, plan a distilled/width-reduced variant (`dim=32`, `num_blocks=[2,3,3,4]`) as a second checkpoint and report the accuracy/cost curve. Do not claim lightweight without a stated budget it meets.

---

## 3. Conventions (binding)

* **Dimension ordering.** Prose uses `W×H` only when quoting the dataset ("640×512"). **All code, configs, tensors, shapes and tables use `H×W`.** NSLSR native = `H=512, W=640`. Development LR = `H=256, W=320`.
* **Tensor contract.** Images are `float32`, shape `[B,1,H,W]`, range `[0,1]`.
* **Divisibility.** Pad or crop so that `H % 16 == 0 and W % 16 == 0`. Upstream uses `base=16` `[VERIFIED: utils/dataset_utils.py]`; the architectural minimum is 8, but 16 is the required value here.
* **Clamping.** Never clamp the model output during training (it kills gradients in the saturated region). Clamp to `[0,1]` only for metric computation and for saving images.
* **Naming.** Model class `SWIRAdaIRSR`. Config key `out_scale`. No other name for the scale.

---

## 4. Verified upstream facts

Commit checked: `c-yn/AdaIR @ ccb8b98e49614e07badd0641e5163fa7635c2f02` (2025-12-24). The agent must re-record the commit it actually clones.

```python
# net/model.py  [VERIFIED]
class AdaIR(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim=48,
                 num_blocks=[4,6,6,8], num_refinement_blocks=4,
                 heads=[1,2,4,8], ffn_expansion_factor=2.66,
                 bias=False, LayerNorm_type='WithBias', decoder=True):
```

| Fact | Value |
|---|---|
| Frequency modules | `fre1 = FreModule(dim*2**3, heads[2], bias)` on the latent; `fre2 = FreModule(dim*2**2, ...)` on decoder L3; `fre3 = FreModule(dim*2**1, ...)` on decoder L2 |
| Call signature | `fre1(inp_img, latent)` — the **raw input image** is argument `x`, bilinearly resized to the feature resolution |
| Output layer | `nn.Conv2d(int(dim*2**1), out_channels, 3,1,1)`; `out = self.output(x) + inp_img` → H×W → H×W |
| Decoder L1 / refinement width | `int(dim*2**1)` = 96 at `dim=48` |
| Training defaults | AdamW, lr 2e-4, L1, `warmup_epochs=15`, `max_epochs=180`, `--epochs 150`, `--patch_size 128`, `--batch_size 8` |
| Parameters | ≈ 28.8 M |
| Licence | MIT (AdaIR); MIT (LSFDNet). Data licence separate and `[UNVERIFIED]` |

Files to read before editing: `net/model.py`, `utils/dataset_utils.py`, `utils/degradation_utils.py`, `utils/schedulers.py`, `train.py`, `test.py`, `options.py`.

---

## 5. Upstream defects — required fixes and assertions

### D-1: `FreModule` channel mismatch for 1-channel input

`FreModule.__init__(self, dim, num_heads, bias, in_dim=3)` builds `self.conv` / `self.conv1` with `in_dim`, but `AdaIR` never forwards `inp_channels`. `[VERIFIED: L290, L392–394]`

**Fix.** Pass `in_dim=inp_channels` to all three constructions.

**Assertion (`tests/test_fremodule_channels.py`).**
```python
m = SWIRAdaIRSR(inp_channels=1, out_channels=1)
for f in (m.fre1, m.fre2, m.fre3):
    assert f.conv.in_channels == 1 and f.conv1.in_channels == 1
_ = m(torch.randn(1,1,128,128))   # must not raise
```

### D-2: degenerate, non-differentiable frequency mask (**highest-impact upstream defect / method-variant trigger**)

```python
def fft(self, x, n=128):          # [VERIFIED L337-365]
    threshold = self.rate_conv(F.adaptive_avg_pool2d(x,1)).sigmoid()   # in (0,1)
    for i in range(mask.shape[0]):
        h_ = (h//n * threshold[i,0,:,:]).int()
        w_ = (w//n * threshold[i,1,:,:]).int()
        mask[i, :, h//2-h_:h//2+h_, w//2-w_:w//2+w_] = 1
```

Because `n=128` and `threshold < 1`, `h_ = int((h//128)·threshold)` is 0 for the small feature maps used in this project and remains 0 at dimension 128 because `int(1·threshold)=0` for `threshold<1`. Evaluated for this project:

| Module | LR 128×128 | LR 256×320 | LR 512×640 |
|---|---|---|---|
| fre1 (H/8) | 16×16 → 0×0 | 32×40 → 0×0 | 64×80 → 0×0 |
| fre2 (H/4) | 32×32 → 0×0 | 64×80 → 0×0 | 128×160 → 0×0 |
| fre3 (H/2) | 64×64 → 0×0 | 128×160 → 0×0 | 256×320 → potentially non-zero depending on the learned threshold |

At the intended patch sizes the low-frequency mask is therefore predicted to be empty in most/all AFLBs. In addition, the integer cast makes the mask boundary non-differentiable with respect to `rate_conv`.

**Additional dead-parameter audit.** Static reading of the released `FreModule` indicates both `score_gen` and the extra projection `conv` are allocated but are not used by the shown forward path; `conv1` is the projection used by the FFT path. The agent must verify this against the exact cloned commit before deleting anything.

**Critical testing caveat — `para1=0`.** Upstream AdaIR initializes the residual scale of the frequency branch (`para1`) to zero. Therefore a first full-model backward pass can legitimately produce zero gradients inside the frequency branch even after the mask is made differentiable. A test that simply runs `model(x).mean().backward()` and expects non-zero `rate_conv` gradients is invalid.

**Required upstream evidence.**
1. Record mask statistics on the unmodified upstream code at the intended resolutions.
2. Record which `FreModule` parameters are unused/dead on the exact cloned commit.
3. Record `para1` initialization.
4. Do not describe the condition as an upstream "bug" in a paper until runtime evidence is collected; use "released-code behavior" or "implementation limitation" in internal logs.

**Method variant (required for this project): Soft-FMiM.** Replace the hard integer box with a differentiable, resolution-relative soft mask:

$$
M(u,v)=\sigma\!\Big(\tfrac{1}{\tau}\big(r_\theta-\rho(u,v)\big)\Big),\qquad
\rho(u,v)=\sqrt{\big(\tfrac{2u}{h}\big)^2+\big(\tfrac{2v}{w}\big)^2}.
$$

This is **not silently treated as stock AdaIR**. It is an explicit architectural variant of AdaIR's frequency mining and must be independently switchable:

```yaml
frequency_mask:
  mode: upstream_integer   # exact released behavior
  # or:
  mode: soft_relative      # proposed Soft-FMiM
  tau: 0.05
```

For the isotropic form, `rate_conv(...).sigmoid()` predicts a normalized cutoff $r_\theta$. An anisotropic variant may use separate $r_h,r_w$ from the existing two output channels. The coordinate system must be defined around the shifted FFT center and normalized so behavior is resolution-relative. Build the mask fully vectorized; remove the per-sample Python loop in the soft variant.

If `score_gen` and/or `conv` are confirmed dead, remove them from `SoftFreModule` or deliberately wire them into the method. Do not keep unintended trainable dead parameters.

**Assertions (`tests/test_fre_mask.py`).**

```python
# A. released behavior: record, do not assume
mask_up = fre_upstream.debug_mask(x)
log(mask_up.mean(), mask_up.sum())

# B. soft mask is non-degenerate
mask = fre_soft.debug_mask(x)
assert 0.01 < mask.mean().item() < 0.99

# C. both spectral branches contain signal
low, high = fre_soft.debug_split(x)
assert low.abs().mean() > 0
assert high.abs().mean() > 0

# D. isolate differentiability of the new cutoff.
# Upstream para1 starts at 0, so neutralize that gate for this unit test.
with torch.no_grad():
    fre_soft.para1.fill_(1.0)

fre_soft.zero_grad(set_to_none=True)
out = fre_soft(x_img, y_feat)
loss = out.square().mean()
loss.backward()

g = fre_soft.rate_conv[0].weight.grad
assert g is not None
assert g.abs().sum() > 0
```

**Gradient-coverage rule.** Do not assert that every model parameter has a gradient after one forward pass. The model contains mutually exclusive `s=1` and `s=2` output heads, so one head is intentionally inactive in a scale-homogeneous batch. Instead, collect gradient coverage across both scale passes:

```python
covered = set()
for s in (1, 2):
    model.zero_grad(set_to_none=True)
    loss = model(x_for_scale[s], out_scale=s).mean()
    loss.backward()
    covered |= {n for n,p in model.named_parameters()
                if p.requires_grad and p.grad is not None}

unexpected = expected_trainable_names - covered
assert not unexpected, unexpected
```

The expected set must exclude deliberately frozen parameters and any explicitly documented conditional branch that is not exercised by the test.

### D-3: AMP + FFT instability

`torch.fft.fft2` under `autocast` with fp16 inputs is a deployment risk on CUDA for non-power-of-two transform sizes such as 32×40. Upstream issue #8 is referenced but remains `[UNVERIFIED — confirm on the project's GPU/CUDA/PyTorch stack]`.

**Fix.** Disable autocast only around the numerically sensitive spectral operations. Keep subsequent attention/convolution in the normal AMP dtype unless the real stack demonstrates a stability problem. This avoids turning the whole AFLB into fp32.

```python
def spectral_split(self, x):
    orig_dtype = x.dtype
    with torch.autocast(device_type=x.device.type, enabled=False):
        x32 = x.float()
        fft = torch.fft.fft2(x32, norm="forward", dim=(-2, -1))
        mask = build_mask(..., dtype=x32.dtype, device=x32.device)
        low32  = torch.fft.ifft2(fft * mask,       norm="forward").real
        high32 = torch.fft.ifft2(fft * (1-mask), norm="forward").real

    return low32.to(orig_dtype), high32.to(orig_dtype)

def forward(self, x, y):
    low, high = self.spectral_split(self.conv1(x))
    # attention / modulation proceeds under the caller's AMP context
    ...
```

If the implementation uses `fftshift`, complex magnitude, or another exact upstream convention, preserve that convention in both the integer-mask and Soft-FMiM arms; only the mask construction should differ in ablation A2.

**Assertion.** Run forward+backward under the real AMP configuration on the target GPU for all intended spatial sizes, asserting finite activations, finite loss and finite gradients.

### D-4: unused argument in the forward signature

`forward(self, inp_img, noise_emb=None)` — `noise_emb` is never used. Remove it in `SWIRAdaIRSR`; the API contract is §8.

---

## 6. Problem formulation

Learn one $f_\theta$ mapping a degraded SWIR observation to a restored image at a requested output scale:

$$
\hat{x} = f_\theta(y,s),\qquad s\in\{1,2\},
$$

with a **scale-specific degradation operator**

$$
y=\mathcal D_s(x;\phi),\qquad \phi\sim p(\phi\mid s).
$$

* $x\in[0,1]^{1\times H_r\times W_r}$ — NSLSR SWIR reference image (**not** a noise-free ground truth; see §9.4).
* $\mathcal D_1$ applies restoration degradations but **no spatial downsampling**.
* $\mathcal D_2$ applies restoration degradations and contains **exactly one** ×2 downsampling operation.
* Optional degradation types are low-light, backscatter/veiling, optical blur and sensor noise.
* $\theta$ is shared across every degradation composition and both output scales.

This notation is binding. Never write `downsample(D(x))` if `D` may already contain downsampling; that ambiguity can accidentally create ×4 degradation.

**Two training regimes, one checkpoint:**

| Regime | Input | Target | Purpose |
|---|---|---|---|
| `s=2` (SR regime) | $y=\mathcal D_2(x)$ at $H/2\times W/2$ | $x$ at $H\times W$ | super-resolution deliverable |
| `s=1` (restoration regime) | $y=\mathcal D_1(x)$ at $H\times W$ | $x$ at $H\times W$ | denoising + low-light deliverables at **native resolution** |

Rationale for `s=1`: denoising and low-light enhancement must remain evaluable at native resolution, while `s=2` measures the SR task. `s` is a caller-controlled output-scale setting and is not a degradation-class label.

### 6.1 Scale sampling is per batch, not per sample

The two regimes produce different input tensor sizes for the same HR crop. A default PyTorch collate function cannot stack `s=1` and `s=2` samples in the same batch.

Therefore:

\[
\boxed{\text{one optimizer batch = one output scale}}
\]

Sample `s` once per batch, with default:

```yaml
scale_sampling:
  p_s1: 0.5
  p_s2: 0.5
  unit: batch
```

Valid implementations:

1. a batch sampler that emits `(indices, scale)`, or
2. two scale-specific DataLoaders interleaved by the trainer.

Do not pad `s=2` LR samples up to the `s=1` spatial size merely to mix scales in a batch; that changes the task.

Within a scale-homogeneous batch, degradation flags and degradation strengths may still be sampled independently per example.

---

## 7. Degradation operator $\mathcal{D}$

### 7.1 Ordering (physics-motivated, still an approximation)

For a chosen scale $s$:

$$
\mathcal D_s =
\mathcal Q\circ
\mathcal D_{\text{noise}}\circ
\mathcal S_s\circ
\mathcal D_{\text{optics}}\circ
\mathcal D_{\text{scatter}}\circ
\mathcal D_{\text{illum}},
$$

where

$$
\mathcal S_1 = \mathrm{Identity},
\qquad
\mathcal S_2 = \downarrow_2.
$$

Thus $\mathcal D_2$ contains exactly one ×2 spatial reduction and $\mathcal D_1$ contains none.

`Q` — configurable re-quantisation. During NSLSR benchmarking, reproduce the verified source encoding. During project-camera adaptation, use the measured camera output/ADC encoding instead. Source bit depth and deployment-camera bit depth are not assumed to be equal.

```python
levels = 2**bits - 1
y = np.round(y * levels) / levels
```

Configuration:

```yaml
quantization:
  mode: source       # source | target_camera | random | off
  bits: null         # resolved from data_profile or camera_profile
```

Quantization is mandatory only when the selected simulation mode is intended to represent a quantized acquisition path; do not hard-code NSLSR's file bit depth into future deployment simulation.

Each component is an independently switchable module with its own RNG stream so that a fixed seed reproduces the same $\phi$.

### 7.2 Illumination

$$x_L=\mathrm{clip}(\alpha\,x^{\gamma},0,1),\qquad \alpha\sim U[0.15,0.8],\ \gamma\sim U[1.0,2.2]$$

### 7.3 Backscatter / haze / smoke

$$I = J\,t + A(1-t),\qquad t=\exp(-\beta d)$$

NSLSR provides no depth map; use (a) constant $t$ per image, (b) a smooth low-frequency $t$ field, or (c) a horizon-biased field for maritime geometry. Label as *physics-inspired, uncalibrated*.

The classical atmospheric model is only a **physics-inspired veiling approximation** for development. The funded project explicitly mentions laser backscatter; active SWIR path radiance can depend on laser power, range, geometry, scattering concentration, target reflectance, exposure and gain. Therefore do **not** hard-code `A_eff = alpha * A_base` as a physical law.

Use a configurable coupling model:

$$
A_{\mathrm{eff}}=\alpha^\kappa A_{\mathrm{base}},
\qquad \kappa\in[0,1],
$$

or an independently sampled/scaled backscatter term when justified by the chosen simulator.

Interpretation:

* $\kappa=1$: exposure scales target and veiling terms together,
* $\kappa=0$: backscatter amplitude is independent of the synthetic scene-darkening factor,
* intermediate $\kappa$: partial coupling.

Configuration:

```yaml
backscatter:
  model: atmospheric_veiling      # provisional; not a calibrated active-laser model
  exposure_coupling_kappa: [0.0, 1.0]
```

Until real project-camera measurements exist, all such backscatter parameters must be labelled *physics-inspired and uncalibrated*. Do not claim this equation faithfully simulates active laser backscatter.

### 7.4 Optics and downsampling

Randomised kernel family: isotropic Gaussian, anisotropic Gaussian, mild motion; `sigma_range: [0.2, 2.0]`. Do **not** train with bicubic-only downsampling. `LSFDNet/basicsr/data/degradations.py` already vendors the Real-ESRGAN kernel/noise utilities `[VERIFIED]` — reuse them rather than reimplementing, and cite accordingly.

### 7.5 Sensor noise

$$y = x + n_{\text{shot}}(x) + n_{\text{read}} + n_{\text{FPN}}$$

Progression, each switchable: **v0** Gaussian (debug only) → **v1** Poisson–Gaussian $\mathrm{Poisson}(gx)/g + \mathcal{N}(0,\sigma_r^2)$ → **v2** + column offsets + row offsets + a low-frequency fixed-pattern map. Reference for SWIR-specific modelling: Jiang et al., ICCVW 2025 (§19). Replace with measured statistics from real dark/flat frames when the camera arrives — this is a simulator change, not an architecture change.

### 7.6 Composite sampling and metadata

Within a **scale-homogeneous batch**, optional degradation flags may be sampled independently per example. Defaults: `blur_p 0.70`, `noise_p 0.60`, `low_light_p 0.45`, `backscatter_p 0.35`. Preserve a meaningful fraction of near-clean samples so the model learns not to over-correct. `__getitem__` returns a `meta` dict with every flag and sampled parameter, used **only** for stratified evaluation and logging. The scale is batch metadata, not an independently sampled per-example degradation label.

### 7.7 Held-out simulator (required, addresses circularity)

Define two disjoint parameter sets in the config: `sim_A` (training) and `sim_B` (evaluation only) differing in kernel family, noise mode and transmission/veiling model. Every headline synthetic table reports both `sim_A` and `sim_B`. Call this **simulator-shift / synthetic-OOD robustness**, not external validity. A large A→B gap is a finding, not a failure to hide; real-camera/naturally degraded evaluation is the external-validity arm.

---

## 8. Model specification

```python
class SWIRAdaIRSR(nn.Module):
    def __init__(self, inp_channels=1, out_channels=1, dim=48,
                 num_blocks=(4,6,6,8), num_refinement_blocks=4,
                 heads=(1,2,4,8), ffn_expansion_factor=2.66, bias=False,
                 layernorm_type='WithBias',
                 scales=(1,2), gated_base=True, fre_tau=0.05): ...

    def forward(self, x: Tensor, out_scale: int = 2) -> Tensor:
        """x: [B,1,H,W] float32 in [0,1];  returns [B,1,sH,sW]."""
```

Backbone identical to upstream AdaIR with fixes D-1…D-4 applied. Output stage:

```python
feat = self.refinement(out_dec_level1)          # [B, int(dim*2**1), H, W]
res  = self.heads[str(out_scale)](feat)         # ModuleDict: '1' -> conv, '2' -> PixelShuffle(2)
base = x if out_scale == 1 else F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
if self.gated_base:
    g, b = self.base_gate(feat)                 # [B,1,1,1] each, from pooled features
    out = res + g * base + b
else:
    out = res + base
return out                                       # no clamp during training (§3)
```

**Feature widths** are derived, never literal: head input `= int(dim*2**1)`. `SRHeadX2(in_channels=96)` in v1 violates INV-4 — fix it.

**Gated base.** With $y\approx\alpha x$, a fixed residual skip can force the residual branch to synthesize most of the signal under severe low light. The base gate is therefore an explicit method component and must be ablated.

Make its parameterization and initialization identity-preserving:

$$
g = 1+\tanh(\Delta g),\qquad
b = b_{\max}\tanh(\Delta b),
$$

with the final predictor weights/biases zero-initialized, so at initialization:

$$
g=1,\qquad b=0.
$$

Thus the model begins equivalent to ordinary residual learning rather than accidentally beginning with a 0.5 skip as a zero-logit sigmoid would. `gated_base` on/off is required ablation A4. Log the learned distribution of $g$ by degradation category.

**PixelShuffle init.** Use ICNR initialisation on the ×2 expansion convolution and verify no systematic checkerboard artifact on the fixed validation panel.

**Padding.** A `check_image_size()` wrapper reflect-pads to a multiple of 16 and crops the output back to `s·H × s·W`.

**Tiled inference.** Provide `tile_size` / `tile_overlap` with feathered blending for memory-constrained deployment, but do **not** require tiled output to be numerically equivalent to whole-image output. AdaIR's FFT branch depends on the spatial support of the input, so tile-wise FFTs are intrinsically different from a full-frame FFT.

Evaluate tiled inference empirically:

* PSNR/SSIM difference versus whole-frame inference,
* seam/overlap-region error,
* sensitivity to tile overlap,
* latency and peak memory.

Record $\Delta$PSNR(tile−whole) rather than predeclaring an unrealistically tiny equality tolerance. Whole-frame inference is the primary benchmark whenever memory permits.

---

## 9. Data specification

### 9.1 Splits (pinned — replaces "preserve where possible")

`[VERIFIED from LSFDNet README §4.2]` NSLSR ships as four folders, each containing `SWIR/`, `LWIR/`, `labels/`:

| Role | Folder | Use here |
|---|---|---|
| Train | `data/NSLSR_train/SWIR/` | restoration training |
| Val | `data/NSLSR_val/SWIR/` | model selection, per-degradation validation |
| Test | `data/NSLSR_test/SWIR/` | final restoration metrics — touched once |
| Detection | `data/NSLSR_val/` and `data/NSLSR320_val/` + `labels/` | §12 |

`NSLSR320_val` is centre-cropped to 320×320 from 640×512 for detection reproduction `[VERIFIED]`. It is **not** a ×2 downsampled SR input and must never be substituted for the synthetic 256×320 LR image used by the ×2 restoration experiment. Total dataset = 1,205 pairs `[VERIFIED]`. LWIR is **not** used by this model; note it as available for a future fusion extension.

**Acquisition risk.** The dataset is distributed via Baidu Pan `[VERIFIED]`. Treat download as a blocking prerequisite; on success, write `data/MANIFEST.sha256` with per-file hashes and counts per split, and never re-download silently.

### 9.2 Format inspection (before writing the loader)

Extensions observed in the upstream loader: `.bmp`, `.tif`, `.jpg`, `.png` `[VERIFIED]`. LSFDNet reads with plain `cv2.imread` (8-bit, 3-channel BGR) `[VERIFIED]` — that is *their* pipeline's assumption, not proof of the file's true bit depth.

Run and record for ≥ 20 files per split: extension, PIL mode, numpy dtype, shape, channel count, whether channels are identical, min/max/percentiles. Then:

* 8-bit → `x = I / 255.0`
* 16-bit → normalise by the **verified** encoding; do not assume 65535
* 3 identical channels → take channel 0; do **not** call `.convert('L')` on 16-bit data

Write the outcome into `configs/data_profile.yaml` and make the loader read it, so the assumption is explicit and versioned.

### 9.3 Leakage audit (blocking, run before any reported number)

NSLSR was captured by a synchronous binocular rig across time periods `[VERIFIED]` — adjacent frames are plausibly near-duplicates. Compute a perceptual hash and a nearest-neighbour mean-absolute-difference for every test image against every train image. Fail the milestone if any cross-split pair exceeds the configured similarity threshold; report the distribution either way.

### 9.4 Reference-quality limitation (must appear in the paper)

The 640×512 NSLSR SWIR image is a *captured* frame containing native sensor noise, not a clean reference. Training `noisy-LR → captured-HR` teaches the model to reproduce the sensor's own noise and caps attainable PSNR. Never describe it as ground truth, and never claim that a 1280×1024 output contains physically measured detail.

### 9.5 Patch rule

Crop from HR first, then degrade — never crop LR and HR independently.

```
s=2 :  HR crop 256×256  →  D  →  LR 128×128   (both multiples of 16)
s=1 :  HR crop 256×256  →  D  →  LR 256×256
```

### 9.6 Augmentation

Horizontal flip, vertical flip and 90° rotations only if they do not violate maritime scene assumptions (a horizon-biased backscatter field interacts with vertical flips — resolve in config). No RGB colour jitter. Geometric augmentation is applied to the HR crop *before* degradation and is never mixed into the degradation module.

---

## 10. Training

### 10.1 Schedule

Repository-reference schedule: AdamW, lr 2e-4, `LinearWarmupCosineAnnealingLR(warmup=15, max_epochs=E)`. Do not copy `E=180` blindly — NSLSR is far smaller than the upstream multi-task mixture; set `E` from a convergence run and log the choice.

**Paper/code distinction.** Record separately what the AdaIR paper reports and what the exact released commit executes. The paper reports Adam-based training, whereas the inspected repository uses AdamW. Never label one as the universal "AdaIR default" without identifying the source.

Initialisation: **pretrained-partial is the default starting arm**, because the dataset is small relative to the ≈28.8 M-parameter backbone. Scratch vs pretrained remains required ablation A5.

For RGB→SWIR-compatible input projections, channel-average only weights that remain semantically compatible:

$$
W_{\text{gray}}=\frac13\sum_c W_{\text{RGB},c}.
$$

Candidate layers include `patch_embed.proj` and the actually used input projection(s) inside the frequency module.

Because Soft-FMiM changes the mask semantics, **do not blindly inherit the upstream cutoff predictor**. Default policy:

* load compatible encoder/decoder/attention weights,
* load compatible image-projection weights,
* reinitialize the Soft-FMiM cutoff predictor (`rate_conv`) unless an explicit ablation tests inherited initialization,
* do not load the new `s=1`/`s=2` output heads or base gate.

If static/runtime auditing confirms upstream `conv` or `score_gen` is unused, do not initialize or carry those dead modules into `SoftFreModule`.

The exact source checkpoint must be recorded (which AdaIR task setting / all-in-one checkpoint). Print and log every missing/unexpected key; `strict=False` without an inspected key report is forbidden.

### 10.2 Loss

Baseline: $\mathcal{L}=\lVert\hat{x}-x\rVert_1$. Prove convergence first.

Candidate project loss (only after the L1 baseline is stable):

$$
\mathcal{L}=
\lambda_1\mathcal{L}_1+
\lambda_s(1-\mathrm{SSIM})+
\lambda_e\lVert\nabla\hat{x}-\nabla x\rVert_1+
\lambda_f\mathcal{L}_{\mathrm{freq}}.
$$

Defaults `1.0 / 0.1 / 0.05 / 0.05` are provisional and config-driven.

**Do not prescribe a raw complex-spectrum loss without evidence.** A magnitude-only spectrum loss omits phase information, but a naïve complex L2 FFT loss can be largely redundant with image-space L2 by Parseval/Plancherel equivalence. Therefore `L_freq` is an experimental slot, not a required formulation.

Candidate arms:

1. no frequency loss,
2. amplitude-spectrum loss,
3. real/imaginary L1 spectrum loss,
4. a focal-frequency-style objective if implemented and properly cited.

Normalize any spectral loss for the number of bins / transform convention and ablate every adopted term (A3). The final method may omit `L_freq` if it adds no reproducible value beyond L1 + structural/edge terms.

LPIPS: evaluation only, never a training loss in the baseline. State backbone (Alex or VGG), package version, and the grayscale→3-channel replication rule; note that values are not comparable with RGB-trained papers.

### 10.3 Reproducibility

Seed `torch`, `numpy`, `random`, per-worker seeds, `cudnn.deterministic`. Log: commit hash, full config, seed, split manifest hash, bit depth/normalisation, sampled degradation distribution, loss curves, all metrics, best epoch + selection metric, checkpoint path, params, MACs, latency, peak memory, GPU/CUDA/PyTorch versions. Save fixed visual validation samples per degradation category every N epochs.

---

## 11. Evaluation protocol

### 11.1 Metric conventions (pin these — different libraries give different numbers)

* PSNR: `data_range=1.0`, computed per image on `[0,1]` clamped outputs, then averaged. No border shave.
* SSIM: one pinned implementation and window size, stated in the paper.
* LPIPS: as §10.2.
* Report gradient/edge similarity as a secondary metric (relevant to distant target contours).
* Efficiency: params, MACs @ 256×320 **and** @ 512×640, peak GPU memory, ms/frame (mean over ≥100 runs after warm-up). Never compare MACs across papers without stating input size.

### 11.2 Fixed validation subsets

Pre-generate and freeze degradation parameters (do not resample per epoch — that makes epoch-to-epoch comparison noise-dominated). Evaluate separately on: SR-only; +noise; +low-light; +backscatter; +blur; low-light+noise; full composite. Repeat the whole grid for `s=1` and `s=2`, and for `sim_A` and `sim_B`.

Aggregate PSNR alone is never reported without the per-degradation breakdown.

### 11.3 Real-degraded arm (addresses circularity)

Identify naturally dark / hazy NSLSR frames. No reference exists, so report: no-reference metrics (with an explicit caveat that NIQE/BRISQUE are tuned on natural RGB images), qualitative panels, and detection mAP. This arm carries the external-validity claim; the synthetic arms do not.

### 11.4 Statistics

Headline table: ≥3 seeds, mean ± std. Ablations may be single-seed if disclosed. Do not report a difference smaller than the seed spread as an improvement.

---

## 12. Detection evaluation

Labels live in `labels/` per split `[VERIFIED — format to be confirmed, likely normalised YOLO txt]`. Confirm before writing any coordinate transform. If normalised, rescaling the image does **not** require changing the labels; if absolute pixels, scale by `s`.

**Required arms, identical frozen detector weights, identical letterbox size, confidence and NMS thresholds:**

| Arm | Input to detector | Purpose |
|---|---|---|
| A0 | native NSLSR reference SWIR | reference arm (not assumed to be noise-free or an upper bound) |
| A1 | degraded input, native size | lower bound |
| A2 | bicubic ×`s` of the degraded input | **resolution control** |
| A3 | restored output, `s=2` | the claim |
| A4 | restored output, `s=1` | isolates restoration from upscaling |

Without A2 and A4, an mAP gain in A3 is confounded by pixel count and is not attributable to restoration. State whether the detector was trained on clean HR SWIR and keep it frozen across all arms. No detector gradients flow into the restoration model at this stage.

Report restoration metrics and mAP jointly. A PSNR gain with an mAP loss is a publishable finding about the mismatch between fidelity metrics and small-target detectability — do not suppress it.

---

## 13. Experiment matrix with acceptance and kill criteria

| ID | Configuration | Acceptance | Kill / redesign trigger |
|---|---|---|---|
| E0 | Bicubic ×2 | reference numbers recorded | — |
| E1 | SR-only training, `s=2` | val PSNR > E0 by a margin exceeding seed spread | ≤ E0 ⇒ architecture or data bug; stop |
| E2 | E1 + noise | ≥ E1 on noisy val, ≤0.3 dB drop on clean val | large clean-val drop ⇒ capacity/sampling problem |
| E3 | E2 + low-light | improves low-light val | no gain ⇒ test A4 (gated base) before proceeding |
| E4 | E3 + backscatter + full composite | improves composite val without collapsing single-degradation cases | composite training worse than every specialist on every task ⇒ the all-in-one premise fails; report as a negative result |
| E5 | E4 + composite loss | improves SSIM/edge without PSNR loss beyond seed spread | PSNR loss with no perceptual gain ⇒ revert to L1 |
| E6 | E4 with `s∈{1,2}` joint | `s=1` competitive with a `s=1`-only model | large gap ⇒ report the cost of scale-sharing |
| **A1** | frequency modules on / off | quantifies AdaIR's contribution | — |
| **A2** | released integer FMiM vs proposed Soft-FMiM (D-2) | quantifies the architectural method variant | if no difference, Soft-FMiM has no practical value under this setup — report that honestly |
| A3 | L1 vs each loss term | per-term justification | — |
| A4 | gated vs ungated bicubic base | expected to matter most for low-light | — |
| A5 | scratch vs pretrained init | — | — |
| A6 | structured SWIR noise vs Gaussian | tests the simulator's value | — |
| A7 | `sim_A`-trained evaluated on `sim_B` | simulator-shift / synthetic-OOD robustness | large gap ⇒ narrow the simulator-generalization claims; do not call this external validity |

A2 and A4 are the ablations that carry the method's identity. A1 must be interpreted together with A2 because the released integer-mask behavior and Soft-FMiM are materially different frequency modules.

---

## 14. Baselines — grouped by the task actually evaluated

Do not force same-resolution restoration models into an ×2 SR comparison without adding and disclosing an SR head.

### 14.1 `s=1` native-resolution restoration

1. Native degraded input / identity reference where applicable.
2. Task specialists: denoise-only, low-light-only, etc., using comparable capacity.
3. All-in-one competitors retrained on the same SWIR training distribution: stock AdaIR, AirNet, PromptIR, subject to verified implementation/citation.
4. Proposed shared-scale model evaluated with `out_scale=1`.

### 14.2 `s=2` super-resolution

1. Bicubic ×2.
2. At least one established ×2 SR specialist on grayscale/SWIR (e.g. EDSR/SwinIR/HAT-class, exact citation/config to be verified).
3. **Controlled AdaIR-SR baseline:** AdaIR backbone + the *same* ×2 output head and training data, using the released frequency-mask behavior.
4. Proposed AdaIR-SR + Soft-FMiM / other explicitly named method components.

The controlled AdaIR-SR baseline is essential: otherwise gains from merely adding an SR head are confounded with gains from the proposed frequency modification.

### 14.3 Unified-scale cost

Compare:

* one checkpoint jointly trained for `s∈{1,2}`,
* one `s=1`-only checkpoint,
* one `s=2`-only checkpoint.

This quantifies the accuracy/capacity cost of sharing one backbone across both output scales.

### 14.4 Simulator baseline

Include a Real-ESRGAN-style blind degradation pipeline as an alternative **training-data** baseline, isolating "better degradation simulator" from "better architecture."

Never tune the proposed model more aggressively than the baselines. Log the hyperparameter-search budget, training steps and data exposure for each arm.

---

## 15. Repository layout and file contract

```
AdaIR/
├── net/
│   ├── model.py                 # upstream, unmodified, tagged branch preserved
│   └── swir_adair_sr.py         # NEW: SWIRAdaIRSR, FreModuleSoft, scale heads, base gate
├── utils/
│   ├── swir_sr_dataset.py       # NEW: NSLSRSWIRDataset
│   ├── swir_degradation.py      # NEW: modular D, sim_A/sim_B, metadata, quantisation
│   ├── swir_metrics.py          # NEW: pinned PSNR/SSIM/LPIPS/edge
│   └── swir_io.py               # NEW: bit-depth-aware read/write, data_profile
├── configs/
│   ├── swir_base.yaml           # §16 schema
│   ├── data_profile.yaml        # generated by §9.2 inspection
│   └── sim_b.yaml               # held-out simulator
├── tools/
│   ├── inspect_dataset.py       # §9.2
│   ├── leakage_audit.py         # §9.3
│   └── bench.py                 # params / MACs / latency / memory
├── train_swir.py  test_swir.py  infer_swir.py  detect_eval.py
└── tests/
    ├── test_fremodule_channels.py   # D-1
    ├── test_fre_mask.py             # D-2
    ├── test_amp_fft.py              # D-3
    ├── test_shapes.py  test_backward.py  test_padding.py  test_tiling.py
    ├── test_degradation_determinism.py
    └── test_dataset_alignment.py
```

Dataset interface:
```python
class NSLSRSWIRDataset(Dataset):
    def __init__(self, root, split, hr_patch=256,
                 degradation_cfg=None, augment=True, data_profile=None): ...

    def get_item(self, i: int, out_scale: int) -> dict:
        # out_scale is supplied by a scale-homogeneous batch sampler/loader.
        return {"lr": Tensor,   # [1,h,w]
                "hr": Tensor,   # [1,s*h,s*w]
                "scale": int,
                "name": str,
                "meta": dict}
```

A normal `__getitem__` may wrap this for a fixed-scale Dataset view, but do not independently sample `out_scale` inside each item when using ordinary tensor collation.
```

---

## 16. Milestone gates

**M0 — Upstream preserved.** Clone, tag, record commit, run one upstream smoke test, record environment.

**M1 — Architecture correct.**
- [ ] 1-channel in/out; `FreModule(in_dim=1)` wired (D-1 assertion passes)
- [ ] D-2 assertions: degeneracy demonstrated on the original, non-degenerate mask and non-zero `rate_conv` gradient after the fix
- [ ] D-3 forward+backward under real AMP on the target GPU, all finite
- [ ] gradient-coverage audit passes **across both `s=1` and `s=2` branches**; no unintended dead trainable parameter remains
- [ ] scale-homogeneous batching test passes; a batch never contains mixed spatial sizes from `s=1` and `s=2`
- [ ] shapes `[B,1,128,128]→[B,1,256,256]` (s=2), `[B,1,256,320]→[B,1,512,640]` (s=2), `[B,1,512,640]→[B,1,1024,1280]` (s=2), and `s=1` identity-size cases
- [ ] padding wrapper handles non-multiple-of-16 inputs
- [ ] tiled inference is seam-checked and its PSNR/SSIM delta vs whole-frame inference is reported; numerical equality is **not** required
- [ ] params, MACs, latency, peak memory reported at both resolutions

**M2 — Data and SR baseline.**
- [ ] dataset downloaded, `MANIFEST.sha256` written, counts per split logged
- [ ] format inspection complete; `data_profile.yaml` committed
- [ ] leakage audit run and reported
- [ ] deterministic degradation test passes (same seed → identical $\phi$ and identical output)
- [ ] batch-scale sampler/interleaved-loader determinism is tested and logged
- [ ] HR-crop → aligned-LR alignment test passes
- [ ] E1 converges and beats E0 by more than the seed spread
- [ ] best-checkpoint selection by validation PSNR works; config+seed+commit snapshotted

**M3 — Restoration and detection.** E2–E6, ablations A1–A7, detection arms A0–A4, real-degraded arm, ≥3 seeds on the headline table.

Only after M2 does composite degradation training become the main focus. Only after M3 does any architectural extension (GAP-2…GAP-6, Mamba, diffusion, joint detection training) get considered.

---

## 17. Prohibited actions

1. Replacing AdaIR with an unrelated architecture.
2. Separate checkpoints per degradation.
3. Passing a degradation label to `forward`.
4. Hard-coding any spatial size or feature width.
5. `.convert('RGB')` / `.convert('L')` on SWIR files before §9.2 inspection; silent 16→8-bit conversion.
6. Reporting a number from a run whose config, seed and commit are not logged.
7. `strict=False` weight loading without an inspected key report.
8. Aggregate-only PSNR without the per-degradation and per-scale breakdown.
9. Claiming the 1280×1024 output contains physically measured detail.
10. Claiming "lightweight", "real-time" or "state-of-the-art" without a stated budget/benchmark the model actually meets.
11. Adding Mamba, diffusion, YOLO-in-the-loop or extra heads before M3.
12. Tuning anything on the test split.
13. Sampling `s=1` and `s=2` independently per example inside a normal collated batch.
14. Calling `sim_B` evaluation "external validity"; it is simulator-shift robustness.
15. Using `NSLSR320_val` as the ×2 SR LR input; it is a center-cropped detection subset, not a downsampled pair.
16. Describing Soft-FMiM as stock AdaIR or as a silent implementation repair; it is an explicit method variant.
17. Requiring tile inference to numerically equal whole-frame inference when the model contains global FFT operations.

---

## 18. Open questions — must be answered before M3 (some before M1)

1. **GAP-2…GAP-6:** does the PI accept a unified backbone, or are the three proposal figures contractual deliverables that must each be implemented? *(Blocks the paper's framing; answer before M2.)*
2. **GAP-1:** what is the numeric "lightweight" budget — parameters, MACs, or ms/frame on which hardware?
3. **GAP-7:** for single-channel SWIR, does "채널별 조도 적응" mean feature channels, or was the figure drawn assuming RGB input?
4. Does the real project camera output 640×512, and at what bit depth and frame rate? The ×2 deployment claim depends on this.
5. Is any real higher-resolution SWIR ground truth obtainable? Without it, the ×2 deployment output is unverifiable by construction.
6. NSLSR data licence and redistribution terms (separate from the MIT code licence).
7. Detection: which detector, trained on what, and is it frozen? Label format confirmation.
8. Target venue and deadline — determines whether the baseline set in §14 is achievable in time.
9. Dual-use / release policy: is the trained checkpoint or the simulator releasable?
10. Can project-camera dark frames, flat fields, gain/exposure sweeps, and controlled smoke/haze/laser-backscatter captures be obtained? These measurements determine whether the provisional degradation simulator can be physically calibrated.

---


## 19. v2.1 implementation-critical notes

### 19.1 One checkpoint does not mean every branch executes every batch

The invariant is one architecture / one checkpoint. Conditional output-scale heads are permitted. Because scale batches are homogeneous, only the selected output head receives gradient on that step. This is expected behavior, not a dead-gradient defect.

### 19.2 Soft-FMiM naming and provenance

Use these names consistently in code and experiments:

* `upstream_integer`: exact released integer mask behavior,
* `soft_relative`: proposed differentiable resolution-relative mask.

The paper/repository may call the original mechanism FMiM/AFLB; do not rename the original retrospectively. "Soft-FMiM" is a project-internal working name until a final paper name is chosen.

### 19.3 Reference quality and claims

NSLSR 640×512 is a captured SWIR reference, not noise-free physical truth. Therefore early experiments demonstrate recovery toward the NSLSR reference after **added synthetic degradation**. They do not prove removal of the reference sensor's native noise.

If the project camera allows static repeated captures, consider building pseudo-clean references by temporal averaging / robust stacking under controlled scenes. That is a future data-quality improvement, not part of the initial NSLSR benchmark.

### 19.4 Real camera deployment

The architecture remains resolution-agnostic:

* development ×2: `256×320 -> 512×640` in H×W convention,
* deployment ×2: `512×640 -> 1024×1280`.

The principal deployment uncertainty is the degradation/domain gap, not the spatial-size change.

### 19.5 Detection reference wording

The native NSLSR frame is a **reference arm**, not a guaranteed "clean upper bound." Restored imagery may in principle yield higher detector mAP than the native reference; that result would not be contradictory.

---

## 20. Verified references

* Y. Cui et al., *AdaIR: Adaptive All-in-One Image Restoration via Frequency Mining and Modulation*, ICLR 2025. arXiv:2403.14614. Code: https://github.com/c-yn/AdaIR (MIT).
* Y. Guo et al., LSFDNet / NSLSR dataset: https://github.com/Yanyin-Guo/LSFDNet (MIT). README documents 1,205 registered SWIR–LWIR pairs at 640×512 and the four-folder split. Dataset hosted on Baidu Pan.
* H. Jiang, H. Wang, Y. Zheng, *See the Invisible with SWIR: Learning to Enhance via Synthetic Noise Modeling*, ICCV Workshops 2025 (Responsible Imaging). CVF open access.
* AdaIR upstream issue #8 (FFT/AMP) — `[UNVERIFIED: referenced in v1, not independently checked]`.

Bibliographic metadata for AirNet, PromptIR, SwinIR/HAT and Real-ESRGAN is **not** included here because it was not verified in this pass. Mark as `[CITATION REQUIRED]` until checked against the publisher record.

---

## 21. Most important next action

Before any training: audit the **unmodified** upstream `FreModule` at the intended resolutions. Record the integer-mask occupancy, `para1` initialization, and every parameter that actually participates in `forward`. Then run the corrected isolated differentiability test for Soft-FMiM with `para1` temporarily set to 1.0. This establishes the released-code behavior without confusing zero-initialized residual gating with mask non-differentiability, and it determines the cleanest controlled baseline for the paper.
