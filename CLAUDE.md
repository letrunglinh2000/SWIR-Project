# SWIR-AdaIR-SR

All-in-one restoration **and** ×2 super-resolution for single-band SWIR imagery, built on
the AdaIR backbone and trained on the NSLSR maritime dataset. Output is for an academic
paper, so provenance and honesty about what has and has not been measured matter more
than speed.

---

## 0. Read this first

**`SWIR_AdaIR_SR_SPEC.md` (v2.1, 865 lines) is the governing document.** It encodes
decisions already made by the PI. Read the relevant section before editing; do not
deviate silently. Where this file and the spec disagree, the spec wins.

**Use the right interpreter.** The base conda env has torch 1.13 and is missing
`lightning`, `wandb` and `skvideo`. Everything here is developed and tested against:

```
/data2/letrunglinh/conda_env_linh/linh/bin/python     # 3.11.8, torch 2.6.0+cu124, skimage 0.25.2
```

`skvideo` is absent in both envs — that is why `utils/val_utils.py` imports it lazily.

**The GPUs are usually shared.** Four A100 80GB, routinely ~97% busy with another user's
job leaving ~13 GB free. Size smoke tests to fit; never kill the other process. Any
latency number measured under contention is invalid — `tools/bench.py` records this
automatically.

---

## 1. The hard invariants

These come from spec §1 and §17. They are the ones that actually bite:

- **One architecture, one checkpoint, all degradations and both output scales.** No
  per-degradation checkpoints.
- **The model never receives a degradation label.** `forward(x, out_scale)` only.
  `out_scale` is a deployment setting, not a degradation class.
- **One optimizer batch = one output scale.** `s=1` and `s=2` produce different tensor
  sizes from the same HR crop, so a scale is drawn per *batch*, never per example. Under
  DDP this must hold *across ranks too*: the sampler draws the scale from a stream seeded
  only by `(seed, epoch)`, before the global batch is sharded, so it cannot depend on
  rank. `assert_scale_synchronised` re-checks it every step with one small all-reduce.
- **No hard-coded spatial size or feature width.** Head input is `int(dim * 2**1)`,
  derived.
- **Upstream AdaIR stays runnable and unmodified** at tag `upstream-baseline`.
  `net/model.py` must show an empty diff against it.
- **Never report a number** whose config, seed and commit are not logged.
- **Never report an aggregate PSNR** without the per-degradation and per-scale breakdown.
- **Never tune on the test split.**
- **Never call `sim_B` results "external validity"** — it is simulator-shift robustness.
- **Never call the model "lightweight" or "real-time"** without a stated budget it meets.

---

## 2. Layout — what is upstream and what is ours

Upstream AdaIR is preserved verbatim so the released baseline remains a controlled
comparison arm. Everything the project adds lives in new files.

```
net/model.py                 UPSTREAM — do not touch (empty diff vs upstream-baseline)
net/swir_adair_sr.py         SWIRAdaIRSR, SoftFreModule, scale heads, base gate, tiling
utils/swir_io.py             bit-depth-aware IO, driven by configs/data_profile.yaml
utils/swir_degradation.py    modular D_s, sim_A/sim_B, per-component RNG, metadata
utils/swir_sr_dataset.py     NSLSRSWIRDataset + scale-homogeneous batch sampler
utils/swir_metrics.py        pinned PSNR / SSIM / edge / LPIPS
utils/swir_losses.py         L1 baseline + optional SSIM / edge / frequency terms
configs/                     swir_base.yaml, sim_b.yaml, data_profile.yaml, splits.json
tools/                       inspect_dataset, make_splits, leakage_audit, bench,
                             to_single_channel (3-ch JPEG -> 1-ch PNG),
                             preview_degradation (dump what the model is fed)
tests/                       63 tests — M1 gate + DDP invariants
train_swir.py test_swir.py infer_swir.py detect_eval.py
runs/M1_M2_EVIDENCE.md       the gate evidence log (tracked; rest of runs/ is not)
```

`utils/val_utils.py` carries the only edit to an upstream file: `skvideo` made optional and
the SSIM kwarg selected by signature. Without it the upstream scripts cannot even import
under skimage 0.25, so this change is what makes the "upstream stays runnable" invariant
satisfiable.

---

## 3. The data story

### What NSLSR actually is

A **detection** dataset reused as a restoration source: 1,164 distinct frames of maritime
SWIR + LWIR pairs, pixel-registered, with single-class (ship) YOLO labels. There are **no
clean/degraded pairs**, so the stored frame is taken as the restoration target and the
input is synthesised. The stored frame is a *captured* image with native sensor noise — it
is never "ground truth", and a ×2 output never contains physically measured detail.

LWIR is present and unused; it is a future fusion extension.

### Encoding (§9.2) — settled, versioned in `configs/data_profile.yaml`

**The active tree is `datasets/NSLSR_gray` — true single-band 8-bit PNG.**
`datasets/NSLSR` (the shipped tree) is kept unmodified for detection reproduction.

The shipped SWIR frames are **3-component JPEGs**: the `SOF0` header declares Y/Cb/Cr with
2×2 luma and 1×1 chroma sampling (4:2:0), so PIL reports mode `RGB`. The content is luma
only — Cb and Cr sit at 128 ± 0.3, and 99.2% of pixels are within ±1 of neutral — but the
container forces every reader to decide how to collapse three channels, and chroma
quantisation leaks a few levels of false colour near edges.

**The band is recovered as BT.601 luma, not by picking a channel.** Every RGB channel is
luma plus a chroma-quantisation error, and `cv2` decodes **BGR**, so `take_channel_0` is
the *blue* channel — which carries the largest chroma coefficient
(`B = Y + 1.772(Cb−128)` against `1.402` for R) and is the worst of the three. Measured
over 60 frames, distance from true luma: **B 0.131, G 0.107, R 0.102** levels on average,
up to 6; and 21 of 60 frames have R ≠ B by up to 7 levels. This bit me once already: a
converter that read with PIL (RGB) while the loader read with cv2 (BGR) silently took a
different channel.

`tools/to_single_channel.py` writes the PNG tree through **the same
`utils.swir_io.reduce_channels` the loader uses**, so the two cannot drift, and verifies
every frame round-trips exactly. Verified after conversion: max difference 0.0 across all
three splits, so nothing previously measured is invalidated. Cost: 103 MB → 281 MB
(PNG is lossless; re-encoding to JPEG would stack a second lossy generation onto an
already-lossy image and teach the model to undo compression artefacts).

Never call `.convert('L')` or `.convert('RGB')` on a SWIR file; go through
`utils/swir_io.read_image`, which reads the profile.

**Note on LSFDNet (the dataset's own code):** their loader is `cv2.imread(f)  # BGR`
followed by `/255`, feeding **3 channels** straight into a YOLO/BasicSR stack. They never
collapse to one channel, which is why the dataset was stored as 3-component JPEG in the
first place. Our 1-channel treatment is a deliberate departure — state it if we ever
compare against their reported numbers.

### The degradation operator — what the model actually sees

NSLSR has no degraded/clean pairs, so `D_s` synthesises the input from the target.
Fixed order, each component with its own RNG stream:

```
HR crop
  1. low-light      p=0.45   x <- alpha * x^gamma     alpha in [0.15,0.8], gamma in [1.0,2.2]
  2. backscatter    p=0.35   x <- x*t + A(1-t)        horizon-biased t, UNCALIBRATED
  3. blur           p=0.70   x <- x (*) k             iso/aniso Gaussian, sigma in [0.2,2.0]
  ---------------- sensor boundary ----------------
  4. downsample     ALWAYS   s=1 identity | s=2 exactly one /2 (area|bilinear|bicubic)
  5. sensor noise   p=0.60   Poisson shot + read + column/row/FPN structure (v2)
  6. quantise       ALWAYS   8-bit, from the data profile
LR -> model input
```

Steps 1-3 are the world (scene, atmosphere, optics) and happen at HR; steps 5-6 are the
detector and happen after the downsample, because they belong to the sensor reading out at
its own pixel pitch. Adding noise before the decimation would let the averaging suppress
it -- an easier, unphysical problem.

**Degradations compose; they are not one-per-image.** The flags are drawn independently per
example, so the model sees arbitrary combinations and is never told which are present.
Measured over 756 training samples:

| active degradations | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| share | 4.8% | 23.7% | **39.7%** | 25.5% | 6.3% |

Two at once is the common case. The ~5% clean fraction is deliberate -- the model must
learn not to over-correct an already-good frame.

**`out_scale` is NOT one of these flags.** It is drawn per *batch* and passed to
`forward(x, out_scale)`; super-resolution is a deployment setting, not a degradation class
(INV-3). So a single sample can be low-light + blurred + noisy *and* be asked to double
resolution in the same forward pass.

**The seven evaluation subsets pin the flags** (`EVAL_SUBSETS` in `utils/swir_degradation.py`)
so each degradation gets an isolated score; only `composite` uses the training
distribution. This is why an eval preview looks like one degradation per image while
training does not -- do not mistake the former for the latter.

`sim_A` trains. `sim_B` (motion kernels, noise v1, smooth transmission, different
decimation filters) is evaluation only; the A->B gap is simulator-shift robustness, never
external validity.

To look at any of this rather than reason about it:

```bash
$E tools/preview_degradation.py --subsets composite --index 0 1 2 3   # real training samples
$E tools/preview_degradation.py --n 4                                  # all seven subsets
$E tools/preview_degradation.py --sim_b configs/sim_b.yaml             # the held-out simulator
```

Panels go to `runs/preview/`; the leftmost is the HR target, the rest are model inputs
(nearest-upscaled at s=2 so the lost detail stays visible), captioned with the flags that
were actually sampled.

### The split defect — this is the big one

**The shipped folders are not an independent split.** `tools/leakage_audit.py` found:

| | shipped folders | segment split |
|---|---|---|
| train / val / test | 803 / 361 / 118 | 756 / 144 / 144 |
| id overlap | **118 — `NSLSR_test` ⊂ `NSLSR_val`** | 0 |
| held-out frames adjacent to a train frame | **116 + 356** | 0 |
| pHash near-duplicate pairs | **34 / 118**, min Hamming **0** | 2 / 144 |

The folders stride one contiguous capture sequence: test = every 10th frame, val = every
~4th, train = the rest. Two independently disqualifying consequences — selecting on val
*is* selecting on test, and the network has effectively seen every test frame.

`tools/make_splits.py --scheme segment` is the remedy: contiguous 40-frame segments
assigned whole, with a 2-frame guard band at each edge. `configs/splits.json` is that
split and is what `configs/swir_base.yaml` uses. `--scheme original` reproduces the
shipped folders for comparison with published detection numbers and is labelled
`expected_to_fail` in its own manifest.

**Still open (needs the PI):** two pairs remain flagged at the strict default
(`0714~0706`, `0715~0708`) — 6–8 frames apart across a boundary in a near-static sea
scene. `--guard 10` clears one but costs a third of the training data (756 → 524). Accept
and disclose, widen the guard, or justify a looser MAD threshold. **Do not loosen the
threshold just to make the audit pass.**

**Frame-count discrepancy:** the spec quotes 1,205 frames `[VERIFIED from LSFDNet README]`;
on disk there are 1,164 distinct ids spanning 0..1204, with 41 ids in no split.
`NSLSR320_val` was present earlier in the project and has since been deleted. Unreconciled
— do not reuse the 1,205 figure.

---

## 4. The method story

### Four upstream defects (spec §5), all fixed with assertions

- **D-1** `FreModule` built its input conv with a hard-coded `in_dim=3`; `AdaIR` never
  forwarded `inp_channels`. Fixed; asserted.
- **D-2** *The important one.* The released mask half-width is `(h // 128) * threshold`,
  which truncates to **0** for every feature map this project uses. Measured on the
  unmodified commit: low-band `|·|mean` is **exactly 0.000000** at fre1/fre2/fre3 for a
  128px patch, and 0.001492 at fre3 only at full frame. The integer cast also means
  `rate_conv` receives no gradient — the "learned" cutoff never learned anything.
- **D-3** `torch.fft` under autocast. Fixed by scoping fp32 to the spectral ops only, not
  the whole block.
- **D-4** unused `noise_emb` argument removed.

Also confirmed on that commit and deliberately **not** carried over: `self.conv` and
`self.score_gen` are allocated but never referenced in `forward`.

### Soft-FMiM

`soft_relative` replaces the integer box with `M = σ((r − ρ)/τ)`, where
`ρ = sqrt((2u/h)² + (2v/w)²)` is 0 at DC and 1 at Nyquist — so the cutoff is
resolution-relative and differentiable.

**This is an explicit architectural variant, never "stock AdaIR" and never described as a
silent bug fix.** Both modes ship and are switchable (`model.frequency_mask.mode`) because
ablation A2 compares them. In internal logs and paper text, describe the released
behaviour as an *implementation limitation* or *released-code behaviour*, not a "bug".

### The gradient test that would have lied

`para1` initialises to **exactly zero** upstream, so the frequency branch is gated off on
the first backward pass regardless of the mask. A test calling
`model(x).mean().backward()` and expecting `rate_conv` gradient fails for the wrong
reason. The real test sets `para1 = 1.0` first. See
`tests/test_fre_mask.py::test_D_cutoff_predictor_is_differentiable`.

---

## 5. How to run it

Order matters — the loader refuses to run without a data profile.

```bash
E=/data2/letrunglinh/conda_env_linh/linh/bin/python

$E tools/inspect_dataset.py                    # §9.2 -> configs/data_profile.yaml
$E tools/make_splits.py                        # §9.1/9.3 -> configs/splits.json + MANIFEST.sha256
$E tools/leakage_audit.py --splits_json configs/splits.json \
     --query_split test --ref_splits train     # blocking gate
$E tools/preview_degradation.py --n 4          # eyeball what D_s produces
$E -m pytest tests/ -q                         # M1 gate + DDP invariants, 63 tests
$E tools/bench.py                              # §11.1 — idle GPU only

$E train_swir.py --config configs/swir_base.yaml           # single GPU
$E test_swir.py --ckpt runs/base/best.pth      # full subset x scale x simulator grid
$E infer_swir.py --ckpt runs/base/best.pth --input <dir> --output <dir>
```

Multi-GPU (DDP, one process per GPU):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 $Ebin/torchrun --standalone --nproc_per_node=4 \
    train_swir.py --config configs/swir_base.yaml
```

`train.batch_size` is **per GPU**; the effective batch is `batch_size x world_size`, and
both are recorded in `env.json` along with `steps_per_epoch`. Rank 0 alone validates,
checkpoints and writes the logs; the other ranks wait on a barrier. Checkpoints are saved
from the unwrapped module, so a DDP-trained `best.pth` loads into `test_swir.py`
unchanged. Scaling the LR with world size is **not** done automatically — decide it
deliberately and log it.

Debug flags: `train_swir.py --max_steps N --epochs 1`, and
`eval.val_full_frame: false` + `eval.val_crop` in the config when the GPU is contended
(full-frame `s=2` validation is 1024×1280 and will OOM on a shared card).

---

## 6. Where the project stands

| Gate | State | Note |
|---|---|---|
| M0 upstream preserved | **pass** | tag `upstream-baseline`, `net/model.py` empty diff |
| M1 architecture | **pass** | 63 tests; D-1…D-4 fixed; shapes, padding, AMP, tiling, gradient coverage, DDP scale sync |
| M2 data | **blocked** | leakage audit; E1 convergence run not yet done |
| M3 experiments | not started | E1–E6, A1–A7, detection arms, ≥3 seeds |

**No training numbers exist yet.** Everything run so far has been smoke tests of ≤12
steps. Do not quote a PSNR from this repo until E1 has actually converged under a logged
config, seed and commit.

Efficiency measured (28.76 M params): 183.7 GMACs @256×320, 734.8 GMACs @512×640,
peak 775 / 2733 MiB. **Latency is not quotable** — measured with four other processes on
the GPU.

---

## 7. Open questions blocking progress

From spec §18; these need the PI, not a code change:

1. Does the PI accept a unified backbone, or are the three proposal figures contractual
   (GAP-2…GAP-6)?
2. What is the numeric "lightweight" budget — params, MACs, or ms/frame on what hardware?
3. For single-channel SWIR, does "채널별 조도 적응" mean *feature* channels?
4. Does the project camera output 640×512, at what bit depth and frame rate?
5. Is any real higher-resolution SWIR ground truth obtainable? Without it the ×2
   deployment output is unverifiable by construction.
6. NSLSR data licence and redistribution terms.
7. Detection: which detector, trained on what, and is it frozen? (`detect_eval.py` is
   written but cannot run without this.)
8. Target venue and deadline.
9. Release policy for the checkpoint and the simulator.
10. Can project-camera dark frames / flat fields / controlled smoke captures be obtained?
    These decide whether the degradation simulator can ever be calibrated.

Plus the guard-band decision in §3 above.

---

## 8. Conventions that are easy to get wrong

- **H×W everywhere in code**, tensors and tables. `W×H` only when quoting the dataset's
  own "640×512".
- Tensors are `float32`, `[B,1,H,W]`, range `[0,1]`.
- **Never clamp the model output during training** — it kills gradients in the saturated
  region. Clamp only for metrics and for saving.
- Crop HR first, then degrade. Never crop LR and HR independently.
- Geometric augmentation happens on the HR crop *before* degradation and never inside the
  degradation module. Vertical flip is disabled while the backscatter field is
  horizon-biased.
- The crop seed is deliberately independent of `out_scale` so index *i* gives the same HR
  content at both scales; only the degradation seed varies with scale. Otherwise the E6
  scale-sharing comparison is confounded.
- `D_2` contains **exactly one** ×2 reduction and `D_1` contains none. Never write
  `downsample(D(x))` — that silently creates a ×4.
- The degradation `meta` dict is for stratified evaluation and logging **only**. It never
  reaches the model.
- Training-time validation uses `sim_A` only, for cost. `test_swir.py` runs both
  simulators — that is where the reported grid comes from.
- `strict=False` weight loading without an inspected key report is forbidden;
  `load_pretrained_partial` prints loaded / adapted / skipped / randomly-initialised keys.

---

## 9. Git

`main`, single commit `first commit`, remote `origin` = `letrunglinh2000/SWIR-Project`.
Upstream `c-yn/AdaIR` is **not** a remote any more — re-add it as `upstream` if you need to
pull their changes. Do not add `Co-Authored-By` trailers to commits
(`includeCoAuthoredBy: false` is set in `~/.claude/settings.json`).

`datasets/` is gitignored. `runs/` is too, **except** the deliverables:
`M1_M2_EVIDENCE.md`, `leakage_audit*.json` and `bench.json` are tracked, because a
milestone claim with no committed evidence is worth nothing.
