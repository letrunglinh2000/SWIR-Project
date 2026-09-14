# SWIR-AdaIR-SR — M0/M1/M2 evidence log

Spec: `SWIR_AdaIR_SR_SPEC.md` v2.1. Upstream tag: `upstream-baseline`
(`2f6bed79b2b801338515f71983e95e822cba4af3`, content of `c-yn/AdaIR @ ccb8b98`).
Environment: `/data2/letrunglinh/conda_env_linh/linh` — Python 3.11.8,
torch 2.6.0+cu124, scikit-image 0.25.2, 4x A100 80GB.

---

## §21 — Upstream audit (the "most important next action")

Run against the **unmodified** upstream `FreModule`, before any repair.

### 1. Released integer-mask occupancy

| Module @ resolution | feature map | low-band \|·\|mean | high-band \|·\|mean |
|---|---|---|---|
| fre1 @128px | 16x16 | **0.000000** | 0.4398 |
| fre2 @128px | 32x32 | **0.000000** | 0.4240 |
| fre3 @128px | 64x64 | **0.000000** | 0.4557 |
| fre3 @512x640 | 256x320 | 0.001492 | 0.4447 |

The spec's predicted table is **confirmed**, including that fre3 becomes non-zero
only at full frame. At every patch size actually used, the released low-frequency
branch is identically zero.

Per §5, this is recorded as **released-code behaviour / implementation
limitation**, not as a "bug", in any paper text.

### 2. Dead parameters on this commit

`self.conv` and `self.score_gen` are allocated in `FreModule.__init__` and
**never referenced** in `forward`. `conv1`, `rate_conv`, `para1`, `para2` are
live. Both dead modules are therefore **not carried over** into `SoftFreModule`
(asserted by `tests/test_fremodule_channels.py::test_dead_upstream_modules_not_carried_over`).

### 3. `para1` initialisation

`para1` is initialised to **exactly zero** (`para2` = 1). This is why a naive
`model(x).mean().backward()` gradient test is invalid: the frequency branch is
gated off at step 0 regardless of mask differentiability. The corrected test
(`test_D_cutoff_predictor_is_differentiable`) sets `para1 = 1.0` first.

---

## §5 — Defect status

| ID | Status | Evidence |
|---|---|---|
| D-1 `FreModule` channel mismatch | fixed | `tests/test_fremodule_channels.py` (3 tests) |
| D-2 degenerate / non-differentiable mask | fixed as **Soft-FMiM**, with the released behaviour retained as `upstream_integer` | `tests/test_fre_mask.py` (6 tests) |
| D-3 AMP + FFT | fixed, fp32 scoped to the spectral ops only | `tests/test_amp_fft.py` (9 tests, real GPU, fp16 + bf16) |
| D-4 unused `noise_emb` argument | removed; API is `forward(x, out_scale)` | `tests/test_shapes.py` |

Soft-FMiM is registered as an **explicit method variant**, never as stock AdaIR
and never as a silent repair (§17.16, §19.2). `test_upstream_integer_mode_gets_no_cutoff_gradient`
pins the contrast that motivates ablation A2: the released mask yields **no**
gradient to `rate_conv`; the soft mask does.

---

## M1 — Architecture gate: **PASS**

- [x] 1-channel in/out, `FreModule(in_dim=1)` wired
- [x] D-2 assertions: degeneracy demonstrated on the original, non-degenerate mask and non-zero `rate_conv` gradient after the fix
- [x] D-3 forward+backward under real AMP on the target GPU, all finite
- [x] gradient-coverage audit across `s=1` **and** `s=2`; no unintended dead trainable parameter
- [x] scale-homogeneous batching; a batch never mixes `s=1`/`s=2` spatial sizes
- [x] shapes 128→256, 256x320→512x640, and the `s=1` identity cases
- [x] padding wrapper handles non-multiple-of-16 inputs
- [x] tiled inference seam-checked, delta vs whole-frame **reported not asserted equal**
- [x] params / MACs / latency / peak memory at both resolutions

`63 passed` (`pytest tests/ -q`) — 54 for the M1 gate plus 9 covering the
distributed-training invariants (rank-invariant scale draw, disjoint index
partition, and a real gloo process group proving the cross-rank guard fires).

Base gate is identity-preserving at init (`g=1`, `b=0`) — asserted, not assumed.

### Efficiency (§11.1)

| input HxW | scale | GMACs | ms/frame | peak MiB |
|---|---|---|---|---|
| 256x320 | 1 | 183.7 | 233.8 | 775 |
| 256x320 | 2 | 183.9 | 239.6 | 775 |
| 512x640 | 1 | 734.8 | 913.5 | 2733 |
| 512x640 | 2 | 735.7 | 914.0 | 2733 |

**Latency is NOT a valid deployment number**: the GPU was shared with 4 other
compute processes during measurement (recorded in `runs/bench.json`). Re-run on
an idle GPU before quoting. Params 28.76 M — §17.10 forbids calling this
"lightweight" without a stated budget it meets (GAP-1 is still open).

---

## M2 — Data gate: **BLOCKED** (one blocking finding, one open question)

- [x] format inspection complete, `configs/data_profile.yaml` committed
- [x] `datasets/MANIFEST.sha256` written (1044 files under the segment split)
- [x] deterministic degradation test passes (same seed → identical φ and output)
- [x] batch-scale sampler determinism tested
- [x] HR-crop → aligned-LR alignment test passes
- [ ] **leakage audit** — see below
- [ ] E1 converges and beats E0 by more than the seed spread — not yet run
- [x] best-checkpoint selection by validation PSNR works; config+seed+commit snapshotted

### BLOCKING: the shipped folders are not an independent split

`tools/leakage_audit.py` on the shipped folders:

| measure | value |
|---|---|
| `NSLSR_test` ∩ `NSLSR_val` | **118 / 118 — test is a strict SUBSET of val** |
| test frames with an immediately adjacent train frame | **116 / 118** |
| val frames with an immediately adjacent train frame | 356 / 361 |
| pHash near-duplicate pairs (test vs train) | **34 / 118**, min Hamming **0** |
| nearest-neighbour thumbnail MAD | min 0.0056 |

The folders stride one contiguous capture sequence — test = every 10th frame,
val = every ~4th, train = the rest. Two consequences:

1. Selecting a model on `NSLSR_val` **is** selecting it on `NSLSR_test`, which
   violates §17.12 by construction.
2. Test frames are temporally adjacent to training frames, so any test PSNR is
   inflated.

**Remedy implemented.** `tools/make_splits.py --scheme segment` partitions the
frame-id axis into contiguous 40-frame segments, assigns whole segments to
train/val/test, and drops a 2-frame guard band at each segment edge:

| | shipped folders | segment split |
|---|---|---|
| train / val / test | 803 / 361 / 118 | 756 / 144 / 144 |
| id overlap between any two splits | 118 (val∩test) | **0** |
| held-out frames adjacent to a train frame | 116 + 356 | **0** |
| pHash near-duplicate pairs flagged | 34 | **2** |

`configs/splits.json` is the segment split and is what `configs/swir_base.yaml`
uses. `--scheme original` reproduces the shipped folders for comparison with
published detection numbers and is labelled `expected_to_fail` in the manifest.

**Still open — a judgement call for the PI.** Two pairs remain flagged at the
default thresholds (`0714~0706`, Hamming 2, MAD 0.0234; `0715~0708`, Hamming 4,
MAD 0.0418). They are 6–8 frames apart across a segment boundary in a
near-static sea/sky scene. Widening the guard to 10 removes one of them but
costs a third of the training data (756 → 524). The options are: accept and
disclose the two pairs, raise `--guard`, or justify a looser MAD threshold for
near-static maritime scenes. **I have not loosened the threshold to force a
pass** — the audit still reports FAILED at the strict default.

### Frame-count discrepancy

§9.1 quotes 1,205 pairs `[VERIFIED from LSFDNet README]`. On disk there are
**1,164 distinct frame ids** spanning 0..1204, with **41 ids present in no
split** (e.g. 246–248, 321–327). `NSLSR320_val` is absent from this checkout.
Recorded in `configs/data_profile.yaml` (`total_matches_spec: false`); the
quoted number should not be reused until reconciled.

---

## §9.2 — Encoding (drives the loader; §17.5 satisfied)

| field | value |
|---|---|
| dtype / bit depth | `uint8` / 8 |
| shape | 512x640x3, PIL mode `RGB` |
| channels identical | **no** — worst-file mean deviation 0.455, p99.9 6.0, max 8 levels |
| channel rule | `take_channel_0` |
| normalisation | `divide_by_255` |

The deviation is far below genuine multi-band data (which differs by tens of
levels); this is JPEG chroma subsampling of a grayscale source. The decision is
made from the **distribution**, not the maximum, and every statistic is recorded
in the profile so it is auditable. `.convert('L')` / `.convert('RGB')` are never
called on SWIR files (§17.5).

---

## §9.2 addendum — the band is luma, not a channel

The shipped SWIR JPEGs are 3-component (`SOF0`: Y 2×2, Cb/Cr 1×1, i.e. 4:2:0), which is
why PIL reports `RGB`. Chroma is neutral — Cb 127.96 ± 0.28, Cr 127.99 ± 0.10, 99.2% of
pixels within ±1 of 128 — so the content is luma, and the residual channel spread is
chroma-quantisation noise from the subsampled 4:2:0 path.

Distance from true luma over 60 frames:

| channel | mean (levels) | worst max |
|---|---|---|
| B — `cv2` channel 0 | 0.131 | 6 |
| G — `cv2` channel 1 | 0.107 | 5 |
| R — `cv2` channel 2 = PIL channel 0 | 0.102 | 5 |

`take_channel_0` under `cv2` is therefore the **blue** channel, the worst of the three
(`B = Y + 1.772(Cb−128)`). The rule is now `luma`, and `datasets/NSLSR_gray` holds a
lossless 1-band PNG re-container produced through the same `reduce_channels` the loader
uses. Verified: max difference 0.0 in [0,1] units across all three splits, so no earlier
measurement changes. 21/60 frames have R ≠ B by up to 7 levels, so the distinction is
real, not theoretical.

Upstream context: LSFDNet's own loader is `cv2.imread(f)` + `/255` with 3 channels
throughout — they never collapse the band, which explains the storage format.

---

## §12 — Detection label format: CONFIRMED

361 label files, 836 boxes, single class `0`, 5 fields per line, all of
`cx cy w h` within [0,1] → **normalised YOLO**. Rescaling the image therefore
requires no label transform. `detect_eval.py` implements arms A0–A4 with a
frozen detector and no gradient path into the restoration model; it is M3 work
and needs §18 Q7 answered (which detector, trained on what).

---

## Not yet done

E1–E6, ablations A1–A7, detection arms, the real-degraded arm, and the ≥3-seed
headline table are all **M2-gated or M3** and have not been run. No training
numbers are reported anywhere in this document, per §17.6.
