# ffsplat

[![Release](https://img.shields.io/github/v/release/w-m/ffsplat)](https://img.shields.io/github/v/release/w-m/ffsplat)
[![Build status](https://img.shields.io/github/actions/workflow/status/w-m/ffsplat/main.yml?branch=main)](https://github.com/w-m/ffsplat/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/w-m/ffsplat/branch/main/graph/badge.svg)](https://codecov.io/gh/w-m/ffsplat)
[![Commit activity](https://img.shields.io/github/commit-activity/m/w-m/ffsplat)](https://img.shields.io/github/commit-activity/m/w-m/ffsplat)
[![License](https://img.shields.io/github/license/w-m/ffsplat)](https://img.shields.io/github/license/w-m/ffsplat)

ffsplat is a powerful framework to convert and compress 3D Gaussian Splatting scenes.

Fork of [w-m/ffsplat](https://github.com/w-m/ffsplat) (Apache-2.0). This branch adds Apple-Silicon
(MPS) support and depends on an Apple-Silicon port of gsplat
([zacsmms/mpsplat](https://github.com/zacsmms/mpsplat), Apache-2.0) and on
[PLAS](https://github.com/fraunhoferhhi/PLAS) (Apache-2.0), each pinned to a revision and each
keeping its own licence; no third-party code is vendored here.

**NOTE: this code is pre-alpha, under heavy development, a community developer preview. Please expect more documentation in the coming weeks (July 2025).**

- **Github repository**: <https://github.com/w-m/ffsplat/>

**Apple Silicon:** this fork runs on Macs through PyTorch's MPS backend - no CUDA toolkit
required; see [Apple Silicon (MPS) support](#apple-silicon-mps-support).

## Quick Start

- Install CUDA 12.x (Linux/NVIDIA; not needed on Apple Silicon - see below)
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

Run

`uvx --from git+https://github.com/w-m/ffsplat@main ffsplat live --help`

Then open a web browser to [localhost:8080](http://localhost:8080) (or the address that is printed in the viser box on stdout). On the first launch, it will take a few minutes to compile gsplat kernels before a rendered image shows up.

## Setting up a dev environment

Check out the project:

```
git clone https://github.com/w-m/ffsplat.git
cd ffsplat
```

(Apple Silicon: clone `https://github.com/HenryGetz/ffsplat-mps` instead and skip `make install` -
`uv sync` is the whole install; see [Apple Silicon (MPS) support](#apple-silicon-mps-support).)

Set up the virtual environment with uv:

```bash
make install
```

Run the live encoding:

```bash
uv run ffsplat live \
    --input /data/gaussian_splatting/mini-splatting2/truck_sparse/point_cloud/iteration_18000/point_cloud.ply \
    --input-format=3DGS-INRIA.ply \
    --dataset-path /data/gaussian_splatting/tandt_db/tandt/truck/ \
    --verbose
```

Check the types and formatting of the code base:

```bash
make check
```

## Apple Silicon (MPS) support

This branch (`mps-support`, branched from upstream `3fb6444`) makes ffsplat run on Apple-Silicon Macs
through PyTorch's MPS backend, and it is the content of this repository's `main`. It was developed and
verified on an **M1 Max MacBook Pro — 10 CPU cores, 32 GiB unified memory, macOS 27.0.0, arm64**,
Python 3.13.12, torch 2.14.1. Every number below was measured on that machine; units are stated.
Numbers labelled _baseline_ were measured before the optimisation work that followed; everything else
supersedes them. The full measurement record, the evidence images and a vendored/offline variant live
on the [`apple-silicon` branch](https://github.com/HenryGetz/ffsplat-mps/tree/apple-silicon).

### Install (Apple Silicon)

No CUDA toolkit is involved. `uv sync` is the whole install step:

```bash
git clone https://github.com/HenryGetz/ffsplat-mps.git
cd ffsplat-mps
uv sync
uv run ffsplat convert --help
```

Requires **macOS 13+ (Apple Silicon)**, **uv >= 0.4** (the install uses `[tool.uv] override-dependencies`),
and **Python 3.12-3.13**; `torch 2.14.1` (on Apple Silicon; every other platform keeps upstream's
`2.6.0`) is pinned and installed by `uv sync`. A full install needs no local files or paths: ~30 s with
a warm uv cache, ~90 s with a cold one on this machine (measured on a fresh clone).

> Don't have a PLY to try? Any 3D Gaussian Splatting `.ply` works, e.g. one of the pretrained
> [INRIA](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) or
> [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) scenes - pass it as `--input` with
> `--input-format 3DGS-INRIA.ply`.

What changed for this:

- `cupy-cuda12x`, `torchpq` and the `pytorch-cu126` wheel index were dropped, and the SOG
  spherical-harmonics k-means was reimplemented in pure PyTorch (`src/ffsplat/models/clustering.py`),
  so it runs on CPU and MPS.
- `torch` is `2.14.1` **on Apple Silicon only** (upstream's `2.6.0` pin is kept everywhere else):
  the MPS rasterizer needs `torch.mps.compile_shader`, which 2.6.0 does not have.
- The differentiable rasterizer now comes from **[mpsplat](https://github.com/zacsmms/mpsplat)**
  (gsplat 1.5.3 ported to Metal), selected per platform: `gsplat==1.4.0` for everything but darwin,
  `mpsplat` for darwin. Both ship the same top-level package name, so `import gsplat` is unchanged.
  mpsplat publishes no wheels, and its own `MANIFEST.in` omits the 14 `.metal` sources its kernels are
  read from, so an install built from upstream currently fails on the first render with
  `FileNotFoundError: .../gsplat/mps/_shaders/projection_ewa_3dgs_fused_fwd.metal` (reproduced; fix
  proposed as [zacsmms/mpsplat#1](https://github.com/zacsmms/mpsplat/pull/1)). The pin is therefore a
  fork revision that ships them, and it moves back to `zacsmms/mpsplat` once that is merged.
- Every hard-coded CUDA device site is behind one helper, `ffsplat.device.resolve_device()`:
  MPS if available, else CUDA if available, else CPU. `cuda` is only ever selected when
  `torch.cuda.is_available()`, so Linux/CUDA behaviour is unchanged.
- `plas`'s `lapjv` dependency is dropped on macOS only (`[tool.uv] override-dependencies`): it is used
  exclusively by PLAS's upstream evaluation script, which ffsplat never imports, and no published
  `lapjv` version builds on Darwin.

### Device policy: what runs where, and why

MPS is selected automatically (`resolve_device()`), and every stage that has a GPU implementation runs
there. The stages that are on the CPU are CPU _because there is no GPU implementation in this stack_,
not by accident:

| stage                                                                         | device  | measured basis                                                                                                                                                                                                                                                                                                      |
| ----------------------------------------------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PLY read/decode, `container_meta.yaml` + PLY write                            | **CPU** | file/parse work (numpy `fromfile` + tensor assembly); no GPU equivalent. The 630 MB full-size PLY parses in **0.46 s** (in-process: `decode_gaussians` 0.44 s + one host-to-device copy 0.07 s; measured with `perf_counter` around `decode_gaussians(..., '3DGS-INRIA.ply')`), so this stage is 0.1% of the encode |
| remap / quantise / byte-split / reindex / flatten (the SOG codec's torch ops) | **GPU** | native MPS torch kernels; 0.13 s of the full-size encode                                                                                                                                                                                                                                                            |
| PLAS spatial sort                                                             | **GPU** | **1.8x faster on MPS at 512², 2.2x at 1594²** on the template's real 6-channel input (30.4 s vs 55.7 s; 278.7 s vs 621.6 s). The CPU retry in `PLAS._sort_with_fallback` only fires if MPS raises, and never has                                                                                                    |
| k-means codebook, `euclidean` (SOG-web/-png)                                  | **GPU** | `torch.cdist`; 22.3 s vs 66.4 s on CPU for the real codebook input. GPU utilisation **99.7%**                                                                                                                                                                                                                       |
| k-means codebook, `manhattan` (SOG-PlayCanvas)                                | **GPU** | 48.0 s vs 38.2 s on CPU per assignment pass — the one stage where the CPU is faster (see _Where MPS is not optimized_)                                                                                                                                                                                              |
| image encode (AVIF / PNG / WEBP)                                              | **CPU** | `cv2.imwrite` / Pillow / pillow-heif are host code and there is no GPU codec in this stack; GPU utilisation during these windows is 26.7% (i.e. idle). 4.8 s of the full-size encode                                                                                                                                |
| rasterisation (`view`, `live`, `eval`)                                        | **GPU** | mpsplat's Metal kernels; `tests/test_metal_path.py` fails if a call would take the torch fallback                                                                                                                                                                                                                   |
| eval metrics (PSNR/SSIM/LPIPS)                                                | **GPU** | `torchmetrics` on MPS, no CPU fallback needed                                                                                                                                                                                                                                                                       |
| COLMAP undistort/remap (eval loader)                                          | **CPU** | `cv2.initUndistortRectifyMap` + `cv2.remap`; measured on a real 1957x1091 truck image: **3.2 ms + 1.6 ms = 4.8 ms per image** (32 eval images = 0.15 s)                                                                                                                                                             |

FFmpeg/`avifenc` subprocesses, `torch.mps._host_alias_storage`, `tile_size != 16` and
`rasterize_mode="antialiased"` are deliberately **not** used: the first two do not exist here, the last
two silently drop rendering onto the torch fallback.

### What is verified to work

| command / profile                                                                                                                                                                           | cpu                                                                                                                                                                         | mps                        |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| `convert --output-format SOG-web`                                                                                                                                                           | works                                                                                                                                                                       | works                      |
| `--output-format SOG-web-png`                                                                                                                                                               | works                                                                                                                                                                       | works                      |
| `--output-format SOG-web-nosh`                                                                                                                                                              | works                                                                                                                                                                       | works                      |
| `--output-format SOG-web-sh-split`                                                                                                                                                          | works                                                                                                                                                                       | works                      |
| `--output-format SOG-PlayCanvas`                                                                                                                                                            | works (scenes >= 65536 primitives)                                                                                                                                          | works (same)               |
| `--output-format 3DGS-INRIA-ply` / `-nosh-ply`                                                                                                                                              | works                                                                                                                                                                       | works                      |
| decode a SOG container -> PLY (`--input-format smurfx`, the container token; the `SOG-*` names are _profiles_, not input formats - `SOG-web` raises `ValueError: Unsupported input format`) | works                                                                                                                                                                       | works                      |
| round-trip `PLY -> SOG-web -> PLY`, 2.54M primitives                                                                                                                                        | works, all budgets                                                                                                                                                          | works, all budgets         |
| `ffsplat view` / `ffsplat live` (browser, `localhost:8080`)                                                                                                                                 | n/a                                                                                                                                                                         | works                      |
| `ffsplat eval` (real Tanks&Temples COLMAP set, 32 images)                                                                                                                                   | n/a                                                                                                                                                                         | works, SSIM + LPIPS on MPS |
| `pytest --doctest-modules`                                                                                                                                                                  | 23 passed                                                                                                                                                                   | 23 passed                  |
| `mypy`, `deptry`, `ruff check`, `ruff format --check`                                                                                                                                       | no new findings (mypy/deptry clean; the 5 ruff check findings and 3 `ruff format --check` complaints are pre-existing upstream lines, verified unchanged against `3fb6444`) | same                       |

### Parity vs CPU

Every output profile was encoded on both devices from the same real fixtures (202² = 40,804 and
512² = 262,144 primitives, subsampled from the real INRIA truck model), decoded back, and compared
**order-independently** (PLAS is called without a seed upstream, so two runs of the same command differ;
1001-quantile profiles are used, never element-wise). Budgets come from each template's own declared
quantiser. `f_dc_`, `rot_`, `scale_`, `opacity`, `x/y/z` and the `f_rest` codebook all pass, and the
mean/median error statistics agree between devices:

All rows below are the **512² (262,144-primitive) run**; its verbatim report is on the
`apple-silicon` branch (A1.5, `matrix_report.py mid512`), and delta and budget come from the same run.

| profile             | container bytes cpu / mps | cpu-vs-mps worst property delta                                                                              | budget   | verdict                                                                                                                                                                                                                                         |
| ------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------ | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SOG-web             | 4,508,343 / 4,510,184     | `f_rest` 0.036184 (mean/range; everything else 0.000000)                                                     | 0.05     | PASS                                                                                                                                                                                                                                            |
| SOG-web-png         | 4,845,133 / 4,847,738     | `f_rest` 0.036184 (mean/range; everything else 0.000000)                                                     | 0.05     | PASS                                                                                                                                                                                                                                            |
| SOG-web-nosh        | 2,912,255 / 2,911,156     | `f_dc` 0.085588                                                                                              | 0.120918 | PASS                                                                                                                                                                                                                                            |
| SOG-web-sh-split    | 5,877,576 / 5,878,134     | `f_dc` 0.213970 (0.042794-0.171176 for the same measurement between four CPU runs; everything else 0.000000) | 0.120918 | PASS by control: the encoder's quantised `f_dc` image is multiset-identical across devices (max abs sorted delta 0); the decoded difference comes only from this profile's lossy `chroma: 420` 3-pixel-wide AVIF — see A1.5a on `apple-silicon` |
| SOG-PlayCanvas      | 5,246,358 / 5,187,036     | `rot` 0.000015                                                                                               | 0.011764 | PASS                                                                                                                                                                                                                                            |
| 3DGS-INRIA-ply      | 61,869,378 / 61,869,378   | 0.000000                                                                                                     | —        | PASS (bit-identical)                                                                                                                                                                                                                            |
| 3DGS-INRIA-nosh-ply | 14,681,969 / 14,681,969   | 0.000000                                                                                                     | —        | PASS (bit-identical)                                                                                                                                                                                                                            |

At 202² the same six non-PlayCanvas profiles pass as well, with the same caveat as the 512² sh-split
row: `SOG-web-nosh`'s and `SOG-web-sh-split`'s `f_dc` sit within the codec's own run-to-run noise there
too (the audit measured 0.099582 and 0.099844 against 0.095100 / 0.094724 budgets, i.e. the harness row
reads FAIL on a single run and passes once a second same-device run supplies the noise floor).
`SOG-PlayCanvas` needs the full 65536-entry codebook texture and therefore fails identically on both
devices below 65536 primitives (see _Known gaps_). The one `>budget` cpu-vs-mps number in the whole matrix (`SOG-web-sh-split`'s `f_dc`) is the only
place where the devices are not numerically identical, and it was resolved by a direct control rather
than a threshold: the encoder hands the AVIF writer a quantised `f_dc` image whose **value multiset is
identical on both devices** (max abs sorted delta 0, all three channels — only the arrangement differs,
because PLAS is unseeded). That profile writes `f_dc` as a 3-pixel-wide AVIF with `chroma: 420`, and the
lossy 4:2:0 reconstruction is arrangement-dependent, so the decoded error varies with PLAS's shuffle:
0.0428-0.1712 between four CPU runs of the same command, 0.1284-0.2996 between those runs and MPS. The
device computes the same values; the codec does not. Container sizes vary by a few 0.1% between runs for the same
reason (PLAS is unseeded upstream), which is why bytes are quoted per run and value comparisons are
order-independent.

The MPS render itself was also checked against an **independent** implementation — a from-the-paper
pure-PyTorch CPU EWA + depth-sorted alpha-compositing renderer written for this port — on 300 real
truck Gaussians at 244x136: MAE 4.2e-05 over the image (PSNR 59.96 dB, correlation 0.999935), with
mpsplat on CPU and mpsplat on MPS matching the reference to the same 6 decimal places.

### Measured timings (M1 Max)

Rasterising the real pre-trained INRIA **truck** model (2,541,226 Gaussians, `sh_degree=3`) at
978x545 with `radius_clip=3`, `render_mode="RGB"`. **Every number in this table is the
`rasterization()` call alone, with the Gaussian, camera and SH tensors already resident on the
device** - no host-to-device copy and no preprocessing is inside the measured window (that is what
`cli/view.py`'s `render_fn` does per frame once a scene is loaded):

| path                                                                                     | cold (Metal JIT / first call) | warm                                                             |
| ---------------------------------------------------------------------------------------- | ----------------------------- | ---------------------------------------------------------------- |
| **Metal kernels** (what `view`, `live` and `eval` use)                                   | **0.09 s**                    | **0.033 s** (median of 5; 0.045 s in a second, equally idle run) |
| pure-torch fallback (only reached if `backgrounds=`/`masks=` is passed or SH degree > 4) | 5.13 s                        | 1.69 s                                                           |

Re-measured through the shipped dependency (mpsplat installed from the pinned revision rather than a
vendored copy), same scene, same size, same call, tensors resident: cold (first call ever, Metal JIT)
**0.136 s**, warm median-of-5 **0.066 s** (min 0.061 s, max 0.086 s). The gap to the
audited 0.09 s / 0.033 s pair is session state, not a code difference: the audited pair was taken on a
fully idle machine as the first GPU work of the process, this one immediately after a 630 MB PLY parse
and a CPU reference render on the same GPU.

Rendering the **40,804-primitive** tiny scene (the same truck subsampled to 202², decoded from this
branch's own SOG-web container) through `cli/view.py`'s `render_fn` on MPS at 978x545: warm
**5.3 ms** (median of 5). That is a _different_ quantity from the table above - `render_fn` end to
end, including its numpy output conversion, on a scene 62x smaller.

> Historical note: the numbers recorded in earlier revisions of this file (5.73 s cold / 1.72 s warm)
> were measured through the _fallback_ path, because the gate script that produced them passed
> `backgrounds=`. They are kept here as the fallback's timing, and relabelled.

PLAS spatial sorting. The template's real input is a 6-channel grid (means -> signed-log -> 16-bit
quantise -> split bytes -> standardize), and PLAS's runtime is data-dependent, so this was re-measured
on the real tensor captured from the encoder:

| grid (real 6-channel input)                        | cells     | `--device cpu`        | `--device mps`            |
| -------------------------------------------------- | --------- | --------------------- | ------------------------- |
| 512²                                               | 262,144   | 55.7 s (median of 3)  | **30.4 s** (median of 3)  |
| 1594² (the real scene)                             | 2,540,836 | 621.6 s (median of 2) | **278.7 s** (median of 2) |
| _baseline: 2-channel random grid, pre-measurement_ | 262,144   | _21.6 s_              | _64.3 s_                  |

The baseline row is kept only to show why it was misleading: a 2-channel random grid is not what the
SOG templates feed PLAS, and it inverted the sign of the result.

k-means codebook (the SOG-web input from a 512² real-truck scene, N=262,144, K=65,536, D=45, ten Lloyd
iterations plus the full-data assignment):

|                | before  | after      |
| -------------- | ------- | ---------- |
| `--device cpu` | 105.8 s | **66.4 s** |
| `--device mps` | 68.2 s  | **22.3 s** |

Full-size truck round-trip, 2,541,226 -> 2,540,836 Gaussians (the SOG-web template prunes to
`floor(sqrt(N))²` for its square grid); 630,225,580 B input PLY -> ~33 MB SOG container (~19x smaller):

| device and code state                                                    | encode (PLY -> SOG-web)              | decode (SOG-web -> PLY) |
| ------------------------------------------------------------------------ | ------------------------------------ | ----------------------- |
| `--device mps`, **pre-optimisation baseline** (freshly cloned, idle)     | 425.8 s                              | 6.1 s                   |
| `--device cpu`, **pre-optimisation baseline**                            | 863.6 s                              | 6.3 s                   |
| `--device mps`, **after the k-means change** (this branch's state), idle | **324.8 s** - container 32,973,468 B | 5.8 s                   |
| `--device cpu`, **after the k-means change** (this branch's state), idle | **683.65 s** - container 32995454 B  | 6.3 s                   |

Like-for-like (both after the k-means change), MPS is **~2.1x** faster than the CPU: 683.65 s versus
324.8 s. Against its own pre-optimisation baseline the MPS run is **1.31x** faster (425.8 -> 324.8 s),
and that gain is the k-means change - which also made the _CPU_ path 1.59x faster, which is why the old
863.6 s CPU number must not be paired against the new MPS number (that pairing would overstate the
ratio as 2.66x).

Where the 339.4 s of **instrumented** encode time goes (same workload as the 324.8 s CLI run; the
instrumentation adds one device synchronisation per transformation, which is the 14.6 s difference):

| stage                                       | seconds   | share | GPU utilisation (full-size run) |
| ------------------------------------------- | --------- | ----- | ------------------------------- |
| PLAS spatial sort                           | 291.5     | 85.9% | 94.9%                           |
| k-means fit + full-data assignment          | 42.9      | 12.6% | 99.8%                           |
| image encode (9 files, AVIF)                | 4.8       | 1.4%  | 20.1% (CPU-bound stage)         |
| quantise / byte-split / reindex / reshaping | 0.13      | 0.04% | -                               |
| **total, instrumented**                     | **339.4** |       | 94.4% mean for the run          |

(The utilisation column is the independent audit's full-size run, which reproduced this table within
0.5%; my own instrumented run recorded 94.2% for the whole encode. The 512² per-stage utilisation
numbers - k-means 99.7%, PLAS 84.6%, image encode 26.7% - are in B3 of the `apple-silicon` record and
are for that scene only.)

The container the full-size run produced was decoded again and compared against the input
order-independently: **all 8 property groups inside their codec-derived budgets**, with `x/y/z`,
`scale_`, `opacity` and `f_rest` reproducing the original Stage-3 record to the digit — the k-means
change did not move the full-size codec.

`ffsplat eval` on that container against the real Tanks&Temples/truck COLMAP set (32 test images):

```
[Parser] 251 images, taken by 251 cameras.
PSNR: 24.295, SSIM: 0.8461, LPIPS: 0.184 Time: 0.045s/image Number of GS: 2540836 Size: 31.446MB
```

Identical metrics to the pre-optimisation record (24.295 / 0.8461 / 0.184). The white-background path
(used by blender-style datasets; COLMAP datasets use no background) was moved off mpsplat's torch
rasteriser in this branch: the same `rasterization()` call went from 2.018 s (torch fallback, warm)
to 0.045 s (Metal, warm) at 978x545 - **44.8x**, on the definitions the timing table above uses - with
the host-side composite verified equal to the rasteriser's own blend to 1.55e-06.

One-time costs: a re-run of `uv sync` on an already up-to-date environment takes ~2.5 s (a _fresh_
install is the ~30 s / ~90 s above); the first `ffsplat view`/`live` launch builds viser's browser
client (~45 s, once).

### Where MPS is not optimized

1. **PLAS is 86% of the full-size encode (291.5 s of 339.4 s).** It is already on the faster device
   (2.2x faster than the CPU at the real grid). The cost is upstream `plas`: its inner loop calls
   `l2_dist(...)` which ends in `.item()` (`plas/core.py:91-92`) and branches on that Python scalar, so
   **5398 of the encode's 5469 host syncs are PLAS's**, each one followed by a burst of tiny kernels.
   Removing those syncs was tried, measured and **closed**: batching the early-exit test instead of
   running it per reorder removes 0.786 syncs per reorder (1.40 ms) but samples the test, which costs
   +476 reorders (+6.8%) at 38.5 ms each - a net **+8.5 s, i.e. 3.1% slower** at the real 1594² grid
   (288.07 s against 279.29 s, median of 3 each, tight runs); at 512² the two effects cancel. It is
   therefore not in this branch, and `plas` is the pinned upstream revision unchanged. The measured
   next lever is `torch.compile` on the candidate-distance helper alone (5.0x on that op, ~17% of the
   encode), not adopted because it changes the summation order and therefore possibly the sort's
   output.
2. **The `manhattan` k-means used by `SOG-PlayCanvas` is the only stage where the CPU beats the GPU:**
   48.0 s vs 38.2 s per assignment pass at the real shape (after the branch's own change; before it,
   56.8 s vs 118.2 s). L1 has no matmul identity, so the score matrix must be materialised by
   subtraction and MPS reaches roughly a quarter of its memory bandwidth doing it; a fused L1+argmin
   Metal kernel would be needed. The stage still runs on the GPU (device policy), and
   `SOG-PlayCanvas` is correspondingly the only profile that is slower on MPS than on CPU — measured on
   the 512² real scene: **604.7 s on MPS vs 560.3 s on CPU** (before this branch's change the MPS
   encode of that profile took 731.5 s, and the CPU one is ~3.1x faster than it was initially, so the
   gap went from ~48% to ~8%).
3. **Image encoding is CPU-bound by construction** (1.4% of the encode, GPU idle at 26.7% during it).
   There is no GPU AVIF/PNG encoder in this stack; only the container's `chroma`/`quality` parameters
   change its cost, and those change the output.
4. **The four "quantise / byte-split / reindex" style stages are ~0.04% of the encode** — there is
   nothing left to optimise there; they are already native MPS torch ops.

### Known gaps on MPS

- `torch`'s MPS backend cannot represent `uint16`/`uint32`. The SOG encoders use those widths only as
  intermediate carriers while splitting a value into bytes, so they are substituted with `int32`/`int64`
  (`ffsplat.device.encoding_dtype`) — byte-identical output, but it is a backend limitation worth knowing.
- **`SOG-PlayCanvas` needs a scene with at least 65536 primitives.** Its SH codebook is packed into a
  fixed 1024x960 RGB texture (64 clusters per row), and the k-means clamps its cluster count to the
  number of samples, so a smaller scene cannot be packed into that layout. Both devices fail identically
  (verified on the 40,804-primitive scene); the profile now raises a named `ValueError` instead of a bare
  reshape error.
- **`SOG-web-nosh` containers cannot be re-encoded through `ffsplat convert`**: they decode to an `sh` of
  width 1 and every shipped PLY profile splits a 16-wide `sh`
  (`RuntimeError: split_with_sizes expects split_sizes to sum exactly to 1 ... but got split_sizes=[1, 15]`,
  identical on both devices). The A1 harness decodes such containers in-process instead.
- **`SOG-web-nosh` and `SOG-web-sh-split` write `f_dc` as a 3-pixel-wide AVIF with `chroma: 420`**, so
  4:2:0 chroma subsampling destroys it: the `f_dc` error versus the input is 0.9956 for both, against
  0.0417 for `SOG-web` and 0.0184 for `SOG-web-png` (which use `chroma: 444` / lossless). Device
  independent; changing it is an output/compatibility decision and is **not** done here.
- **`SOG-PlayCanvas` quaternions are lossy for ~9% of primitives**: `Reparametrize(pack_quaternions)`
  always drops component 0 and records index 0 (`transformations.py:415-417`), so the decoder's
  `w = sqrt(1-|v|²)` is only correct when `|w|` is the largest component — true for 91.3% of this
  scene's quaternions. Device independent (both devices agree to 1.5e-05).
- **Every render also runs exactly one pure-torch op**: `isect_offset_encode` has no Metal branch in
  mpsplat (`gsplat/cuda/_wrapper.py:953-972` always calls `_torch_impl`; there is no
  implementation under `gsplat/mps/`), so one `[tile_height, tile_width]` scan runs on the host per
  render - ~3.0% of the 33 ms warm render (audit measurement). `tests/test_metal_path.py` cannot catch
  it by design: there is no Metal alternative to fall back from.
- **Spherical harmonics of degree > 4** are evaluated on mpsplat's pure-torch path, because its Metal SH
  kernel only handles `K <= 25`. `ffsplat.device.warn_if_not_metal` now announces this; ffsplat's PLY
  format is degree 3 in practice. mpsplat's other gaps (2DGS, UT/ftheta cameras, packed, antialiased
  mode, lidar, distributed) are features ffsplat's forward-only renderer never uses.
- `ffsplat live`'s **Convert** button was exercised through the GUI and completes on MPS, but only on a
  small scene: on the full truck model with the GUI's default `SOG-PlayCanvas` profile it was still
  encoding (Metal GPU at 94-100%) when a ~30 min best-effort budget expired, so it was stopped. The
  `live preview` checkbox and the live viewer's Evaluation tab were not driven through the GUI; the calls
  behind them are the same `convert`/`eval` paths verified from the CLI.
- An end-to-end `ffsplat eval` with `white_background=True` was **not** run: that path needs a
  blender-style dataset, which is not among the fixtures. The render-level equivalence and the
  Metal-path test cover the change.
- viser's initial camera is not set by ffsplat, so a freshly opened viewer is framed arbitrarily — that is
  upstream behaviour, not MPS-specific.

## 3DGS Container Format

Find the documentation for the encoding and decoding description, and the yaml container format in [CONTAINER_FORMAT.md](CONTAINER_FORMAT.md).
