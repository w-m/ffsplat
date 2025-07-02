# ffsplat

[![Release](https://img.shields.io/github/v/release/w-m/ffsplat)](https://img.shields.io/github/v/release/w-m/ffsplat)
[![Build status](https://img.shields.io/github/actions/workflow/status/w-m/ffsplat/main.yml?branch=main)](https://github.com/w-m/ffsplat/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/w-m/ffsplat/branch/main/graph/badge.svg)](https://codecov.io/gh/w-m/ffsplat)
[![Commit activity](https://img.shields.io/github/commit-activity/m/w-m/ffsplat)](https://img.shields.io/github/commit-activity/m/w-m/ffsplat)
[![License](https://img.shields.io/github/license/w-m/ffsplat)](https://img.shields.io/github/license/w-m/ffsplat)

ffsplat is a powerful framework to convert and compress 3D Gaussian Splatting scenes.

**NOTE: this code is pre-alpha, under heavy development, a community developer preview. Please expect more documentation in the coming weeks (July 2025).**

- **Github repository**: <https://github.com/w-m/ffsplat/>

## Quick Start

- Install CUDA 12.x
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

Run

`uvx --from git+https://github.com/w-m/ffsplat@main ffsplat-live --help`

Then open a web browser to [localhost:8080](http://localhost:8080) (or the address that is printed in the viser box on stdout).

## Setting up a dev environment

Check out the project with submodules:

```
git clone --recurse-submodules https://github.com/w-m/ffsplat.git
cd ffsplat
```

Set up the virtual environment with uv:

```bash
make install
```

Run the live encoding:

```bash
uv run -m ffsplat.cli.live \
    --input /data/gaussian_splatting/mini-splatting2/truck_sparse/point_cloud/iteration_18000/point_cloud.ply \
    --input-format=3DGS-INRIA.ply \
    --dataset-path /data/gaussian_splatting/tandt_db/tandt/truck/ \
    --verbose
```

Check the types and formatting of the code base:

```bash
make check
```

### 3DGS Container Format

Find the documentation for the encoding and decoding description, and the yaml container format in [CONTAINER_FORMAT.md](CONTAINER_FORMAT.md).
