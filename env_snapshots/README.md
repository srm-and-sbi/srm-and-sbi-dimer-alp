# Environments & install guide — `srm-and-sbi-dimer-alp`

How to set up the Python environment for this project (a ReaDDy reaction-diffusion +
SBI inference pipeline).

> **Canonical environment:** `SRM_AND_SBI_ENVY_V0` (instructions below). The repo's
> top-level `README.md` points here as the canonical install guide. A `READY_MARS`
> environment (Python 3.9) is available as an alternative/fallback and is documented at
> the end of this file.

> **Local env snapshots.** Dated `conda env export` / `conda list --explicit` captures of
> the built environments are kept locally for the author's own reference and audit; they are
> not shipped with the repo. The canonical, machine-independent path for everyone is the
> from-scratch recipe below.

---

## SRM_AND_SBI_ENVY_V0 — specification

Python **3.13**. The scientific stack is **identical on every machine**; only the
PyTorch build differs by hardware backend.

**Why this stack.** torch ≥ 2.9 compiles substantially faster on AMD (ROCm) GPUs and is a
drop-in upgrade requiring no code changes. ReaDDy caps Python at 3.13, and Python 3.14
breaks `torch.compile`, so 3.13 is the ceiling. zarr is pinned to the 2.x line because
zarr-3 breaks array writes and slows reads.

| component | version | note |
|---|---|---|
| python | 3.13.14 | readdy caps this at ≤ 3.13; Python 3.14 breaks `torch.compile` |
| readdy | 2.0.14 | conda-forge |
| numpy | 2.4.6 | |
| scipy | 1.17.1 | |
| scikit-image | 0.26.0 | |
| **zarr** | **2.18.7** | **pinned** — zarr-3 breaks array writes and slows reads |
| numcodecs, h5py, tifffile, pandas, matplotlib, seaborn, einops, tqdm, scikit-learn, statsmodels, imageio, networkx, pillow | conda-forge | conda layer |
| **psutil, ipython** | conda-forge | **required** — see gotchas |
| sbi | 0.26.1 | pip (with nflows 0.14, zuko 1.6.0) |
| **torch** | **machine-specific** | see backend table |

### Pick your PyTorch backend

| your hardware | torch build | pip `--index-url` |
|---|---|---|
| **AMD GPU** (ROCm) | `2.9.1+rocm6.4` (+ `pytorch-triton-rocm 3.5.1`) | `https://download.pytorch.org/whl/rocm6.4` |
| **NVIDIA GPU, recent driver** (CUDA ≥ 12.6) | `2.9.1+cu126` | `https://download.pytorch.org/whl/cu126` |
| **NVIDIA GPU, older driver** (CUDA 11.x) | `2.7.1+cu118` | `https://download.pytorch.org/whl/cu118` |
| **No GPU, or GPU too small** | `2.9.1` (CPU, from PyPI) | *(none — default PyPI)* |

Rule of thumb: **AMD → ROCm; NVIDIA → a `cuXXX` wheel whose CUDA version ≤ your driver's
max** (run `nvidia-smi`, read "CUDA Version" top-right — minor-version compat lets e.g.
`cu126` run on a 12.4 driver and `cu118` on an 11.4 driver); **no/small GPU → the CPU
build** and train on CPU. Match the torch *version* across machines that share artifacts
(both GPU machines here use `2.9.1`); the small-GPU PC uses `2.7.1` only because that is
the newest build with a `cu118` wheel its driver supports.

Worked examples by hardware class: an AMD MI210 (ROCm) HPC node → `rocm6.4`; an
RTX 6000 Ada workstation (driver 12.4) → `cu126`; a 4 GB laptop-class GPU (driver 11.4) →
`cu118`/CPU — **its 4 GB GPU OOMs the compiled inference (one conv3d needs ~6 GB), so it
trains on CPU.**

---

## Install from scratch (canonical method)

### Before you start
1. Have **conda** installed (Miniconda / Miniforge: <https://github.com/conda-forge/miniforge>).
   Optionally `conda install -n base -c conda-forge mamba` for a much faster solver — the
   `conda create` below pulls ~17 packages and can take several minutes on the classic solver.
2. Clone the repo and enter it (steps 1–2 below can run from anywhere, but step 3 assumes
   you are **inside the clone**):
   ```bash
   git clone https://github.com/srm-and-sbi/srm-and-sbi-dimer-alp.git
   cd srm-and-sbi-dimer-alp
   ```

### Steps
```bash
# 1. conda-forge scientific layer.  --override-channels is REQUIRED (see gotcha #2).
conda create -y -n SRM_AND_SBI_ENVY_V0 --override-channels -c conda-forge \
  python=3.13 readdy=2.0.14 numpy=2.4.6 scipy=1.17.1 scikit-image=0.26.0 zarr=2.18.7 \
  numcodecs h5py pandas matplotlib seaborn einops tifffile tqdm scikit-learn \
  statsmodels imageio networkx pillow pip psutil ipython
conda activate SRM_AND_SBI_ENVY_V0

# 2. pip layer: torch (PICK ONE LINE for your backend) + the sbi ecosystem, resolved together.
# AMD / ROCm:
pip install "torch==2.9.1+rocm6.4" sbi==0.26.1 nflows==0.14 zuko==1.6.0 \
  --index-url https://download.pytorch.org/whl/rocm6.4 --extra-index-url https://pypi.org/simple
# NVIDIA CUDA ≥ 12.6:
pip install "torch==2.9.1+cu126" sbi==0.26.1 nflows==0.14 zuko==1.6.0 \
  --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
# NVIDIA CUDA 11.x:
pip install "torch==2.7.1+cu118" sbi==0.26.1 nflows==0.14 zuko==1.6.0 \
  --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple
# CPU only (no GPU, or GPU too small):
pip install torch==2.9.1 sbi==0.26.1 nflows==0.14 zuko==1.6.0

# 3. the project package itself, editable, WITHOUT re-resolving deps (see gotcha #3).
#    Run from inside the cloned repo root:
pip install -e . --no-deps

# 4. sanity check
python -c "import readdy, torch, sbi, zarr, numpy, psutil, srm_and_sbi_dimer_alp; \
print('OK', torch.__version__, 'cuda/hip avail:', torch.cuda.is_available())"
```
(`cuda/hip avail` is `True` only when a GPU is actually visible — on an HPC node that means
inside a GPU Slurm allocation, not the login node; on the CPU path `False` is expected.)

### Critical gotchas (every one was hit during the build — do not skip)

1. **`psutil` and `ipython` must be installed explicitly.** They are not pulled
   transitively; without `psutil` the simulation stage dies immediately with
   `ModuleNotFoundError: No module named 'psutil'` (imported in `utils.py`).
2. **Use `--override-channels -c conda-forge`.** If the machine's `.condarc` lists
   `defaults` first with strict priority, a plain `conda create` silently resolves an
   older numpy-1.x / Python-3.9 resolution instead of the one specified here.
3. **Install the project with `--no-deps`.** `pyproject.toml` declares `sbi==0.26.1`
   as its only runtime dependency. Without `--no-deps`, a plain `pip install -e .`
   re-resolves that dependency tree — and because the call carries no PyTorch index, it
   **re-resolves torch from default PyPI, silently replacing your backend-specific build**
   (`rocm6.4`/`cu126`/`cu118`) with a generic wheel. `--no-deps` just links the package;
   the runtime deps are already satisfied by the conda + pip layers.
4. **Pick the torch backend for your hardware** (see the backend table for the build
   and its matching pip `--index-url`).
5. **Drive pip via the env's own interpreter, not a PATH-resolved `pip`.** If the shell
   auto-activates another conda env, `conda run -n <env>` and a bare `pip` can resolve to
   the *wrong* Python. Find the prefix with `conda env list` (or
   `conda activate SRM_AND_SBI_ENVY_V0 && which python`) and use it explicitly:
   `<prefix>/bin/python -m pip install ...`.
6. **HPC placement & disk** (worked example: a node whose `/work` scratch is not backed up).
   On such a node `/work` is **not backed up** (scratch/test envs only) — build on `/home`
   (backed up). If conda's `envs_dirs` defaults to `/work` there, create with an explicit
   prefix: `conda create -p /home/<user>/miniconda3/envs/SRM_AND_SBI_ENVY_V0 ...`.
   `/home` has a ~30 GB quota and the finished env is ~17 GB; keep the conda package cache on
   `/home` too so conda hardlinks (fast). **Watch the peak:** mid-build the env + package cache
   + the multi-GB torch wheel can transiently approach the quota *before* any cleanup, which can
   cause failed writes. Use `PIP_CACHE_DIR=/work` (or `--no-cache-dir`) for the big torch wheel,
   and run `conda clean --all -y` once the conda layer is in place — not only at the very end.

---

## Alternative environment: READY_MARS (Python 3.9)

A program-wide Python 3.9 environment, available as an alternative/fallback to the
canonical stack. Three captured mirrors are **not identical** (different torch builds and
some drift):

| role | python | torch | readdy | zarr | sbi |
|---|---|---|---|---|---|
| **workstation (CPU)** | 3.9.20 | conda `pytorch 2.5.1` CPU (`cpu_mkl`) | 2.0.13 | 2.18.2 | 0.23.2 |
| **AMD HPC node** | 3.9.19 | pip `torch 2.5.1+rocm6.2` (+ triton-rocm 3.1.0) | 2.0.12 | 2.18.2 | 0.23.2 |
| **CUDA workstation** | 3.9.20 | conda `pytorch 2.5.1+cuda12.4` **and** pip `torch 2.6.0` ⚠ | 2.0.12 | 2.18.2 | 0.23.3 |

Rebuild for the target role with the matching torch backend (same index / `--no-deps` /
channel-order caveats as above; the AMD build needs
`--extra-index-url https://download.pytorch.org/whl/rocm6.2`). The CUDA workstation's double
torch install is drift, not design — a clean rebuild should keep only the conda CUDA build.
