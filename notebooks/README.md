# notebooks/

Interactive Jupyter notebooks for inspecting pipeline outputs by eye. These are viewers, not
part of the production pipeline: they render an already-produced artifact and never run a
simulation or an inference.

- **`SRM_AND_SBI_DIMER_ALP_DETECTOR_Posterior_Predictive_Video.ipynb`** — the posterior-predictive
  check. Views one persisted clip (`*_Synthetic_Video.npz`, written by
  `Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_DETECTOR_Posterior_Predictive_Video.py`) as an
  experimental-vs-synthetic pair: a scrubber (frame / center / zoom sliders) and a real-time
  player. A pure viewer — it needs only `numpy`, `matplotlib`, and `ipywidgets`; no project
  package and no `MACHINE_PROFILE`.
- **`Video_Scrubber.ipynb`** — frame-by-frame viewer and player for DLI video **sets**. Loads a
  generated video set through the project package, so it needs the package importable and a
  `MACHINE_PROFILE` pointing at the machine that holds the data.

Both render images **losslessly** (the scrubber draws the raw arrays; the player embeds PNG
frames). Do not route these images through a lossy codec.

---

## Prerequisites — exactly two

**1. The correct environment: `SRM_AND_SBI_ENVY_V0`.** These notebooks are supported on this
environment only. It carries both the project package and the notebook tooling: `jupyterlab`,
`ipywidgets`, `ipykernel`. If a machine's `SRM_AND_SBI_ENVY_V0` does not yet have the tooling,
add it once (additive; it does not touch the scientific stack):

```bash
<envy_v0_prefix>/bin/pip install jupyterlab ipywidgets ipykernel
```

Do **not** run the notebooks from any other environment (a base or legacy environment). Those
carry an older `ipywidgets` whose JupyterLab widget frontend is incompatible and fails with
`Failed to load model class 'VBoxModel' from module '@jupyter-widgets/controls'`. The matched
pairing is `jupyterlab` 4 with `ipywidgets` 8, which `SRM_AND_SBI_ENVY_V0` provides.

**2. The data.**
- Posterior-Predictive-Video: a rendered clip `*_Synthetic_Video.npz` (run the engine script to
  produce it). The clip is self-contained, so any absolute path works.
- Video_Scrubber: a generated video set under the active machine profile's `data_bank_root`.

Nothing else is required.

---

## Running — the reliable recipe

The one rule that prevents every recurring failure: **the JupyterLab *server* must be launched
from `SRM_AND_SBI_ENVY_V0`, addressed by that environment's own `jupyter-lab`.** The widget
frontend is served by the *server* environment, not by the notebook's kernel — so pointing the
kernel at `SRM_AND_SBI_ENVY_V0` is not enough if the server is a different environment.

**Step 1 — find the environment prefix.**

```bash
conda env list            # note the path next to SRM_AND_SBI_ENVY_V0
```

Call the launcher below `<envy_v0_prefix>/bin/jupyter-lab`. Never invoke a bare `jupyter-lab`:
the shell `PATH` may resolve it to a different environment (whose widgets are incompatible), and
that mismatch is the usual cause of the `VBoxModel` error.

**Step 2 — check port hygiene, then launch.** If a server is already bound to the port, a new
launch does **not** take it over — you silently connect to the existing server (possibly a
different environment) instead. List and stop stragglers first:

```bash
<envy_v0_prefix>/bin/jupyter server list                 # what is already running, and where
<envy_v0_prefix>/bin/jupyter server stop 8888            # stop a straggler on that port, if any

cd <repo-root>
<envy_v0_prefix>/bin/jupyter-lab notebooks/SRM_AND_SBI_DIMER_ALP_DETECTOR_Posterior_Predictive_Video.ipynb
```

Open the URL the launch prints (a fresh tab — a tab left over from a previous server carries a
stale login cookie and returns `403 Forbidden` until reloaded with the new URL).

**Step 3 — kernel.** Select **Python 3 (ipykernel)** — launched this way it is
`SRM_AND_SBI_ENVY_V0`'s own kernel, which has `numpy`, `matplotlib`, and `ipywidgets`.

**Step 4 — set the notebook's inputs (below) and run all cells.**

**Alternative — VSCode (no server to manage).** Open the `.ipynb`, then select the
`SRM_AND_SBI_ENVY_V0` kernel (`<envy_v0_prefix>/bin/python`). VSCode renders `ipywidgets`
natively, so the server-versus-kernel distinction does not arise.

**Remote data (server on another machine).** Launch the server on that machine from *its*
`SRM_AND_SBI_ENVY_V0` with `--no-browser --port 8888`, forward the port
(`ssh -L 8888:localhost:8888 -p <ssh-port> <user>@<host>`), and open the printed URL locally.
`Video_Scrubber.ipynb`'s first cell shows this tunnel.

---

## Per-notebook inputs

**Posterior-Predictive-Video** (pure viewer, no `MACHINE_PROFILE`):
- Set `CLIP_PATH` to the absolute path of a `*_Synthetic_Video.npz`.
- Leave `NORM_MODE = "full"` (a shared full-range window over both panels; `autoscale` and
  `percentile` are alternatives).
- To compare two clips, load one, run all, then change `CLIP_PATH` and run all again.
- For a long clip, set `PLAY_EVERY = 2` in the player cell to keep the embedded player light; it
  stays real-time.

**Video_Scrubber** (needs the package + a profile):
- Set `MACHINE_PROFILE` (in the shell before launching, or in the first code cell) to the machine
  holding the data.
- Set `TASK`, `TOTAL_TIME_SECONDS` (the run duration that produced the set, e.g. `2.0`), and
  `SPLIT` (`TRAIN` / `TEST` / `EVAL`).

---

## Troubleshooting

- **`Failed to load model class 'VBoxModel' …`** — the *server* is not `SRM_AND_SBI_ENVY_V0`; an
  older `ipywidgets` is serving the page. Confirm with `jupyter server list` which environment's
  server holds the port, `jupyter server stop <port>` it, relaunch from
  `<envy_v0_prefix>/bin/jupyter-lab`, and open the new URL in a fresh tab.
- **`403 Forbidden` after switching servers** — a browser tab from the previous server. Open the
  new URL (with its token) in a fresh tab.
- **Sliders do not appear, but there is no error** — the real-time **player** cell needs no
  widgets (it is plain HTML/JS), so it still works; or open the notebook in VSCode with the
  `SRM_AND_SBI_ENVY_V0` kernel.

---

## Committing

Keep committed notebooks **output-cleared** (no embedded images) so the repository stays
diff-friendly; regenerate the interactive view live against whatever data you point the notebook
at.
