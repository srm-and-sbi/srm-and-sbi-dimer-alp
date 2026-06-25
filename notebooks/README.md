# notebooks/

Exploratory Jupyter notebooks for inspecting pipeline outputs interactively.
Not part of the production pipeline.

- `Video_Scrubber.ipynb` — interactive frame-by-frame viewer for DLI video sets
  (`ipywidgets` sliders over simulation index and frame). The first cell explains
  how to run it against data on a remote GPU server via an SSH port-forward,
  or locally.

Committed notebooks should be **output-cleared** (no embedded images) to keep the
repository diff-friendly; regenerate the interactive view live against whatever
data you point the notebook at. Requires `ipywidgets` (present in `SRM_AND_SBI_ENVY_V0`).
