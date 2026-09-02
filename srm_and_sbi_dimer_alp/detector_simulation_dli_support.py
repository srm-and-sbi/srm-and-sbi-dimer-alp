"""Detector DLI forward model: the shared renderer, under the Detector-facing name.

Part of the Detector calibration workflow (see the implementation plan in
DETECTOR_WORKFLOW.md). The diffraction-limited-imaging renderer is source-agnostic — it
reads each of the eleven imaging parameters by key, so a value is rendered the same whether
it arrived as an inference target or a marginalized nuisance — so both DLI stages share one
renderer. That renderer lives in the canonical ``simulation_dli_support`` module as
``render_dli_video``; this module re-exports it under the Detector-facing name
``render_detector_video`` for the Detector DLI stage and the Detector posterior-predictive
analysis, which call it with the six imaging parameters drawn as the learnable inference
target (Theta_Set) and the five SCOPE camera parameters drawn as the marginalized camera
nuisance (Nuisance_SCOPE). The renderer sources its fixed hyperparameters
(``numb_photo_bleach``, ``dimer_mule``) from the canonical parameter table; those values
equal the Detector table's, so the Detector's rendered output is unchanged.
"""

from .simulation_dli_support import render_dli_video as render_detector_video

__all__ = ["render_detector_video"]
