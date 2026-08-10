"""Shared workflow identity for the DIMER pipeline's two mirrored workflows.

The pipeline runs **two mirrored workflows** over the same five stages (RDS, DLI,
Inference, Evaluation, Experiment):

  - **biology**  -- infers the reaction-diffusion parameters (three species
    counts + three diffusivities + four reaction rates) and marginalizes the
    imaging block. This is the unqualified entry point / ``project_alias``.
  - **detector** -- infers the six imaging parameters and marginalizes the
    reaction-diffusion domain and the camera (SCOPE) block. Its data namespaces
    separately under the ``_DETECTOR`` runtime-prefix qualifier.

Both workflows share ONE engine per stage (``run_<stage>`` in the stage-runner
modules). Everything that differs between them is carried by the small
``WorkflowConfig`` below, which each entry point builds and hands to the shared
runner. Keeping the difference in the config -- not in duplicated ``main()``
bodies -- is what keeps the two workflows mirrored by construction: a change to
a stage's engine lands in both, and neither can silently drift from the other.

The genuine per-workflow differences are only: which parameterization module
supplies the tables/bounds/keys, the alias-qualified ``paths``, and a workflow
``tag``. Each stage runner resolves its own stage-specific specializations
(e.g. the RDS reactive-vs-diffusion-only simulation builder) from this config in
one localized place; the shared engine body itself carries no workflow branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class WorkflowConfig:
    """The identity of one workflow, threaded through every shared stage runner.

    Attributes:
        tag: ``"biology"`` or ``"detector"`` -- the workflow selector a stage
            runner uses to resolve its stage-specific specializations, and the
            value stamped into estimator / diagnostic metadata.
        paths: the alias-qualified ``Paths`` object. For biology this is
            ``PARAMETERS.paths``; for detector it is
            ``detector_parameterization.detector_paths(PARAMETERS.paths)``, whose
            ``project_alias`` carries the ``_DETECTOR`` qualifier so detector data
            never collides with biology data.
        param_module: the parameterization module that owns this workflow's
            tables, prior bounds, parameter keys, ``build_prior`` and ``role_of``
            (``parameterization`` for biology, ``detector_parameterization`` for
            detector). Stage runners read the specific symbols they need through a
            small per-stage resolver, since the two modules name a few symbols
            differently by design.
        console_log_paths: the ``paths`` argument for ``console_log_context`` so
            the console log lands in this workflow's namespace. ``None`` for
            biology (the default canonical alias); the detector ``paths`` for
            detector.
    """

    tag: str
    paths: Any
    param_module: ModuleType
    console_log_paths: Any = None


def biology_workflow() -> WorkflowConfig:
    """Build the biology WorkflowConfig (infers reaction-diffusion; unqualified alias)."""
    from srm_and_sbi_dimer_alp import parameterization as biology_param
    return WorkflowConfig(
        tag="biology",
        paths=biology_param.PARAMETERS.paths,
        param_module=biology_param,
        console_log_paths=None,
    )


def detector_workflow() -> WorkflowConfig:
    """Build the detector WorkflowConfig (infers imaging; _DETECTOR-qualified alias)."""
    from srm_and_sbi_dimer_alp import parameterization as biology_param
    from srm_and_sbi_dimer_alp import detector_parameterization as detector_param
    detector_paths = detector_param.detector_paths(biology_param.PARAMETERS.paths)
    return WorkflowConfig(
        tag="detector",
        paths=detector_paths,
        param_module=detector_param,
        console_log_paths=detector_paths,
    )
