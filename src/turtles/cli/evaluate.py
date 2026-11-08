"""``turtles-eval``: full-mesh rollout evaluation (Hydra app; see ``configs/eval.yaml``)."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from turtles.cli._common import guarded
from turtles.experiment import CONFIG_DIR, run_evaluation


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="eval")
def main(cfg: DictConfig) -> None:
    """Hydra entry point."""
    guarded(run_evaluation, cfg)


if __name__ == "__main__":
    main()
