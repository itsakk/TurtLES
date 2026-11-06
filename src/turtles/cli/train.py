"""``turtles-train``: train a model (Hydra app; see ``configs/train.yaml``)."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from turtles.cli._common import guarded
from turtles.experiment import CONFIG_DIR, run_training


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="train")
def main(cfg: DictConfig) -> float:
    """Hydra entry point."""
    return guarded(run_training, cfg)


if __name__ == "__main__":
    main()
