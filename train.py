"""
Entry point for training. Mirrors eval.py's root-level layout; delegates to
train/muzero.py's run() for the actual actor-learner setup and training loop.

Usage
-----
  python train.py                                  # default config
  python train.py model=smax mcts=joint             # SMAX 3m (primary target)
  python train.py train.num_episodes=50000         # override single value
  python train.py train.batch_size=128 mcts.num_simulations=50
"""

import hydra
from omegaconf import DictConfig

from train.muzero import run


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
