import os
import argparse
import wandb
from training.learning_loop import LearningLoop
from training.weight_manager import WeightManager
from nn import MODEL_CLASSES
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override num_workers from config (used by SLURM via $SLURM_CPUS_PER_TASK)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    # SLURM override: se passato da CLI, ha priorità sul yaml
    if args.num_workers is not None:
        config["learning_loop"]["num_workers"] = args.num_workers

    wandb.init(
        project="AIs-Of-Spades",
        entity="pokerai",
        config=config,
    )

    config["weight_manager"]["model_class"] = MODEL_CLASSES[
        config["weight_manager"]["model_class"]
    ]

    weight_manager = WeightManager(config=config["weight_manager"])
    learning_loop = LearningLoop(weight_manager, config)
    learning_loop.start_learning(resume_from=args.resume_from)
