import os
import argparse
from training.learning_loop import LearningLoop
from training.weight_manager import WeightManager
from training.wandb_compat import wandb
from nn import MODEL_CLASSES
import yaml


def print_model_summary(model, model_name: str) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║                   MODEL  SUMMARY                    ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Architecture   : {model_name:<33}║")
    print(f"║  Total params   : {total_params:>12,}                     ║")
    print(f"║  Trainable      : {trainable_params:>12,}                     ║")
    print(
        f"║  Non-trainable  : {total_params - trainable_params:>12,}                     ║"
    )
    print("╠══════════════════════════════════════════════════════╣")
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        print(f"║  {name:<20} {n:>10,} params               ║")
    print("╚══════════════════════════════════════════════════════╝\n")


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
    parser.add_argument(
        "--config-file",
        type=str,
        default="config.yaml",
        help="Configuration file",
    )
    args = parser.parse_args()

    with open(os.path.join(os.path.dirname(__file__), args.config_file), "r") as f:
        config = yaml.safe_load(f)

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
    model_name = config["weight_manager"]["model_class"].__name__

    weight_manager = WeightManager(config=config["weight_manager"])
    print_model_summary(weight_manager.get_current_model(), model_name)

    learning_loop = LearningLoop(weight_manager, config)
    learning_loop.start_learning(resume_from=args.resume_from)
