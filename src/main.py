import os
import argparse
from pathlib import Path
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


def find_latest_checkpoint(config: dict) -> str | None:
    model_class = config["weight_manager"]["model_class"]
    model_class_name = (
        model_class.__name__ if isinstance(model_class, type) else model_class
    )
    checkpoint_dir = Path(config["weight_manager"].get("checkpoint_dir", "checkpoints"))
    model_dir = checkpoint_dir / model_class_name
    checkpoints = sorted(
        model_dir.glob("epoch_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


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
        help="Path to a specific checkpoint file to resume from",
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        default=False,
        help="Resume from the latest checkpoint in the checkpoint directory",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name: sets wandb run name and checkpoint_dir to res/checkpoints/<run-name>",
    )
    args = parser.parse_args()

    with open(os.path.join(os.path.dirname(__file__), args.config_file), "r") as f:
        config = yaml.safe_load(f)

    if args.num_workers is not None:
        config["learning_loop"]["num_workers"] = args.num_workers

    if args.run_name is not None:
        config["weight_manager"]["checkpoint_dir"] = f"res/checkpoints/{args.run_name}"

    config["weight_manager"]["model_class"] = MODEL_CLASSES[
        config["weight_manager"]["model_class"]
    ]
    model_name = config["weight_manager"]["model_class"].__name__

    resume_from = args.resume_from
    if args.resume_latest:
        if resume_from is not None:
            print(
                "[main] WARNING: --resume-latest and --resume-from both set; using --resume-from"
            )
        else:
            resume_from = find_latest_checkpoint(config)
            if resume_from:
                print(f"[main] --resume-latest: resuming from {resume_from}")
            else:
                print(
                    "[main] --resume-latest: no checkpoints found, starting from scratch"
                )

    wandb.init(
        project="AIs-Of-Spades",
        entity="pokerai",
        name=args.run_name,
        config=config,
    )

    weight_manager = WeightManager(config=config["weight_manager"])
    print_model_summary(weight_manager.get_current_model(), model_name)

    learning_loop = LearningLoop(weight_manager, config)
    learning_loop.start_learning(resume_from=resume_from)
