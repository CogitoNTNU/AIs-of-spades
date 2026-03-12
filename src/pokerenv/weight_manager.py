import os
import torch
import glob
import random


class WeightManager:
    def __init__(self, config):
        self.model_class = config.get("model_class")
        self.models_dir = config.get("models_dir")
        self.max_models = config.get("max_models", 50)
        self.keep_latest = config.get("keep_latest", 20)
        self.sampling_mode = config.get("sampling_mode", "uniform")
        self.checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
        self.snapshots = []
        self.cache = {}

        os.makedirs(self.models_dir, exist_ok=True)

    def save(self, model, optimizer, epoch: int):  # aggiungi optimizer
        path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}.pt")
        state_dict = {
            "model_state_dict": {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(state_dict, path)
        print(f"Saved checkpoint: {path}")

        self.snapshots.append(
            {
                "epoch": epoch,
                "path": path,
                "state_dict": state_dict,
            }
        )

        self._trim_pool()

    def _trim_pool(self):

        if len(self.snapshots) <= self.max_models:
            return

        latest = self.snapshots[-self.keep_latest :]
        older = self.snapshots[: -self.keep_latest]

        if older:
            to_remove = random.choice(older)
            os.remove(to_remove["path"])
            self.snapshots.remove(to_remove)
            if to_remove["path"] in self.cache:
                del self.cache[to_remove["path"]]
            print(f"Pruned old checkpoint: {to_remove['path']}")

    def load(self, filename: str):
        if filename in self.cache:
            return self.cache[filename]

        model = self.model_class()
        checkpoint = torch.load(filename)

        # Extract the actual model weights
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)  # fallback if raw state_dict

        self.cache[filename] = model
        return model

    def sample_opponent(self):
        if not self.snapshots:
            # If no checkpoints are available, return a randomly initialized model
            return self.model_class()

        if self.sampling_mode == "uniform":
            chosen_file = random.choice(self.snapshots)["path"]
        elif self.sampling_mode == "linear":
            weights = list(range(1, len(self.snapshots) + 1))
            chosen_file = random.choices(
                [s["path"] for s in self.snapshots], weights=weights, k=1
            )[0]
        elif self.sampling_mode == "exponential":
            weights = [2**i for i in range(len(self.snapshots))]
            chosen_file = random.choices(
                [s["path"] for s in self.snapshots], weights=weights, k=1
            )[0]
        else:
            raise ValueError(f"Unknown sampling mode: {self.sampling_mode}")

        return self.load(chosen_file)
