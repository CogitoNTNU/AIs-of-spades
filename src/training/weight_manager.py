import random
import torch
from pathlib import Path
from typing import Optional

from nn.poker_net import PokerNet


class WeightManager:
    def __init__(self, config: dict):
        self.model_class = config["model_class"]
        self.models_dir = Path(config["models_dir"])
        self.max_models = config.get("max_models", 50)
        self.keep_latest = config.get("keep_latest", 20)
        self.sampling_mode = config.get("sampling_mode", "uniform")
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))

        self.snapshots: list[dict] = []
        self.cache: dict[str, PokerNet] = {}
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @property
    def _model_dir(self) -> Path:
        return self.checkpoint_dir / self.model_class.__name__

    def _sample_snapshot(self) -> dict:
        """Return one snapshot according to the configured sampling mode."""
        snapshots = self.snapshots
        if self.sampling_mode == "uniform":
            return random.choice(snapshots)
        elif self.sampling_mode == "linear":
            weights = list(range(1, len(snapshots) + 1))
        elif self.sampling_mode == "exponential":
            weights = [2**i for i in range(len(snapshots))]
        else:
            raise ValueError(f"Unknown sampling mode: {self.sampling_mode!r}")
        return random.choices(snapshots, weights=weights, k=1)[0]

    def _trim_pool(self) -> None:
        """Remove one random old checkpoint when the pool exceeds max_models."""
        if len(self.snapshots) <= self.max_models:
            return

        older = self.snapshots[: -self.keep_latest]
        if not older:
            return

        victim = random.choice(older)
        Path(victim["path"]).unlink(missing_ok=True)
        self.snapshots.remove(victim)
        self.cache.pop(victim["path"], None)
        print(f"Pruned old checkpoint: {victim['path']}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, model: PokerNet, optimizer, epoch: int) -> None:
        path = self._model_dir / f"epoch_{epoch}.pt"
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
            {"epoch": epoch, "path": str(path), "state_dict": state_dict}
        )
        self._trim_pool()

    def load(self, filename: str) -> PokerNet:
        if filename in self.cache:
            return self.cache[filename]

        checkpoint = torch.load(filename)
        model = self.model_class()
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))

        self.cache[filename] = model
        return model

    def sample_opponent(self) -> PokerNet:
        if not self.snapshots:
            return self.model_class()
        return self.load(self._sample_snapshot()["path"])

    def sample_opponent_state_dict(self) -> Optional[dict]:
        """Return a raw, picklable state_dict; None if no snapshots exist."""
        if not self.snapshots:
            return None
        return self._sample_snapshot()["state_dict"]["model_state_dict"]

    def get_current_model(self) -> PokerNet:
        """Return the most recently saved model, or a fresh one if no snapshots exist."""
        if not self.snapshots:
            return self.model_class()
        return self.load(self.snapshots[-1]["path"])
