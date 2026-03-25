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

        # Snapshots store only lightweight metadata (epoch, path).
        # State dicts are never kept in memory — they are loaded from disk
        # on demand and cached with a bounded LRU-style eviction.
        self.snapshots: list[dict] = []

        # Bounded in-memory cache: at most keep_latest models loaded at once.
        # Keys are str(path), values are CPU state_dicts (not full PokerNet
        # instances, to avoid holding optimizer state or grad buffers).
        self._cache: dict[str, dict] = {}

        self._model_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @property
    def _model_dir(self) -> Path:
        return self.checkpoint_dir / self.model_class.__name__

    def _sample_snapshot(self) -> dict:
        """Return one snapshot metadata dict according to sampling_mode."""
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

    def _load_state_dict(self, path: str) -> dict:
        """
        Load a model state_dict from disk, with a bounded in-memory cache.

        The cache is capped at keep_latest entries.  When it is full the
        oldest entry (first inserted) is evicted before adding the new one.
        This prevents unbounded RAM growth that occurred when state_dicts were
        stored directly inside self.snapshots.
        """
        if path in self._cache:
            return self._cache[path]

        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Evict oldest cache entry if the cache is full.
        if len(self._cache) >= self.keep_latest:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[path] = state_dict
        return state_dict

    def _trim_pool(self) -> None:
        """Remove one random old checkpoint when the pool exceeds max_models."""
        if len(self.snapshots) <= self.max_models:
            return

        older = self.snapshots[: -self.keep_latest]
        if not older:
            return

        victim = random.choice(older)
        victim_path = victim["path"]

        Path(victim_path).unlink(missing_ok=True)
        self.snapshots.remove(victim)
        self._cache.pop(victim_path, None)  # evict from cache if present
        print(f"Pruned old checkpoint: {victim_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, model: PokerNet, optimizer, epoch: int) -> None:
        path = self._model_dir / f"epoch_{epoch}.pt"
        checkpoint = {
            "model_state_dict": {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

        # Only metadata is kept in the snapshot list — no state_dict in RAM.
        self.snapshots.append({"epoch": epoch, "path": str(path)})
        self._trim_pool()

    def load(self, filename: str) -> PokerNet:
        """Load a full PokerNet instance from a checkpoint path."""
        state_dict = self._load_state_dict(filename)
        model = self.model_class()
        model.initialize_internal_state()
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def sample_opponent(self) -> PokerNet:
        """Return a full PokerNet sampled from the snapshot pool."""
        if not self.snapshots:
            model = self.model_class()
            model.initialize_internal_state()
            return model
        return self.load(self._sample_snapshot()["path"])

    def sample_opponent_state_dict(self) -> Optional[dict]:
        """
        Return a picklable CPU state_dict for use in worker processes.

        Returns None if no snapshots exist yet (e.g. epoch 0), in which case
        _run_game will fall back to a freshly initialised random model.
        The returned dict is a reference to the cached copy — workers receive
        their own pickle-deserialized copy so mutations in workers are safe.
        """
        if not self.snapshots:
            return None
        path = self._sample_snapshot()["path"]
        return self._load_state_dict(path)

    def get_current_model(self) -> PokerNet:
        """Return the most recently saved model, or a fresh one if no snapshots exist."""
        if not self.snapshots:
            model = self.model_class()
            model.initialize_internal_state()
            return model
        return self.load(self.snapshots[-1]["path"])
