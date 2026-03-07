import os
import random
import torch
from collections import deque

class OpponentPool:
    def __init__(self, pool_dir, max_size=100, keep_latest=20, device="cpu"):
        self.pool_dir = pool_dir
        self.max_size = max_size
        self.keep_latest = keep_latest
        self.device = device
        self.snapshots = []  # list of dicts: {"epoch": int, "path": str, "state_dict": ...}

        os.makedirs(pool_dir, exist_ok=True)

    def add_snapshot(self, model, epoch):
        path = os.path.join(self.pool_dir, f"pool_{epoch:06d}.pt")
        state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(state_dict, path)

        self.snapshots.append({
            "epoch": epoch,
            "path": path,
            "state_dict": state_dict,
        })

        self._trim_pool()

    def _trim_pool(self):
        if len(self.snapshots) <= self.max_size:
            return

        latest = self.snapshots[-self.keep_latest:]
        older = self.snapshots[:-self.keep_latest]

        if older:
            to_remove = random.choice(older)
            os.remove(to_remove["path"])
            self.snapshots.remove(to_remove)

    def sample_snapshot(self):
        n = len(self.snapshots)
        if n == 0:
            return None

        weights = list(range(1, n + 1))  # newer snapshots get higher weight
        chosen = random.choices(self.snapshots, weights=weights, k=1)[0]
        return chosen["state_dict"]