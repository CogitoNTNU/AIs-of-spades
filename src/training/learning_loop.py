import time
import traceback

import numpy as np
import torch
import torch.optim as optim
import torch.distributions as D
import torch.multiprocessing as mp

from nn.poker_net import PokerNet
from pokerenv.common import PlayerAction
from training.game_loop import Game
from training.weight_manager import WeightManager
from training.wandb_compat import wandb


# ---------------------------------------------------------------------------
# Module-level worker state — initialised once per worker process by
# _worker_init(), then reused across all pool.map() calls.
# ---------------------------------------------------------------------------

_worker_model = None
_worker_model_class = None


def _worker_init(model_class):
    """
    Pool initializer: runs exactly once per worker process after spawn.

    Importing torch and building the model here (instead of inside every
    _run_game call) avoids paying the NFS / import cost on every task.
    On a SLURM cluster with a network filesystem this was the dominant
    source of the ~600s/epoch slowdown vs ~10s/epoch on a local machine.
    """
    global _worker_model, _worker_model_class
    _worker_model_class = model_class
    _worker_model = _build_model(model_class)


# ---------------------------------------------------------------------------
# Worker (CPU-only, runs in subprocess)
# ---------------------------------------------------------------------------


def _run_game(args):
    """
    Subprocess worker: loads updated weights into the pre-built CPU model,
    plays a full game, and returns (reward, trajectory).

    Only the state_dict and game parameters are sent per task — the model
    object itself is reused from _worker_init, which saves the cost of
    re-importing torch and reconstructing the network on every call.

    CUDA must not be used here — mp.Pool workers cannot share a CUDA context.
    Exceptions are caught and re-raised with an embedded traceback because
    mp.Pool on Windows swallows worker tracebacks.
    """
    try:
        state_dict, hands_per_game, opponent_state_dicts = args

        # Reuse the pre-built model; just update its weights.
        _worker_model.load_state_dict(state_dict)
        _worker_model.eval()

        opponents = [
            _build_model(_worker_model_class, osd) for osd in opponent_state_dicts
        ]

        with torch.no_grad():
            reward, trajectory = Game(opponents, _worker_model).play(hands_per_game)

        return reward, trajectory

    except Exception:
        raise RuntimeError(f"Worker _run_game failed:\n{traceback.format_exc()}")


def _build_model(model_class, state_dict=None):
    """Instantiate, optionally load weights, and set to eval mode."""
    model = model_class()
    model.initialize_internal_state()
    if state_dict is not None:
        model.load_state_dict(state_dict)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


class LearningLoop:
    def __init__(self, weight_manager: WeightManager, config: dict):
        self.weight_manager = weight_manager
        self.config = config["learning_loop"]
        self.num_workers = self.config.get("num_workers", mp.cpu_count())

        # Model lives on CPU until start_learning() moves it to the target device.
        model_class = config["weight_manager"]["model_class"]
        self.current_model: PokerNet = _build_model(model_class)

        self.optimizer = optim.Adam(
            self.current_model.parameters(),
            lr=float(self.config.get("learning_rate", 1e-4)),
        )

        # Resolved in start_learning(); stored so helper methods can reference it.
        self.device: torch.device | None = None

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Restore model and optimizer state from a checkpoint.
        Returns the next epoch to resume from.
        Tensors are loaded onto CPU; the caller is responsible for moving the
        model to the target device afterwards.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.current_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(
            f"Checkpoint loaded from '{checkpoint_path}' — resuming from epoch {start_epoch}"
        )
        return start_epoch

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def start_learning(self, resume_from: str | None = None):
        """
        Main training loop.

        Architecture:
          - CPU worker pool  →  runs game simulations, collects trajectories.
          - Main process GPU →  re-runs forward passes, computes REINFORCE loss,
                                performs the gradient step.

        Key design decisions
        --------------------
        spawn context
            Used explicitly so workers start with a clean Python interpreter.
            This avoids PyTorch mutex / allocator deadlocks that occur when
            fork inherits an already-initialised torch state (the Linux default).

        Pool initializer (_worker_init)
            Each worker builds the model ONCE at startup.  Subsequent tasks
            only send the updated state_dict — not the model class — which
            eliminates the per-task import + construction overhead that caused
            the ~600 s/epoch regression on SLURM (NFS makes Python imports
            ~50–100× slower than on a local SSD).

        Note on share_memory_()
            share_memory_() is NOT used here intentionally.  With spawn, the
            child processes do not inherit the parent address space, so shared
            memory tensors still have to be pickled and sent through the pipe
            to each worker — the same cost as a regular CPU tensor.
            share_memory_() is only beneficial with fork or forkserver, where
            the child can directly mmap the parent's pages without copying.
            Using it with spawn adds complexity with no gain.
        """
        epochs = self.config.get("epochs", 1000)
        games_per_epoch = self.config.get("games_per_epoch", 10)
        hands_per_game = self.config.get("hands_per_game", 100)
        save_interval = self.config.get("save_interval", 20)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on device: {self.device}")

        self.current_model = self.current_model.to(self.device)

        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
            # load_checkpoint uses map_location="cpu", so re-send to device.
            self.current_model = self.current_model.to(self.device)

        model_class = self.current_model.__class__

        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=self.num_workers,
            initializer=_worker_init,  # <-- runs once per worker at startup
            initargs=(model_class,),
        ) as pool:
            try:
                for epoch in range(start_epoch, epochs):
                    self._run_epoch(
                        epoch,
                        epochs,
                        games_per_epoch,
                        hands_per_game,
                        save_interval,
                        pool,
                    )
            except KeyboardInterrupt:
                print("KeyboardInterrupt caught — terminating worker pool.")
                pool.terminate()
                pool.join()
                raise

    def _run_epoch(
        self, epoch, epochs, games_per_epoch, hands_per_game, save_interval, pool
    ):
        start_time = time.time()

        # Serialize weights to CPU so they can be pickled across the spawn
        # boundary.  One copy is sent per worker slot (not per game), because
        # Pool reuses worker processes across pool.map calls — each of the
        # num_workers processes receives the state_dict once per epoch.
        state_dict = {k: v.cpu() for k, v in self.current_model.state_dict().items()}

        worker_args = self._build_worker_args(
            state_dict, games_per_epoch, hands_per_game
        )

        t0 = time.time()
        results = pool.map(_run_game, worker_args)
        t_simulation = time.time() - t0

        batch_rewards = [r for r, _ in results]
        batch_trajectories = [t for _, t in results]
        total_steps = sum(len(t) for t in batch_trajectories)

        t0 = time.time()
        loss = self._compute_reinforce_loss(batch_trajectories, batch_rewards)
        t_loss = time.time() - t0

        t0 = time.time()
        self._gradient_step(loss)
        t_grad = time.time() - t0

        t0 = time.time()
        action_stats = self._compute_action_stats(batch_trajectories)
        t_stats = time.time() - t0

        t_total = time.time() - start_time
        t_overhead = t_total - (t_simulation + t_loss + t_grad + t_stats)
        avg_reward = np.mean(batch_rewards)

        self._log_epoch(
            epoch=epoch,
            loss=loss,
            avg_reward=avg_reward,
            batch_rewards=batch_rewards,
            batch_trajectories=batch_trajectories,
            total_steps=total_steps,
            action_stats=action_stats,
            t_total=t_total,
            t_simulation=t_simulation,
            t_loss=t_loss,
            t_grad=t_grad,
            t_stats=t_stats,
            t_overhead=t_overhead,
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"loss: {loss.item():.4f} | avg reward: {avg_reward:.4f} | "
            f"total: {t_total:.2f}s  sim: {t_simulation:.2f}s  "
            f"fwd: {t_loss:.2f}s  bwd: {t_grad:.2f}s  "
            f"overhead: {t_overhead:.2f}s | "
            f"device: {self.device}"
        )

        if epoch % save_interval == 0 or epoch == self.config.get("epochs", 1000) - 1:
            self.weight_manager.save(self.current_model, self.optimizer, epoch)

    def _build_worker_args(self, state_dict, games_per_epoch, hands_per_game):
        # model_class is no longer included — workers already have it from
        # _worker_init, so we avoid pickling the class on every task.
        return [
            (
                state_dict,
                hands_per_game,
                [self.weight_manager.sample_opponent_state_dict() for _ in range(5)],
            )
            for _ in range(games_per_epoch)
        ]

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_epoch(
        self,
        epoch,
        loss,
        avg_reward,
        batch_rewards,
        batch_trajectories,
        total_steps,
        action_stats,
        t_total,
        t_simulation,
        t_loss,
        t_grad,
        t_stats,
        t_overhead,
    ):
        reward_scale = float(np.mean(np.abs(batch_rewards)) + 1e-8)
        diversity_coef_effective = (
            float(self.config.get("diversity_coef", 0.1)) * reward_scale
        )

        wandb.log(
            data={
                "epoch": epoch,
                # Timing breakdown
                "time/total": t_total,
                "time/simulation": t_simulation,
                "time/loss_forward": t_loss,
                "time/grad_step": t_grad,
                "time/stats_logging": t_stats,
                "time/overhead": t_overhead,
                "time/steps_per_sec": total_steps / t_total if t_total > 0 else 0.0,
                "time/frac_simulation": t_simulation / t_total if t_total > 0 else 0.0,
                "time/frac_gpu": (t_loss + t_grad) / t_total if t_total > 0 else 0.0,
                # Training signal
                "train/loss": loss.item() if loss.requires_grad else 0.0,
                "train/avg_reward": avg_reward,
                "train/reward_std": np.std(batch_rewards),
                "train/reward_max": np.max(batch_rewards),
                "train/reward_min": np.min(batch_rewards),
                "train/avg_trajectory_len": np.mean(
                    [len(t) for t in batch_trajectories]
                ),
                "train/total_steps": total_steps,
                "train/grad_norm": self._get_grad_norm(),
                "train/diversity_coef_effective": diversity_coef_effective,
                **action_stats,
            }
        )

    # ------------------------------------------------------------------
    # REINFORCE
    # ------------------------------------------------------------------

    def _compute_reinforce_loss(
        self, batch_trajectories: list, batch_rewards: list
    ) -> torch.Tensor:
        """
        REINFORCE loss over a batch of episodes, computed on self.device.

        For each step t in episode i:
            loss += -A_i * (log_p_discrete_t + log_p_continuous_t)

        Plus a batch-level diversity penalty:
            loss += diversity_coef * Σ_a( relu(freq_a - hi) + relu(lo - freq_a) )

        where freq_a is the fraction of batch steps where action a was sampled.
        The penalty fires only when aggregate action frequencies exit [lo, hi]
        (default [0.01, 0.99]), leaving the model free to be confident on any
        individual step.  diversity_coef is scaled by mean |reward| to stay
        proportional to reward magnitude throughout training.
        """
        device = self.device

        # Normalize rewards into advantages to reduce variance.
        advantages = np.array(batch_rewards, dtype=np.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        step_losses = []
        action_probs_all = []

        for trajectory, advantage in zip(batch_trajectories, advantages):
            if not trajectory:
                continue

            reward_tensor = torch.tensor(advantage, dtype=torch.float32, device=device)

            for obs, action in trajectory:
                obs_gpu = obs.to(device) if isinstance(obs, torch.Tensor) else obs

                action_logits, bet_mean, bet_std = self.current_model.forward(obs_gpu)

                log_p_discrete = D.Categorical(logits=action_logits).log_prob(
                    action.action_tensor.to(device)
                )
                log_p_continuous = D.Normal(bet_mean, bet_std).log_prob(
                    action.bet_tensor.to(device)
                )

                step_losses.append(-reward_tensor * (log_p_discrete + log_p_continuous))
                action_probs_all.append(D.Categorical(logits=action_logits).probs)

        if not step_losses:
            return torch.tensor(0.0, requires_grad=False, device=device)

        reinforce_loss = torch.stack(step_losses).mean()
        diversity_penalty = self._compute_diversity_penalty(
            action_probs_all, batch_rewards
        )

        return reinforce_loss + diversity_penalty

    def _compute_diversity_penalty(
        self, action_probs_all: list, batch_rewards: list
    ) -> torch.Tensor:
        lo = float(self.config.get("diversity_lo", 0.01))
        hi = float(self.config.get("diversity_hi", 0.99))
        reward_scale = float(np.mean(np.abs(batch_rewards)) + 1e-8)
        coef = float(self.config.get("diversity_coef", 0.1)) * reward_scale

        mean_probs = torch.stack(action_probs_all).mean(dim=0)  # (n_actions,)
        penalty = (torch.relu(mean_probs - hi) + torch.relu(lo - mean_probs)).sum()

        return coef * penalty

    # ------------------------------------------------------------------
    # Optimizer helpers
    # ------------------------------------------------------------------

    def _gradient_step(self, loss: torch.Tensor):
        """Backprop + gradient clipping + optimizer step."""
        if not loss.requires_grad:
            return
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.current_model.parameters(), max_norm=1.0)
        self.optimizer.step()

    def _get_grad_norm(self) -> float:
        """L2 norm of all gradients after backward()."""
        total_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in self.current_model.parameters()
            if p.grad is not None
        )
        return total_norm**0.5

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _compute_action_stats(self, batch_trajectories: list) -> dict:
        """Aggregate action frequencies and bet sizes across all trajectories."""
        counts = {PlayerAction.FOLD: 0, PlayerAction.BET: 0, PlayerAction.CALL: 0}
        bet_amounts = []
        traj_lengths = [len(t) for t in batch_trajectories]

        for trajectory in batch_trajectories:
            for _, action in trajectory:
                counts[action.action_type] += 1
                if action.action_type == PlayerAction.BET:
                    bet_amounts.append(action.bet_amount)

        total = sum(counts.values()) or 1

        return {
            "action/fold": counts[PlayerAction.FOLD] / total,
            "action/bet": counts[PlayerAction.BET] / total,
            "action/call": counts[PlayerAction.CALL] / total,
            "action/bet_amount": np.mean(bet_amounts) if bet_amounts else 0.0,
            "game/avg_hands_per_trajectory": np.mean(traj_lengths),
            "game/min_hands_per_trajectory": np.min(traj_lengths),
            "game/max_hands_per_trajectory": np.max(traj_lengths),
        }
