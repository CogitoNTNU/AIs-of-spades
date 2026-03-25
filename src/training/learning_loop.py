import os
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
# Module-level worker state — initialised ONCE per worker process by
# _worker_init(), then reused across every pool.map() call.
#
# Key design: model objects are allocated once at worker startup and reused
# via load_state_dict() on every task.  This avoids re-importing torch and
# re-allocating nn.Module objects on every game, which is catastrophically
# slow on NFS-mounted cluster filesystems (50-100x slower than local SSD).
# ---------------------------------------------------------------------------

_worker_model: "PokerNet | None" = None
_worker_model_class = None

# Pre-allocated opponent slots — one per maximum possible opponent seat.
# load_state_dict() is cheap (tensor copy in RAM); nn.Module.__init__() is
# not (Python class instantiation + NFS import hits on every attribute).
_worker_opponents: list = []

_MAX_OPPONENTS = 5


# ---------------------------------------------------------------------------
# Worker initializer — runs once per subprocess at pool startup
# ---------------------------------------------------------------------------


def _worker_init(model_class):
    global _worker_model, _worker_model_class, _worker_opponents

    _worker_model_class = model_class
    _worker_model = _build_model(model_class)

    # Pre-allocate all opponent slots upfront.  Dynamic sampling still works:
    # each game receives fresh state_dicts chosen by WeightManager; we only
    # avoid re-instantiating the nn.Module objects.
    _worker_opponents = [_build_model(model_class) for _ in range(_MAX_OPPONENTS)]

    print(
        f"[worker {os.getpid()}] init complete — "
        f"main model + {_MAX_OPPONENTS} opponent slots allocated",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Worker task — runs once per game, called by pool.map()
# ---------------------------------------------------------------------------


def _run_game(args):
    t_start = time.time()
    pid = os.getpid()
    print(f"[worker {pid}] game start", flush=True)

    try:
        state_dict, hands_per_game, opponent_state_dicts = args

        # ── Load current model weights ────────────────────────────────
        t0 = time.time()
        _worker_model.load_state_dict(state_dict)
        _worker_model.eval()
        print(
            f"[worker {pid}] current model loaded in {time.time() - t0:.3f}s",
            flush=True,
        )

        # ── Load opponent weights into pre-allocated slots ────────────
        # Each game gets dynamically sampled opponents (diversity preserved).
        # Only load_state_dict() is called — no nn.Module allocation, no NFS
        # import hits.
        t0 = time.time()
        opponents = []
        for slot, osd in zip(_worker_opponents, opponent_state_dicts):
            if osd is not None:
                slot.load_state_dict(osd)
            else:
                # No checkpoint saved yet (e.g. epoch 0) — use random weights.
                slot.initialize_internal_state()
            slot.eval()
            opponents.append(slot)
        n_opp = len(opponents)
        print(
            f"[worker {pid}] {n_opp} opponents loaded in {time.time() - t0:.3f}s",
            flush=True,
        )

        # ── Play ──────────────────────────────────────────────────────
        t0 = time.time()
        with torch.no_grad():
            reward, trajectory = Game(opponents, _worker_model).play(hands_per_game)
        t_game = time.time() - t0

        t_total = time.time() - t_start
        print(
            f"[worker {pid}] game done — "
            f"reward={reward:.3f}  steps={len(trajectory)}  "
            f"game={t_game:.2f}s  total={t_total:.2f}s",
            flush=True,
        )
        return reward, trajectory

    except Exception:
        raise RuntimeError(
            f"[worker {os.getpid()}] _run_game failed:\n{traceback.format_exc()}"
        )


# ---------------------------------------------------------------------------
# Helper: build and optionally warm a model
# ---------------------------------------------------------------------------


def _build_model(model_class, state_dict=None):
    """Instantiate a model, optionally load weights, set to eval mode."""
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

        model_class = config["weight_manager"]["model_class"]
        self.current_model: PokerNet = _build_model(model_class)

        self.optimizer = optim.Adam(
            self.current_model.parameters(),
            lr=float(self.config.get("learning_rate", 1e-4)),
        )

        self.device: torch.device | None = None

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Restore model and optimizer from a checkpoint file.
        Returns the next epoch index to resume from.
        Tensors are loaded to CPU; the caller must move the model to the
        target device afterwards.
        """
        print(f"[main] loading checkpoint from '{checkpoint_path}' ...", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.current_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(
            f"[main] checkpoint loaded — resuming from epoch {start_epoch}",
            flush=True,
        )
        return start_epoch

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def start_learning(self, resume_from: str | None = None):
        """
        Main training loop.

        Architecture
        ------------
        CPU worker pool   runs game simulations and collects trajectories.
        Main process GPU  re-runs forward passes, computes REINFORCE loss,
                          and performs the gradient step.

        Spawn context
        -------------
        We use 'spawn' so workers start with a clean Python interpreter.
        This avoids PyTorch mutex / allocator deadlocks that occur when fork
        inherits an already-initialised torch state (the Linux default).

        Worker initializer
        ------------------
        _worker_init builds the model and all opponent slots ONCE per worker
        at pool startup.  Subsequent tasks only receive updated state_dicts
        (not model classes), eliminating per-task allocation and NFS import
        overhead that caused the ~600 s/epoch regression on SLURM.

        Note on share_memory_()
        -----------------------
        share_memory_() is intentionally NOT used here.  With spawn, child
        processes do not inherit the parent address space, so shared-memory
        tensors are still pickled and sent through the pipe — the same cost
        as a regular CPU tensor.  share_memory_() only helps with fork or
        forkserver.  Using it with spawn adds complexity with no gain.
        """
        epochs = self.config.get("epochs", 1000)
        games_per_epoch = self.config.get("games_per_epoch", 10)
        hands_per_game = self.config.get("hands_per_game", 100)
        save_interval = self.config.get("save_interval", 20)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[main] training device: {self.device}", flush=True)
        print(
            f"[main] config — epochs={epochs}  games/epoch={games_per_epoch}  "
            f"hands/game={hands_per_game}  workers={self.num_workers}",
            flush=True,
        )

        self.current_model = self.current_model.to(self.device)

        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
            # load_checkpoint uses map_location="cpu"; re-send to device.
            self.current_model = self.current_model.to(self.device)

        model_class = self.current_model.__class__

        print(
            f"[main] spawning pool with {self.num_workers} workers ...",
            flush=True,
        )
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=self.num_workers,
            initializer=_worker_init,
            initargs=(model_class,),
        ) as pool:
            print(f"[main] pool ready", flush=True)
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
                print("[main] KeyboardInterrupt — terminating pool", flush=True)
                pool.terminate()
                pool.join()
                raise

    # ------------------------------------------------------------------
    # Single epoch
    # ------------------------------------------------------------------

    def _run_epoch(
        self, epoch, epochs, games_per_epoch, hands_per_game, save_interval, pool
    ):
        print(
            f"\n[main] ── epoch {epoch + 1}/{epochs} ──────────────────────────",
            flush=True,
        )
        t_epoch_start = time.time()

        # Serialize model weights to CPU once.  The same dict is pickled and
        # sent to every worker slot; each worker deserializes its own copy.
        t0 = time.time()
        state_dict = {k: v.cpu() for k, v in self.current_model.state_dict().items()}
        print(
            f"[main] state_dict serialized to CPU in {time.time() - t0:.3f}s",
            flush=True,
        )

        # Sample opponent state_dicts on the main process (WeightManager is
        # not thread/process safe and must not be called from workers).
        t0 = time.time()
        worker_args = self._build_worker_args(
            state_dict, games_per_epoch, hands_per_game
        )
        print(
            f"[main] opponent sampling done in {time.time() - t0:.3f}s "
            f"({games_per_epoch} games × {_MAX_OPPONENTS} opponents)",
            flush=True,
        )

        # ── Simulation ────────────────────────────────────────────────
        print(
            f"[main] dispatching {games_per_epoch} games to pool ...",
            flush=True,
        )
        t0 = time.time()
        results = pool.map(_run_game, worker_args)
        t_simulation = time.time() - t0

        batch_rewards = [r for r, _ in results]
        batch_trajectories = [t for _, t in results]
        total_steps = sum(len(t) for t in batch_trajectories)

        print(
            f"[main] simulation done in {t_simulation:.2f}s — "
            f"total steps={total_steps}  "
            f"avg reward={np.mean(batch_rewards):.4f}  "
            f"reward std={np.std(batch_rewards):.4f}",
            flush=True,
        )

        # ── Loss ──────────────────────────────────────────────────────
        print(f"[main] computing REINFORCE loss on {self.device} ...", flush=True)
        t0 = time.time()
        loss = self._compute_reinforce_loss(batch_trajectories, batch_rewards)
        t_loss = time.time() - t0
        print(
            f"[main] loss={loss.item():.6f}  computed in {t_loss:.3f}s",
            flush=True,
        )

        # ── Gradient step ─────────────────────────────────────────────
        print(f"[main] backward + optimizer step ...", flush=True)
        t0 = time.time()
        self._gradient_step(loss)
        t_grad = time.time() - t0
        print(
            f"[main] grad step done in {t_grad:.3f}s  "
            f"grad_norm={self._get_grad_norm():.4f}",
            flush=True,
        )

        # ── Stats + logging ───────────────────────────────────────────
        t0 = time.time()
        action_stats = self._compute_action_stats(batch_trajectories)
        t_stats = time.time() - t0

        t_total = time.time() - t_epoch_start
        t_overhead = t_total - (t_simulation + t_loss + t_grad + t_stats)
        avg_reward = np.mean(batch_rewards)

        print(
            f"[main] action mix — "
            f"fold={action_stats['action/fold']:.2%}  "
            f"call={action_stats['action/call']:.2%}  "
            f"bet={action_stats['action/bet']:.2%}  "
            f"avg_bet={action_stats['action/bet_amount']:.2f}",
            flush=True,
        )
        print(
            f"[main] timing — "
            f"total={t_total:.2f}s  "
            f"sim={t_simulation:.2f}s ({t_simulation / t_total:.0%})  "
            f"fwd={t_loss:.2f}s  "
            f"bwd={t_grad:.2f}s  "
            f"overhead={t_overhead:.2f}s",
            flush=True,
        )
        print(
            f"[main] epoch {epoch + 1} summary — "
            f"loss={loss.item():.4f}  avg_reward={avg_reward:.4f}  "
            f"steps/s={total_steps / t_total:.1f}",
            flush=True,
        )

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

        # ── Checkpoint ────────────────────────────────────────────────
        should_save = (
            epoch % save_interval == 0 or epoch == self.config.get("epochs", 1000) - 1
        )
        if should_save:
            print(f"[main] saving checkpoint for epoch {epoch} ...", flush=True)
            self.weight_manager.save(self.current_model, self.optimizer, epoch)
            print(f"[main] checkpoint saved", flush=True)

    # ------------------------------------------------------------------
    # Worker args builder
    # ------------------------------------------------------------------

    def _build_worker_args(self, state_dict, games_per_epoch, hands_per_game):
        """
        Build one argument tuple per game.

        Opponent state_dicts are sampled here on the main process (WeightManager
        is not multiprocess-safe).  Each game receives independently sampled
        opponents, preserving population diversity even though opponent slots
        inside workers are reused across games.
        """
        args = []
        for i in range(games_per_epoch):
            opponent_state_dicts = [
                self.weight_manager.sample_opponent_state_dict()
                for _ in range(_MAX_OPPONENTS)
            ]
            n_none = sum(1 for osd in opponent_state_dicts if osd is None)
            if n_none:
                print(
                    f"[main]   game {i}: {n_none}/{_MAX_OPPONENTS} opponents "
                    f"have no checkpoint yet (will use random weights)",
                    flush=True,
                )
            args.append((state_dict, hands_per_game, opponent_state_dicts))
        return args

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
        individual step.  diversity_coef is scaled by mean |reward| so the
        penalty stays proportional to reward magnitude throughout training.
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
                action_probs_all.append(
                    D.Categorical(logits=action_logits).probs.squeeze()
                )

        if not step_losses:
            print(
                "[main] WARNING: no steps found in trajectories — returning zero loss",
                flush=True,
            )
            return torch.tensor(0.0, requires_grad=False, device=device)

        reinforce_loss = torch.stack(step_losses).mean()
        diversity_penalty = self._compute_diversity_penalty(
            action_probs_all, batch_rewards
        )

        print(
            f"[main]   reinforce_loss={reinforce_loss.item():.6f}  "
            f"diversity_penalty={diversity_penalty.item():.6f}  "
            f"steps={len(step_losses)}",
            flush=True,
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

        print(
            f"[main]   mean action probs — "
            + "  ".join(f"a{i}={p:.3f}" for i, p in enumerate(mean_probs.tolist()))
            + f"  coef={coef:.4f}",
            flush=True,
        )
        return coef * penalty

    # ------------------------------------------------------------------
    # Optimizer helpers
    # ------------------------------------------------------------------

    def _gradient_step(self, loss: torch.Tensor):
        """Backprop + gradient clipping + optimizer step."""
        if not loss.requires_grad:
            print("[main] WARNING: loss has no grad — skipping backward", flush=True)
            return
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.current_model.parameters(), max_norm=1.0)
        self.optimizer.step()

    def _get_grad_norm(self) -> float:
        """L2 norm of all parameter gradients after backward()."""
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
