import os
import time
import traceback
from collections import deque

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

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    _worker_model_class = model_class
    _worker_model = _build_model(model_class)
    _worker_opponents = [_build_model(model_class) for _ in range(_MAX_OPPONENTS)]


# ---------------------------------------------------------------------------
# Worker task — runs once per game, called by pool.map()
# ---------------------------------------------------------------------------


def _run_game(args):
    try:
        state_dict, hands_per_game, opponent_state_dicts, game_config = args

        _worker_model.load_state_dict(state_dict)
        _worker_model.eval()

        opponents = []
        for slot, osd in zip(_worker_opponents, opponent_state_dicts):
            if osd is not None:
                slot.load_state_dict(osd)
            else:
                slot.initialize_internal_state()
            slot.eval()
            opponents.append(slot)

        with torch.no_grad():
            trajectory, bonus_events, log_data = Game(
                opponents, _worker_model, game_config
            ).play(hands_per_game)

        return trajectory, bonus_events, log_data

    except Exception:
        raise RuntimeError(
            f"[worker {os.getpid()}] _run_game failed:\n{traceback.format_exc()}"
        )


# ---------------------------------------------------------------------------
# Helper: instantiate a model, optionally load weights, set to eval mode
# ---------------------------------------------------------------------------


def _build_model(model_class, state_dict=None):
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
        self.game_config = config.get("game_loop", {})
        self.num_workers = self.config.get("num_workers", mp.cpu_count())

        model_class = config["weight_manager"]["model_class"]
        self.current_model: PokerNet = _build_model(model_class)

        self.optimizer = optim.Adam(
            self.current_model.parameters(),
            lr=float(self.config.get("learning_rate", 1e-4)),
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.get("lr_decay_steps", 100),
            gamma=self.config.get("lr_decay_gamma", 0.9),
        )

        self.device: torch.device | None = None

        # Per-action baselines: fold=0, bet=1, call=2.
        # Rolling mean over the last N epochs per action — much faster to
        # adapt than EMA with alpha=0.01 (which needs ~460 epochs to converge).
        baseline_window = self.config.get("action_baseline_window", 50)
        self._action_reward_history: list[deque] = [
            deque(maxlen=baseline_window) for _ in range(3)
        ]
        self._action_baselines = np.zeros(3, dtype=np.float64)

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
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "action_baselines" in checkpoint:
            self._action_baselines = np.array(checkpoint["action_baselines"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"[main] resuming from epoch {start_epoch}", flush=True)
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

        print(
            f"[main] device={self.device}  epochs={epochs}  "
            f"games/epoch={games_per_epoch}  hands/game={hands_per_game}  "
            f"workers={self.num_workers}",
            flush=True,
        )

        self.current_model = self.current_model.to(self.device)

        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
            self.current_model = self.current_model.to(self.device)

        model_class = self.current_model.__class__

        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=self.num_workers,
            initializer=_worker_init,
            initargs=(model_class,),
        ) as pool:
            print(f"[main] pool ready — {self.num_workers} workers", flush=True)
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
                print("[main] interrupted — terminating pool", flush=True)
                pool.terminate()
                pool.join()
                raise

    # ------------------------------------------------------------------
    # Single epoch
    # ------------------------------------------------------------------

    def _run_epoch(
        self, epoch, epochs, games_per_epoch, hands_per_game, save_interval, pool
    ):
        t_epoch_start = time.time()

        state_dict = {k: v.cpu() for k, v in self.current_model.state_dict().items()}

        mult_start = float(self.game_config.get("showdown_reward_multiplier_start", 1.0))
        mult_end = float(self.game_config.get("showdown_reward_multiplier_end", 1.0))
        decay_epochs = int(self.game_config.get("showdown_reward_multiplier_epochs", epochs))
        t = min(epoch / max(decay_epochs - 1, 1), 1.0)
        showdown_mult = mult_start + (mult_end - mult_start) * t
        game_config_epoch = {**self.game_config, "showdown_reward_multiplier": showdown_mult}

        worker_args = self._build_worker_args(
            state_dict, games_per_epoch, hands_per_game, game_config_epoch
        )

        # ── Simulation ────────────────────────────────────────────────
        t0 = time.time()
        results = pool.map(_run_game, worker_args)
        t_simulation = time.time() - t0

        batch_trajectories = [r[0] for r in results]
        batch_bonus_events = [r[1] for r in results]
        batch_log_data = [r[2] for r in results]

        # Average numeric log_data values across games for a stable per-epoch metric.
        game_log_data: dict = {}
        if batch_log_data:
            for key in batch_log_data[0]:
                vals = [d[key] for d in batch_log_data if key in d]
                game_log_data[key] = float(np.mean(vals)) if vals else 0.0

        bonus_totals = {
            "elimination": 0,
            "survival": 0,
            "isolation": 0,
            "showdown": 0,
            "steal": 0,
        }
        for game_events in batch_bonus_events:
            for k, v in game_events.items():
                bonus_totals[k] += v

        batch_rewards = [
            np.mean([step[2] for step in t]) for t in batch_trajectories if t
        ]
        total_steps = sum(len(t) for t in batch_trajectories)

        # ── Loss ──────────────────────────────────────────────────────
        t0 = time.time()
        (
            loss,
            mean_probs,
            diversity_penalty,
            loss_discrete,
            loss_continuous,
            adv_stats,
        ) = self._compute_reinforce_loss(batch_trajectories)
        t_loss = time.time() - t0

        # ── Gradient step ─────────────────────────────────────────────
        t0 = time.time()
        grad_norm_raw = self._gradient_step(loss)
        self.scheduler.step()
        t_grad = time.time() - t0

        # ── Stats ─────────────────────────────────────────────────────
        t0 = time.time()
        action_stats = self._compute_action_stats(batch_trajectories)
        t_stats = time.time() - t0

        # Post-clip grad norm — reused in both print and log
        grad_norm = self._get_grad_norm()

        t_total = time.time() - t_epoch_start
        avg_reward = np.mean(batch_rewards)

        probs_str = "  ".join(
            f"{name}={p:.2%}"
            for name, p in zip(["fold", "bet", "call"], mean_probs.tolist())
        )
        baselines_str = "  ".join(
            f"{name}_bl={b:.3f}"
            for name, b in zip(["fold", "bet", "call"], self._action_baselines)
        )
        print(
            f"[{epoch + 1:>5}/{epochs}] "
            f"loss={loss.item():+.4f}  "
            f"reward={avg_reward:+8.1f}±{np.std(batch_rewards):.1f}  "
            f"steps={total_steps:>5}  {probs_str}  "
            f"grad={grad_norm:.3f}  grad_raw={grad_norm_raw:.3f}  "
            f"sim={t_simulation:.1f}s  fwd={t_loss:.1f}s  bwd={t_grad:.1f}s",
            flush=True,
        )
        print(
            f"[{epoch + 1:>5}/{epochs}] "
            f"loss_disc={loss_discrete:+.4f}  loss_cont={loss_continuous:+.4f}  "
            f"{baselines_str}",
            flush=True,
        )

        if diversity_penalty.item() > 0:
            print(
                f"[{epoch + 1:>5}/{epochs}] WARNING diversity penalty = "
                f"{diversity_penalty.item():.4f}",
                flush=True,
            )

        # ── Logging ───────────────────────────────────────────────────
        self._log_epoch(
            epoch=epoch,
            loss=loss,
            loss_discrete=loss_discrete,
            loss_continuous=loss_continuous,
            avg_reward=avg_reward,
            batch_rewards=batch_rewards,
            batch_trajectories=batch_trajectories,
            total_steps=total_steps,
            action_stats=action_stats,
            bonus_totals=bonus_totals,
            games_per_epoch=games_per_epoch,
            adv_stats=adv_stats,
            grad_norm=grad_norm,
            grad_norm_raw=grad_norm_raw,
            game_log_data=game_log_data,
            t_total=t_total,
            t_simulation=t_simulation,
            t_loss=t_loss,
            t_grad=t_grad,
            t_stats=t_stats,
            t_overhead=t_total - (t_simulation + t_loss + t_grad + t_stats),
        )

        # ── Checkpoint ────────────────────────────────────────────────
        should_save = (
            epoch % save_interval == 0 or epoch == self.config.get("epochs", 1000) - 1
        )
        if should_save:
            self.weight_manager.save(
                self.current_model,
                self.optimizer,
                epoch,
                self.scheduler,
                self._action_baselines,
            )
            print(f"[{epoch + 1:>5}/{epochs}] checkpoint saved", flush=True)

    # ------------------------------------------------------------------
    # Worker args builder
    # ------------------------------------------------------------------

    def _build_worker_args(self, state_dict, games_per_epoch, hands_per_game, game_config=None):
        """
        Build one argument tuple per game.

        Opponent state_dicts are sampled here on the main process (WeightManager
        is not multiprocess-safe).  Each game receives independently sampled
        opponents, preserving population diversity even though opponent slots
        inside workers are reused across games.
        """
        if game_config is None:
            game_config = self.game_config
        args = []
        none_count = 0
        for _ in range(games_per_epoch):
            opponent_state_dicts = [
                self.weight_manager.sample_opponent_state_dict()
                for _ in range(_MAX_OPPONENTS)
            ]
            none_count += sum(1 for osd in opponent_state_dicts if osd is None)
            args.append(
                (state_dict, hands_per_game, opponent_state_dicts, game_config)
            )

        if none_count > 0:
            print(
                f"[main] {none_count} opponent slots using random weights "
                f"(no checkpoint yet)",
                flush=True,
            )
        return args

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_epoch(
        self,
        epoch,
        loss,
        loss_discrete,
        loss_continuous,
        avg_reward,
        batch_rewards,
        batch_trajectories,
        total_steps,
        action_stats,
        bonus_totals,
        games_per_epoch,
        adv_stats,
        grad_norm,
        grad_norm_raw,
        game_log_data,
        t_total,
        t_simulation,
        t_loss,
        t_grad,
        t_stats,
        t_overhead,
    ):
        reward_scale = float(np.mean(np.abs(batch_rewards)) + 1e-8)
        continuous_weight = float(self.config.get("continuous_weight", 0.01))
        diversity_coef_effective = float(self.config.get("diversity_coef", 0.1)) * reward_scale

        n_games = max(games_per_epoch, 1)
        bonus_rates = {
            f"bonus/{k}": v / n_games for k, v in bonus_totals.items()
        }

        wandb.log(
            data={
                "epoch": epoch,
                # ── Timing ────────────────────────────────────────────────
                "time/total": t_total,
                "time/simulation": t_simulation,
                "time/loss_forward": t_loss,
                "time/grad_step": t_grad,
                "time/stats": t_stats,
                "time/overhead": t_overhead,
                "time/steps_per_sec": total_steps / t_total if t_total > 0 else 0.0,
                "time/frac_simulation": t_simulation / t_total if t_total > 0 else 0.0,
                "time/frac_gpu": (t_loss + t_grad) / t_total if t_total > 0 else 0.0,
                # ── Loss ──────────────────────────────────────────────────
                "loss/total": loss.item() if loss.requires_grad else 0.0,
                "loss/discrete": loss_discrete,
                "loss/continuous": loss_continuous,
                "loss/continuous_weight": continuous_weight,
                "loss/diversity_coef_effective": diversity_coef_effective,
                # ── Training signal ───────────────────────────────────────
                "train/avg_reward": avg_reward,
                "train/reward_std": np.std(batch_rewards),
                "train/reward_max": np.max(batch_rewards),
                "train/reward_min": np.min(batch_rewards),
                "train/total_steps": total_steps,
                "train/grad_norm": grad_norm,
                "train/grad_norm_raw": grad_norm_raw,
                "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                # ── Per-action baselines ───────────────────────────────────
                "baseline/fold": self._action_baselines[0],
                "baseline/bet": self._action_baselines[1],
                "baseline/call": self._action_baselines[2],
                # ── Per-action advantages ──────────────────────────────────
                **adv_stats,
                # ── Action frequencies & game stats ───────────────────────
                **action_stats,
                # ── Game-level metrics from simulation ────────────────────
                **game_log_data,
                # ── Bonus event rates (per game) ──────────────────────────
                **bonus_rates,
            }
        )

    # ------------------------------------------------------------------
    # REINFORCE
    # ------------------------------------------------------------------

    def _compute_reinforce_loss(
        self, batch_trajectories: list
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, dict]:
        """
        Compute the REINFORCE loss over a batch of episodes.

        Per-action baselines
        --------------------
        Baselines use a rolling mean over the last N epochs of raw rewards
        per action type.  This adapts much faster than EMA with small alpha
        and remains interpretable: the baseline for each action is simply
        the average reward that action has produced recently.

        Continuous loss masking
        -----------------------
        The bet distribution loss is computed only for BET steps.  Computing
        it on FOLD/CALL steps produces spurious gradients on bet_mean and
        bet_std because bet_tensor is a placeholder for those actions.

        Returns
        -------
        loss                    : scalar tensor (reinforce + diversity penalty)
        mean_probs              : [3] mean action probabilities across all steps
        diversity_penalty       : scalar tensor
        reinforce_discrete_mean : float  (for logging)
        reinforce_continuous_mean : float  (for logging)
        adv_stats               : dict of per-action advantage mean/std for wandb
        """
        device = self.device

        episode_rewards = np.array(
            [reward for t in batch_trajectories for _, _, reward in t],
            dtype=np.float32,
        )

        if episode_rewards.std() < 1e-8:
            zero = torch.tensor(0.0, requires_grad=False, device=device)
            return zero, torch.zeros(3, device=device), zero, 0.0, 0.0, {}

        # ── Single pass: collect flat data ───────────────────────────────
        total_steps = sum(len(t) for t in batch_trajectories)

        flat_preprocessed = [None] * total_steps
        flat_actions = [None] * total_steps
        raw_rewards = np.empty(total_steps, dtype=np.float64)
        action_indices = np.empty(total_steps, dtype=np.int64)

        idx = 0
        for traj in batch_trajectories:
            for preprocessed, action, reward in traj:
                flat_preprocessed[idx] = preprocessed
                flat_actions[idx] = action
                raw_rewards[idx] = reward
                action_indices[idx] = int(action.action_tensor.item())
                idx += 1

        if idx == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return (
                zero * sum(p.sum() for p in self.current_model.parameters()),
                torch.zeros(3, device=device),
                torch.tensor(0.0, device=device),
                0.0,
                0.0,
                {},
            )

        # ── Update rolling baselines ──────────────────────────────────────
        # Extend each action's reward history with this epoch's observations,
        # then recompute the baseline as the mean of the rolling window.
        for action_idx in range(3):
            mask = action_indices == action_idx
            if mask.any():
                epoch_mean = float(raw_rewards[mask].mean())
                self._action_reward_history[action_idx].append(epoch_mean)

        self._action_baselines = np.array(
            [np.mean(h) if h else 0.0 for h in self._action_reward_history]
        )

        # ── Per-action advantages on raw rewards ─────────────────────────
        raw_advantages = raw_rewards - self._action_baselines[action_indices]

        # ── Per-action std normalisation ─────────────────────────────────
        # Per-action baselines already centre each action's distribution, so
        # the global mean of raw_advantages ≈ 0.  The remaining problem is
        # that a global std is dominated by the majority action (call ~97%).
        # Dividing each action's slice by its own std gives every action a
        # comparable gradient scale while preserving the cross-action signal
        # already encoded in the baselines.
        advantages = np.empty_like(raw_advantages)
        global_std = float(raw_advantages.std()) + 1e-8  # fallback scale
        for a_idx in range(3):
            mask = action_indices == a_idx
            if mask.any():
                s = raw_advantages[mask]
                # Use per-action std, but floor at global_std to avoid
                # exploding advantages when an action has very few samples
                # (low sample count → std ≈ 0 → division by ~1e-8).
                std = max(float(s.std()), global_std)
                advantages[mask] = s / std
            else:
                advantages[mask] = 0.0

        # ── Sqrt inverse-frequency weights (positive advantages only) ────
        # Weight each step by sqrt(N / n_action) to give minority actions
        # more gradient mass — BUT only when the advantage is positive.
        # Applying amplification to negative advantages creates a feedback
        # loop: rare actions fail → amplified negative gradient → model
        # avoids them even more → they become rarer → collapse to one action.
        freq_weights = np.ones(idx, dtype=np.float32)
        for a_idx in range(3):
            mask = action_indices == a_idx
            n = mask.sum()
            if n > 0:
                freq_weights[mask] = np.sqrt(idx / n)
        # Apply frequency boost only where advantage is positive.
        positive_adv = advantages > 0
        step_weights = np.where(positive_adv, freq_weights, np.ones(idx, dtype=np.float32))
        # Renormalise so sum(w) == N, keeping disc_loss on the same scale as
        # an unweighted mean.  This preserves the continuous_weight ratio.
        step_weights *= idx / step_weights.sum()

        # ── Forward pass ─────────────────────────────────────────────────
        flat_trajectory = list(zip(flat_preprocessed, flat_actions))
        action_logits, bet_mean, bet_std = self.current_model.forward_batch(
            flat_trajectory
        )

        action_batch = torch.stack([a.action_tensor for a in flat_actions]).to(device)
        bet_batch = torch.stack([a.bet_tensor for a in flat_actions]).to(device)
        adv_batch = torch.tensor(advantages, dtype=torch.float32, device=device)
        weight_batch = torch.tensor(step_weights, dtype=torch.float32, device=device)

        # ── Discrete loss ─────────────────────────────────────────────────
        log_p_discrete = D.Categorical(logits=action_logits).log_prob(action_batch)
        reinforce_discrete = -adv_batch * log_p_discrete * weight_batch
        disc_loss = reinforce_discrete.sum() / weight_batch.sum()

        # ── Continuous loss — BET steps only ─────────────────────────────
        is_bet = (action_batch == 1).float()
        bet_std_clamped = bet_std.clamp(min=0.05)
        log_p_continuous = (
            D.Normal(bet_mean, bet_std_clamped).log_prob(bet_batch).squeeze(-1)
        ).clamp(-5.0, 5.0)
        reinforce_continuous = -adv_batch * log_p_continuous * is_bet * weight_batch
        cont_loss = reinforce_continuous.sum() / (is_bet * weight_batch).sum().clamp(min=1.0)

        continuous_weight = float(self.config.get("continuous_weight", 0.01))
        reinforce_loss = disc_loss + continuous_weight * cont_loss

        action_probs = D.Categorical(logits=action_logits).probs
        diversity_penalty, mean_probs = self._compute_diversity_penalty(
            action_probs, episode_rewards
        )

        # ── Per-action advantage and weight stats for logging ─────────────
        adv_stats = {}
        weight_total = step_weights.sum()
        for action_idx, name in enumerate(["fold", "bet", "call"]):
            mask = action_indices == action_idx
            if mask.any():
                adv_for_action = advantages[mask]
                adv_stats[f"advantage/mean_{name}"] = float(adv_for_action.mean())
                adv_stats[f"advantage/std_{name}"] = float(adv_for_action.std())
                adv_stats[f"weight/value_{name}"] = float(step_weights[mask][0])
                adv_stats[f"weight/eff_frac_{name}"] = float(
                    step_weights[mask].sum() / weight_total
                )

        return (
            reinforce_loss + diversity_penalty,
            mean_probs,
            diversity_penalty,
            disc_loss.item(),
            (continuous_weight * cont_loss).item(),
            adv_stats,
        )

    def _compute_diversity_penalty(
        self, action_probs: torch.Tensor, all_rewards: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Penalise collapse toward a single action at the batch level.
        Fires only when mean action probabilities exit [lo, hi].
        The coefficient is scaled by mean |reward| to stay proportional
        to the reward magnitude throughout training.

        Parameters
        ----------
        action_probs : [N, 3]  probabilities already computed by forward_batch
        all_rewards  : flat array of per-step rewards for scale estimation

        Returns
        -------
        penalty    : scalar tensor
        mean_probs : [3] mean probabilities (used for logging)
        """
        lo = float(self.config.get("diversity_lo", 0.01))
        hi = float(self.config.get("diversity_hi", 0.99))

        reward_scale = max(float(np.mean(np.abs(all_rewards)) + 1e-8), 1.0)
        coef = float(self.config.get("diversity_coef", 0.1)) * reward_scale

        mean_probs = action_probs.mean(dim=0)
        penalty = (torch.relu(mean_probs - hi) + torch.relu(lo - mean_probs)).sum()
        return coef * penalty, mean_probs

    # ------------------------------------------------------------------
    # Optimizer helpers
    # ------------------------------------------------------------------

    def _gradient_step(self, loss: torch.Tensor) -> float:
        """Backprop + gradient clipping + optimizer step.
        Returns the pre-clip gradient norm (clip_grad_norm_ computes it internally)."""
        if not loss.requires_grad:
            print("[main] WARNING: loss has no grad — skipping backward", flush=True)
            return 0.0
        self.optimizer.zero_grad()
        loss.backward()
        max_norm = float(self.config.get("grad_clip_norm", 1.0))
        pre_clip_norm = torch.nn.utils.clip_grad_norm_(
            self.current_model.parameters(), max_norm=max_norm
        ).item()
        self.optimizer.step()
        return pre_clip_norm

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
            for _, action, _ in trajectory:
                counts[action.action_type] += 1
                if action.action_type == PlayerAction.BET and action.observation.bet_range.lower_bound > 0:
                    bet_amounts.append(action.bet_amount)

        total = sum(counts.values()) or 1
        return {
            "action/fold": counts[PlayerAction.FOLD] / total,
            "action/bet": counts[PlayerAction.BET] / total,
            "action/call": counts[PlayerAction.CALL] / total,
            "action/bet_amount": np.mean(bet_amounts) if bet_amounts else 0.0,
            "game/avg_hands": np.mean(traj_lengths),
            "game/min_hands": np.min(traj_lengths),
            "game/max_hands": np.max(traj_lengths),
        }
