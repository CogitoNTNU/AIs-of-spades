import time

import numpy as np
import torch
import torch.optim as optim
import torch.distributions as D
from nn.poker_net import PokerNet
import wandb
import torch.multiprocessing as mp

from training.game_loop import Game
from pokerenv.common import PlayerAction
from training.weight_manager import WeightManager


def _run_game(args):
    """
    Worker function executed in a subprocess (CPU-only).
    Instantiates a fresh model from the serialized state dict, plays a full
    game, and returns the scalar reward plus the recorded trajectory.
    CUDA must not be used here — mp.Pool workers cannot share a CUDA context.
    """
    model_class, state_dict, hands_per_game, opponent_state_dicts = args

    model = model_class()
    model.initialize_internal_state()
    model.load_state_dict(state_dict)
    model.eval()

    opponents = []
    for osd in opponent_state_dicts:
        opp = model_class()
        opp.initialize_internal_state()
        if osd is not None:
            opp.load_state_dict(osd)
        opp.eval()
        opponents.append(opp)

    with torch.no_grad():
        game = Game(opponents, model)
        reward, trajectory = game.play(hands_per_game)

    return reward, trajectory


class LearningLoop:
    def __init__(self, weight_manager: WeightManager, config: dict):
        self.weight_manager = weight_manager
        self.config = config["learning_loop"]
        self.num_workers = self.config.get("num_workers", mp.cpu_count())

        # Model lives on CPU until start_learning() moves it to the target device
        self.current_model: PokerNet = config["weight_manager"]["model_class"]()
        self.current_model.initialize_internal_state()

        self.optimizer = optim.Adam(
            self.current_model.parameters(),
            lr=self.config.get("learning_rate", 1e-4),
        )

        # Resolved in start_learning(); stored as an attribute so helper methods
        # (_compute_reinforce_loss, etc.) can reference it without extra arguments
        self.device: torch.device | None = None

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Restores model and optimizer state from a checkpoint saved by WeightManager.
        Returns the next epoch to resume from.
        Note: tensors are loaded onto CPU here; the caller is responsible for
        moving the model to the target device afterwards.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.current_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(
            f"Checkpoint loaded from '{checkpoint_path}' — resuming from epoch {start_epoch}"
        )
        return start_epoch

    def start_learning(self, resume_from: str | None = None):
        """
        Main training loop.

        Architecture:
          - CPU worker pool  →  runs game simulations, collects trajectories
          - Main process GPU →  re-runs forward passes, computes REINFORCE loss,
                                performs the gradient step

        Workers receive a CPU-serialized state dict so they never touch CUDA.
        The gradient step happens exclusively in the main process on `device`.
        """
        epochs = self.config.get("epochs", 1000)
        games_per_epoch = self.config.get("games_per_epoch", 10)
        hands_per_game = self.config.get("hands_per_game", 100)
        save_interval = self.config.get("save_interval", 20)

        # Resolve the compute device once and store it for helper methods
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on device: {self.device}")

        # Move model to GPU (or keep on CPU if unavailable)
        self.current_model = self.current_model.to(self.device)

        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
            # load_checkpoint uses map_location="cpu", so we must re-send to device
            self.current_model = self.current_model.to(self.device)

        with mp.Pool(processes=self.num_workers) as pool:
            try:
                for epoch in range(start_epoch, epochs):
                    start_time = time.time()

                    # Serialize weights to CPU before passing to worker processes.
                    # This is required because CUDA tensors cannot be shared across
                    # forked subprocesses via mp.Pool.
                    state_dict = {
                        k: v.cpu() for k, v in self.current_model.state_dict().items()
                    }

                    args = [
                        (
                            self.current_model.__class__,
                            state_dict,
                            hands_per_game,
                            [
                                self.weight_manager.sample_opponent_state_dict()
                                for _ in range(5)
                            ],
                        )
                        for _ in range(games_per_epoch)
                    ]

                    # --- simulation (CPU workers) ---
                    t0_simulation = time.time()
                    results = pool.map(_run_game, args)
                    t_simulation = time.time() - t0_simulation

                    batch_rewards = [r for r, _ in results]
                    batch_trajectories = [t for _, t in results]

                    # Total steps collected this epoch — useful to normalise other timings
                    total_steps = sum(len(t) for t in batch_trajectories)

                    # --- loss computation (GPU forward passes) ---
                    t0_loss = time.time()
                    loss = self._compute_reinforce_loss(
                        batch_trajectories, batch_rewards
                    )
                    t_loss = time.time() - t0_loss

                    # --- gradient step (backward + optimizer) ---
                    t0_grad = time.time()
                    self._gradient_step(loss)
                    t_grad = time.time() - t0_grad

                    # --- logging & stats (CPU) ---
                    t0_stats = time.time()
                    avg_reward = np.mean(batch_rewards)
                    action_stats = self._compute_action_stats(batch_trajectories)
                    t_stats = time.time() - t0_stats

                    end_time = time.time()
                    t_total = end_time - start_time
                    # Unaccounted overhead: weight serialisation, pool arg building, etc.
                    t_overhead = t_total - (t_simulation + t_loss + t_grad + t_stats)

                    wandb.log(
                        data={
                            "epoch": epoch,
                            # --- timing breakdown ---
                            "time/total": t_total,
                            "time/simulation": t_simulation,  # bottleneck if >> t_loss
                            "time/loss_forward": t_loss,  # bottleneck if GPU is underutilised
                            "time/grad_step": t_grad,  # bottleneck if model is large
                            "time/stats_logging": t_stats,  # should be negligible
                            "time/overhead": t_overhead,  # weight serialisation + pool args
                            # Throughput — steps per second helps compare across batch sizes
                            "time/steps_per_sec": (
                                total_steps / t_total if t_total > 0 else 0.0
                            ),
                            # Fraction of each epoch spent in simulation vs GPU work
                            "time/frac_simulation": (
                                t_simulation / t_total if t_total > 0 else 0.0
                            ),
                            "time/frac_gpu": (
                                (t_loss + t_grad) / t_total if t_total > 0 else 0.0
                            ),
                            # --- core training signal ---
                            "train/loss": loss.item() if loss.requires_grad else 0.0,
                            "train/avg_reward": avg_reward,
                            # Reward spread across games — spikes may signal instability
                            "train/reward_std": np.std(batch_rewards),
                            "train/reward_max": np.max(batch_rewards),
                            "train/reward_min": np.min(batch_rewards),
                            # Short trajectories often mean the agent is folding too early
                            "train/avg_trajectory_len": np.mean(
                                [len(t) for t in batch_trajectories]
                            ),
                            "train/total_steps": total_steps,
                            # Gradient norm — verify that clipping is holding
                            "train/grad_norm": self._get_grad_norm(),
                            **action_stats,
                        }
                    )
                    print(
                        f"Epoch {epoch + 1}/{epochs} | "
                        f"loss: {loss.item():.4f} | avg reward: {avg_reward:.4f} | "
                        f"total: {t_total:.2f}s  sim: {t_simulation:.2f}s  "
                        f"fwd: {t_loss:.2f}s  bwd: {t_grad:.2f}s  "
                        f"overhead: {t_overhead:.2f}s | "
                        f"device: {self.device}"
                    )

                    if epoch % save_interval == 0 or epoch == epochs - 1:
                        self.weight_manager.save(
                            self.current_model, self.optimizer, epoch
                        )

            except KeyboardInterrupt:
                print("KeyboardInterrupt caught — terminating worker pool.")
                pool.terminate()
                pool.join()
                raise

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

        Where:
            A_i                — baseline-subtracted advantage for episode i
            log_p_discrete_t   — log π(a_discrete | s_t)  via Categorical.log_prob()
            log_p_continuous_t — log π(bet | s_t)          via Normal.log_prob()

        Observations and action tensors arrive from CPU workers; they are moved
        to self.device here so that the forward pass runs on GPU.
        """
        device = self.device
        step_losses = []

        # Normalize rewards into advantages to reduce variance
        advantages = np.array(batch_rewards, dtype=np.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for trajectory, advantage in zip(batch_trajectories, advantages):
            if not trajectory:
                continue

            reward_tensor = torch.tensor(advantage, dtype=torch.float32, device=device)

            for obs, action in trajectory:
                # Move observation to GPU; guard against non-tensor obs types
                obs_gpu = obs.to(device) if isinstance(obs, torch.Tensor) else obs

                action_logits, bet_mean, bet_std = self.current_model.forward(obs_gpu)

                discrete_dist = D.Categorical(logits=action_logits)
                log_p_discrete = discrete_dist.log_prob(action.action_tensor.to(device))

                continuous_dist = D.Normal(bet_mean, bet_std)
                log_p_continuous = continuous_dist.log_prob(
                    action.bet_tensor.to(device)
                )

                # REINFORCE: -A * log π(a | s)
                step_loss = -reward_tensor * (log_p_discrete + log_p_continuous)
                step_losses.append(step_loss)

        if not step_losses:
            # No steps recorded for the main agent — nothing to optimize this epoch
            return torch.tensor(0.0, requires_grad=False, device=device)

        # Stack before reducing to keep the full computation graph intact
        return torch.stack(step_losses).mean()

    def _gradient_step(self, loss: torch.Tensor):
        """Backprop + gradient clipping + optimizer step."""
        if not loss.requires_grad:
            return  # nothing to optimize this epoch
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradient norm to prevent exploding gradients (common with REINFORCE)
        torch.nn.utils.clip_grad_norm_(self.current_model.parameters(), max_norm=1.0)
        self.optimizer.step()

    def _get_grad_norm(self) -> float:
        """Compute the L2 norm of all gradients after backward()."""
        total_norm = 0.0
        for p in self.current_model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm**0.5

    def _compute_action_stats(self, batch_trajectories: list) -> dict:
        """Aggregate action frequencies and bet sizes across all trajectories."""
        counts = {PlayerAction.FOLD: 0, PlayerAction.BET: 0, PlayerAction.CALL: 0}
        bet_amounts = []
        hands_per_trajectory = [len(t) for t in batch_trajectories]

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
            "game/avg_hands_per_trajectory": np.mean(hands_per_trajectory),
            "game/min_hands_per_trajectory": np.min(hands_per_trajectory),
            "game/max_hands_per_trajectory": np.max(hands_per_trajectory),
        }
