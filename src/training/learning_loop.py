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

        self.current_model: PokerNet = config["weight_manager"]["model_class"]()
        self.current_model.initialize_internal_state()

        self.optimizer = optim.Adam(
            self.current_model.parameters(),
            lr=config.get("learning_rate", 1e-4),
        )

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Restores model and optimizer state from a checkpoint saved by WeightManager.
        Returns the next epoch to resume from.
        """
        checkpoint = torch.load(checkpoint_path)
        self.current_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(
            f"Checkpoint loaded from '{checkpoint_path}' — resuming from epoch {start_epoch}"
        )
        return start_epoch

    def start_learning(self, resume_from: str | None = None):
        epochs = self.config.get("epochs", 1000)
        games_per_epoch = self.config.get("games_per_epoch", 10)
        hands_per_game = self.config.get("hands_per_game", 100)
        save_interval = self.config.get("save_interval", 20)

        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)

        state_dict = {k: v.cpu() for k, v in self.current_model.state_dict().items()}

        args = [
            (
                self.current_model.__class__,
                state_dict,
                hands_per_game,
                [self.weight_manager.sample_opponent_state_dict() for _ in range(5)],
            )
            for _ in range(games_per_epoch)
        ]

        with mp.Pool(processes=self.num_workers) as pool:
            try:
                for epoch in range(start_epoch, epochs):
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

                    results = pool.map(_run_game, args)

                    batch_rewards = [r for r, _ in results]
                    batch_trajectories = [t for _, t in results]

                    loss = self._compute_reinforce_loss(
                        batch_trajectories, batch_rewards
                    )
                    self._gradient_step(loss)

                    avg_reward = np.mean(batch_rewards)
                    action_stats = self._compute_action_stats(batch_trajectories)
                    wandb.log(
                        {
                            "epoch": epoch,
                            # Core training signal
                            "train/loss": loss.item() if loss.requires_grad else 0.0,
                            "train/avg_reward": avg_reward,
                            # Reward distribution across games (helps spot instability)
                            "train/reward_std": np.std(batch_rewards),
                            "train/reward_max": np.max(batch_rewards),
                            "train/reward_min": np.min(batch_rewards),
                            # Trajectory length — if it drops, your agent may be folding too early
                            "train/avg_trajectory_len": np.mean(
                                [len(t) for t in batch_trajectories]
                            ),
                            # Gradient norm — useful to verify clipping is working
                            "train/grad_norm": self._get_grad_norm(),
                            **action_stats,
                        }
                    )
                    print(
                        f"Epoch {epoch + 1}/{epochs} | loss: {loss.item():.4f} | avg reward: {avg_reward:.4f}"
                    )

                    if epoch % save_interval == 0 or epoch == epochs - 1:
                        # WeightManager handles saving, pool management, and pruning
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
        REINFORCE loss over a batch of episodes.

        For each step t in episode i:
            loss += -R_i * (log_p_discrete_t + log_p_continuous_t)

        Where:
            R_i                — scalar reward for episode i
            log_p_discrete_t   — log π(a_discrete | s_t), tensor from Categorical.log_prob()
            log_p_continuous_t — log π(bet | s_t),        tensor from Normal.log_prob()
                                (zero tensor if action was not BET)

        Both log_prob tensors are still attached to the computation graph
        of current_model, so .backward() propagates gradients correctly.
        """
        step_losses = []
        advantages = np.array(batch_rewards) - np.mean(batch_rewards)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for trajectory, advantage in zip(batch_trajectories, advantages):
            if not trajectory:
                continue
            reward_tensor = torch.tensor(advantage, dtype=torch.float32)
            for obs, action in trajectory:
                action_logits, bet_mean, bet_std = self.current_model.forward(obs)

                discrete_dist = D.Categorical(logits=action_logits)
                log_p_discrete = discrete_dist.log_prob(action.action_tensor)

                continuous_dist = D.Normal(bet_mean, bet_std)
                log_p_continuous = continuous_dist.log_prob(action.bet_tensor)

                # REINFORCE: -R * log π(a | s)
                step_loss = -reward_tensor * (log_p_discrete + log_p_continuous)
                step_losses.append(step_loss)

        if not step_losses:
            # No steps taken by the main character — nothing to optimize
            return torch.tensor(0.0, requires_grad=False)

        # Stack into a single tensor before summing — keeps the full graph intact
        total_loss = torch.stack(step_losses).mean()

        return total_loss

    def _gradient_step(self, loss: torch.Tensor):
        if not loss.requires_grad:
            return  # nothing to optimize this epoch
        self.optimizer.zero_grad()
        loss.backward()
        # Clip to avoid exploding gradients (common issue with REINFORCE)
        torch.nn.utils.clip_grad_norm_(self.current_model.parameters(), max_norm=1.0)
        self.optimizer.step()

    def _get_grad_norm(self) -> float:
        """Compute total gradient norm after backward()."""
        total_norm = 0.0
        for p in self.current_model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm**0.5

    def _compute_action_stats(self, batch_trajectories: list) -> dict:
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
