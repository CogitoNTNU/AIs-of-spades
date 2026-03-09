# learning_loop.py
import numpy as np
import torch
import torch.optim as optim

from pokerenv.weight_manager import WeightManager
from pokerenv.game_loop import Game


class LearningLoop:
    def __init__(self, weight_manager: WeightManager, config: dict):
        self.weight_manager = weight_manager
        self.config = config["learning_loop"]

        self.current_model = config["weight_manager"]["model_class"]()
        self.current_model.initialize_internal_state()

        self.optimizer = optim.Adam(
            self.current_model.parameters(),
            lr=config.get("learning_rate", 1e-4),
        )

    def start_learning(self):
        epochs = self.config.get("epochs", 1000)
        games_per_epoch = self.config.get("games_per_epoch", 10)
        hands_per_game = self.config.get("hands_per_game", 100)

        for epoch in range(epochs):
            batch_trajectories = []
            batch_rewards = []

            for _ in range(games_per_epoch):
                game = Game(self.weight_manager, self.current_model)
                reward, trajectory = game.play(hands_per_game)
                batch_rewards.append(reward)
                batch_trajectories.append(trajectory)

            loss = self._compute_reinforce_loss(batch_trajectories, batch_rewards)
            self._gradient_step(loss)

            avg_reward = np.mean(batch_rewards)
            print(
                f"Epoch {epoch + 1}/{epochs} | loss: {loss.item():.4f} | avg reward: {avg_reward:.4f}"
            )

            if epoch % self.config.get("save_interval", 20) == 0 or epoch == epochs - 1:
                self._save_checkpoint(epoch)
                self.weight_manager.save(self.current_model, epoch)

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
            R_i               — scalar reward for episode i
            log_p_discrete_t  — log π(a_discrete | s_t), tensor from Categorical.log_prob()
            log_p_continuous_t — log π(bet | s_t),       tensor from Normal.log_prob()
                                 (zero tensor if action was not BET)

        Both log_prob tensors are still attached to the computation graph
        of current_model, so .backward() propagates gradients correctly.
        """
        step_losses = []

        for trajectory, reward in zip(batch_trajectories, batch_rewards):
            if not trajectory:
                continue

            reward_tensor = torch.tensor(reward, dtype=torch.float32)

            for log_p_discrete, log_p_continuous in trajectory:
                # REINFORCE: -R * log π(a | s)
                step_loss = -reward_tensor * (log_p_discrete + log_p_continuous)
                step_losses.append(step_loss)

        if not step_losses:
            # No steps taken by the main character — nothing to optimize
            return torch.tensor(0.0, requires_grad=False)

        # Stack into a single tensor before summing — keeps the full graph intact
        total_loss = torch.stack(step_losses).sum()

        # Normalize by batch size to keep loss scale stable across config changes
        total_loss = total_loss / len(batch_trajectories)

        return total_loss

    def _gradient_step(self, loss: torch.Tensor):
        if not loss.requires_grad:
            return  # nothing to optimize this epoch
        self.optimizer.zero_grad()
        loss.backward()
        # Clip to avoid exploding gradients (common issue with REINFORCE)
        torch.nn.utils.clip_grad_norm_(self.current_model.parameters(), max_norm=1.0)
        self.optimizer.step()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.current_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            self.config.get("latest_checkpoint_path", "latest_checkpoint.pt"),
        )
