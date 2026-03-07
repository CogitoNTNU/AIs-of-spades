from src.weight_manager import WeightManager
from game_loop import Game
import torch
import torch.nn as nn
import torch.optim as optim
from pokerenv import OpponentPool

class LearningLoop:
    def __init__(self, weight_manager: WeightManager, config):
        self.weight_manager = weight_manager
        self.config = config["learning_loop"]

        # initialize the current model (PN) with random weights
        self.current_model = config["weight_manager"]["model_class"]()

        self.optimizer = optim.Adam(
            list(self.current_model.parameters()),
            lr=config.get("learning_rate", 1e-4),
        )

    def start_learning(self):

        for epoch in range(self.config.get("epochs", 1000)):
            batch_trajectories = []
            batch_rewards = []

            for _ in range(self.config.get("games_per_epoch", 10)):
                game = Game(self.weight_manager, self.current_model)
                reward = game.play(self.config.get("hands_per_game", 100))
                batch_rewards.append(reward)
                batch_trajectories.append(game.trajectory)

            reward = self.compute_reward(batch_trajectories, batch_rewards)
            self.gradient_decent(reward)

            if epoch % self.config.get("save_interval", 20) == 0:
                self.save_latest_checkpoint(epoch)

            if epoch % self.config.get("save_every", 20) == 0 or epoch == self.config.get("epochs", 1000) - 1:
                self.weight_manager.save(self.current_model, epoch)

    def compute_reward(self, batch_trajectories, batch_rewards):
        """
        Replace this with your real reward logic.
        """
        return batch_rewards

    def gradient_decent(self, reward):
        """
        Replace this with your real gradient descent logic.
        """
        pass

    def save_latest_checkpoint(self, epoch):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.current_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, self.config.get("latest_checkpoint_path", "latest_checkpoint.pt"))