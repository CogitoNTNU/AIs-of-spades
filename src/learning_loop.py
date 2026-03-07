import numpy as np
import pokerenv.obs_indices as indices
from pokerenv.table import Table
from pokerenv.common import PlayerAction, Action, action_list
import torch
import torch.nn as nn
import torch.optim as optim
from config import *
from pokerenv import OpponentPool

class LearningLoop:
    def __init__(self):
        self.optimizer = optim.Adam(
            list(PN.parameters()),
            lr=LEARNING_RATE
        )

        self.opponent_pool = OpponentPool(
            pool_dir=OPPONENT_POOL_DIR,
            max_size=MAX_POOL_SIZE,
            keep_latest=KEEP_LATEST_POOL
        )

    def start_learning(self):
        for epoch in range(EPOCHS):
            batch_trajectories = []
            batch_rewards = []

            for _ in range(BATCH_SIZE):
                # Assumes TrainingEpisode can accept opponent_state
                game = Game(HANDS_PER_GAME)
                reward = game.play()
                batch_rewards.append(reward)
                batch_trajectories.append(game.trajectory)

            reward = self.compute_reward(batch_trajectories, batch_rewards)
            self.gradient_decent(reward)

            if epoch % SAVE_INTERVAL == 0:
                self.save_latest_checkpoint(epoch)

            if epoch % POOL_SAVE_INTERVAL == 0:
                self.opponent_pool.add_snapshot(PN, epoch)

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

    def save_latest_checkpoint(self, epoch, path=CHECKPOINT_PATH):
        torch.save({
            "epoch": epoch,
            "model_state_dict": PN.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)