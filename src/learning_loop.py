import numpy as np
import pokerenv.obs_indices as indices
from pokerenv.table import Table
from pokerenv.common import PlayerAction, Action, action_list
import torch
import torch.nn as nn
import torch.optim as optim
from config import *

class LearningLoop():
    def __init__(self):
        self.optimizer = optim.Adam(
            list(PN.parameters()),
            lr=LEARNING_RATE
        )
    def start_learning(self):
        for epoch in range(EPOCHS):
            batch_trajectories = []
            batch_rewards = []
            for _ in range(BATCH_SIZE):
                opponent_state = self.opponent_pool.sample_snapshot()
                episode = TrainingEpisode(HANDS_PER_EPISODE, opponent_state=opponent_state)
                reward = episode.play()
                batch_rewards.append(reward)
                batch_trajectories.append(episode.trajectory)

            reward=self.compute_reward(batch_trajectories, batch_rewards)
            result=self.gradient_decent(reward)
            if epoch % 1000 == 0: 
                self.save_latest_checkpoint(result,epoch)


    def compute_reward():
        pass

    def gradient_decent(self, reward):
        pass

    def save_latest_checkpoint(self,result, epoch, path="latest_checkpoint.pt"):
        torch.save({
            "epoch": epoch,
            "model_state_dict": result.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
