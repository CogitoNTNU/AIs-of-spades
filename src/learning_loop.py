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
    def start_learning():
        for epoch in range(EPOCHS):
            batch_trajectories = []
            batch_rewards = []
            for x in range(BATCH_SIZE):
                episode=TrainingEpisode(HANDS_PER_EPISODE)
                reward = episode.play()
                batch_rewards.append(reward)
                batch_trajectories.append(episode.trajectory)

            reward=self.compute_reward(batch_trajectories, batch_rewards)
            result=self.gradient_decent(reward)
            if epoch % 1000 == 0: 
                self.save_to_file(result,epoch)


    def compute_reward():
        pass

    def save_to_file(result,epoch):
        torch.save(result, f"SUPER_AWESOME_POKER_NEURAL_NETWORK_EPOCH_NUMBER:{epoch}")
