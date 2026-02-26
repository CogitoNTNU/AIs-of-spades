import numpy as np
import pokerenv.obs_indices as indices
from pokerenv.table import Table
from pokerenv.common import PlayerAction, Action, action_list


class ExampleRandomAgent:
    def __init__(self):
        self.actions = []
        self.observations = []
        self.rewards = []

    def get_action(self, observation):
        self.observations.append(observation)
        valid_actions = np.argwhere(observation[indices.VALID_ACTIONS] == 1).flatten()
        valid_bet_low = observation[indices.VALID_BET_LOW]
        valid_bet_high = observation[indices.VALID_BET_HIGH]
        chosen_action = PlayerAction(np.random.choice(valid_actions))
        bet_size = 0
        if chosen_action is PlayerAction.BET:
            bet_size = np.random.uniform(valid_bet_low, valid_bet_high)
        table_action = Action(chosen_action, bet_size)
        self.actions.append(table_action)
        return table_action

    def reset(self):
        self.actions = []
        self.observations = []
        self.rewards = []


active_players = 6
agents = [ExampleRandomAgent() for _ in range(6)]
player_names = {
    0: "TrackedAgent1",
    1: "Agent2",
}  # Rest are defaulted to player3, player4...
# Should we only log the 0th players (here TrackedAgent1) private cards to hand history files
track_single_player = True
# Bounds for randomizing player stack sizes in reset()
low_stack_bbs = 50
high_stack_bbs = 200
hand_history_location = "hands/"
invalid_action_penalty = 0
table = Table(
    active_players,
    player_names=player_names,
    track_single_player=track_single_player,
    stack_low=low_stack_bbs,
    stack_high=high_stack_bbs,
    hand_history_location=hand_history_location,
    invalid_action_penalty=invalid_action_penalty,
)
table.seed(1)

iteration = 0
while True:
    if iteration % 50 == 0:
        table.hand_history_enabled = True
    active_players = np.random.randint(2, 7)
    table.n_players = active_players
    obs = table.reset()
    for agent in agents:
        agent.reset()
    acting_player = int(obs[indices.ACTING_PLAYER])
    while True:
        action = agents[acting_player].get_action(obs)
        obs, reward, done, _ = table.step(action)
        print()
        print(obs)
        print(reward)
        if done:
            # Distribute final rewards
            for i in range(active_players):
                agents[i].rewards.append(reward[i])
            break
        else:
            # This step can be skipped unless invalid action penalty is enabled,
            # since we only get a reward when the pot is distributed, and the done flag is set
            agents[acting_player].rewards.append(reward[acting_player])
            acting_player = int(obs[indices.ACTING_PLAYER])
    iteration += 1
    table.hand_history_enabled = False
    if iteration > 100:
        break
