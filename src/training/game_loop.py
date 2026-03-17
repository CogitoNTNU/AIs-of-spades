import random as rn

from pokerenv.table import Table
from pokerenv.observation import Observation
from pokerenv.common import PlayerState

from training.player_agent import PlayerAgent
from training.weight_manager import WeightManager

MAIN_CHARACTER_NAME = "UGO"


class Game:

    def __init__(self, opponents: list, current_model):
        self.opponents = opponents  # list of 5 pre-built models
        self.current_model = current_model
        self.table = None
        self.agents = []

        # trajectory: list of (log_p_discrete, log_p_continuous) — both PyTorch tensors
        # Only steps where the main character acted are stored.
        self.trajectory = []
        self.reward = 0.0

    def reset(self):
        self.trajectory = []
        self.reward = 0.0

        self.active_opponents = rn.randint(1, 5)
        player_names = {i: "player_%d" % (i + 1) for i in range(6)}
        player_names[0] = MAIN_CHARACTER_NAME

        self.agents = [PlayerAgent(0, MAIN_CHARACTER_NAME, 0, self.current_model)]
        for n in range(1, self.active_opponents + 1):
            self.agents.append(
                PlayerAgent(n, player_names[n], 0, self.opponents[n - 1])
            )

        self.table = Table(
            self.active_opponents + 1,
            players=self.agents,
            stack_low=50,
            stack_high=200,
            hand_history_location="hands/",
            invalid_action_penalty=0,
        )
        self.table.seed(None)
        self.table.reset()

    def play(self, total_hands: int):
        self.reset()
        if self.table is None:
            raise Exception("Table should not be None")

        for hand_index in range(total_hands):
            for agent in self.agents:
                agent.new_hand()
            obs_array = self.table.reset_hand()
            obs = Observation(
                obs_array, self._get_point_of_view(obs_array[0], self.table.hand_log)
            )

            while True:
                acting_player_i = int(obs.player_identifier)

                if acting_player_i >= len(self.agents):
                    raise Exception(
                        "player_identifier %d is out of range (agents: %d)"
                        % (acting_player_i, len(self.agents))
                    )
                acting_agent = self.agents[acting_player_i]
                if acting_agent.state != PlayerState.ACTIVE or acting_agent.all_in:
                    break

                action = acting_agent.get_action(obs)

                if acting_player_i == 0:
                    self.trajectory.append((obs, action))

                obs_array, rewards, done = self.table.step(action)

                if done:
                    main_reward = rewards[0]
                    if main_reward is not None:
                        self.reward += float(main_reward)
                    if self.agents[0].stack <= 0:
                        return self.reward, self.trajectory
                    break

                obs = Observation(
                    obs_array,
                    self._get_point_of_view(obs_array[0], self.table.hand_log),
                )

        return self.reward, self.trajectory

    def _get_point_of_view(self, player, hand_log):
        if player == 0:
            return hand_log
        hand_log = hand_log.copy()
        mask = hand_log[:, 0] != -1.0
        hand_log[mask, 0] = (hand_log[mask, 0] - player) % (self.active_opponents + 1)
        return hand_log
