from pokerenv import Observation
from pokerenv import Actions
from nn import PokerNet

class PlayerAgent:
    def __init__(self, nn:PokerNet):
        # TODO:
        self.internal_state = None

    def get_action(self, observation) -> Actions:
        '''
         Input: an object of Observation class
         Output: an Action
        '''
        # Get observation and feed it to the network
        action_logits, bet_value, internal_state = self.nn.forward(observation, self.internal_state)
        self.internal_state = internal_state
        
        # Sample action according to the distribution given by the network logits
        d = np.random.choice(len(action_logits), p=action_logits)
        return Actions(d, action_logits[d], bet_value)

    def reset(self):
        # TODO:
        self.internal_state = None