#from src.nn.nn import PokerNet
from src.nn.nn_lookahead import PokerNet, test_pokernet_forward, test_planner
from src.nn.nn_lookahead_MCTS import PokerNet, test_mcts

if __name__ == "__main__":
    model = PokerNet()
    test_pokernet_forward()
    test_planner()
    test_mcts()