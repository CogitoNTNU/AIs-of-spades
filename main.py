#from src.nn.nn import PokerNet
from src.nn.nn_ines import PokerNet, test_pokernet_forward, test_planner

if __name__ == "__main__":
    model = PokerNet()
    test_pokernet_forward()
    test_planner()