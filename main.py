from src.nn.nn import PokerNet,test_agent, test_pokernet_forward

if __name__ == "__main__":
    test_pokernet_forward()

    modes = ["simple", "lookahead", "mcts"]

    for mode in modes:
        try:
            test_agent(mode=mode)
            print(f"SUCCESS: Mode '{mode}' executed without errors.")
        except Exception as e:
            print(f"FAILED: Mode '{mode}' encountered an error: {e}")
    