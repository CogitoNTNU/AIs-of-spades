import os
from pokerenv.learning_loop import LearningLoop
from pokerenv.weight_manager import WeightManager
from nn.nn import PokerNet
import yaml

# Load the config file
model_classes = {
    "PokerNet": PokerNet,
    # ...
}
with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r") as f:
    config = yaml.safe_load(f)
config["weight_manager"]["model_class"] = model_classes[config["weight_manager"]["model_class"]]

# Initialize the WeightManager
weight_manager = WeightManager(
    config = config["weight_manager"],
)

# Start the learning loop
learning_loop = LearningLoop(weight_manager, config)
learning_loop.start_learning()