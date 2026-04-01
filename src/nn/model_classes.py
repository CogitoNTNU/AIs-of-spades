from nn.even_net import EvenNet
from nn.fede_net import FedeNet


def list_models():
    return MODEL_CLASSES.keys()


MODEL_CLASSES = {
    "EvenNet": EvenNet,
    "FedeNet": FedeNet,
}
