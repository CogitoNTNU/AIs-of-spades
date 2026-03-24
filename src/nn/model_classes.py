from nn.even_net import EvenNet
from nn.dani_net import DaniNet


def list_models():
    return MODEL_CLASSES.keys()


MODEL_CLASSES = {
    "EvenNet": EvenNet,
    "DaniNet": DaniNet,
}
