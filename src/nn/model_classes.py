from nn.even_net import EvenNet
from nn.simo_net import SimoNet


def list_models():
    return MODEL_CLASSES.keys()


MODEL_CLASSES = {
    "EvenNet": EvenNet,
    "SimoNet": SimoNet,
}
