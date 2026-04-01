from nn.even_net import EvenNet
from nn.simo_net import SimoNet
from nn.fede_net import FedeNet


def list_models():
    return MODEL_CLASSES.keys()


MODEL_CLASSES = {
    "EvenNet": EvenNet,
    "SimoNet": SimoNet,
    "FedeNet": FedeNet,
}
