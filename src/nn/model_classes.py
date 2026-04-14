from nn.even_net import EvenNet
from nn.simo_net import SimoNet
from nn.fede_net import FedeNet
from nn.miheer_net import MiheerNet
from nn.miheer_net2 import MiheerNet2


def list_models():
    return MODEL_CLASSES.keys()


MODEL_CLASSES = {
    "EvenNet": EvenNet,
    "SimoNet": SimoNet,
    "FedeNet": FedeNet,
    "MiheerNet": MiheerNet,
    "MiheerNet2": MiheerNet2,
}
