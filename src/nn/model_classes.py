from nn.even_net import EvenNet


def list_models():
    return MODEL_CLASSES.keys()


MODEL_CLASSES = {
    "EvenNet": EvenNet,
}
