from torch import nn

from models.DenoiseCNN import DenoiseCNN


class ModelFactory:
    @staticmethod
    def DenoiseCNN() -> nn.Module:
        return DenoiseCNN()
