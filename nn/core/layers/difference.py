import numpy as np
from core.layers.base import BaseLayer
class Difference(BaseLayer):

    def __init__(
            self,
            previous_layer,
            subtract_me_layer
    ):
        self.previous_layer = previous_layer
        self.subtract_me_layer = subtract_me_layer
        assert self.subtract_me_layer.y.size == self.previous_layer.y.size
        self.size = self.previous_layer.y.size

    def forward_pass(
            self,
            **kwargs
    ):
        self.y = self.previous_layer.y - self.subtract_me_layer.y

    def backward_pass(
            self
    ):
        self.previous_layer.de_dy += self.de_dy
        self.subtract_me_layer.de_dy -= self.de_dy