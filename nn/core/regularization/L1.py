import numpy as np
from core.regularization.base import BaseRegularizer

class L1(BaseRegularizer):

    def __init__(
            self,
            regularization_amount=1e-2
    ):
        self.regularization_amount = regularization_amount

    def pre_optim_update(
            self,
            layer
    ):
        layer.de_dw += np.sign(layer.weights) * self.regularization_amount
