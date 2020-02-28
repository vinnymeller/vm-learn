from nn.core import BaseRegularizer

class L2(BaseRegularizer):
    def __init__(
            self,
            regularization_amount=1e-2
    ):
        self.regularization_amount = regularization_amount

    def pre_optim_update(
            self,
            layer
    ):
        layer.de_dw += 2 * layer.weights * self.regularization_amount