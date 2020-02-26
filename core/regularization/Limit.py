import numpy as np
from core.regularization.base import BaseRegularizer

class Limit(BaseRegularizer):
    
    def __init__(
            self,
            weight_limit=1
    ):
        self.weight_limit = weight_limit

    def post_optim_update(
            self,
            layer
    ):
        layer.weights = np.minimum(layer.weights, self.weight_limit)
        layer.weights = np.maximum(layer.weights, -self.weight_limit)