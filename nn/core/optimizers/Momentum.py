import numpy as np
from core.optimizers.base import BaseOptimizer

class Momentum(BaseOptimizer):
    """
    Uses minibatch_size, learning_rate, momentum_amount parameters. (thx brandon)
    http://www.cs.toronto.edu/~hinton/absps/naturebp.pdf (paper link)
    """
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(
            **kwargs
        )
        self.previous_adjustment = None

    def update(
            self,
            layer
    ):
        de_dw_batch = self.update_minibatch(layer)
        if de_dw_batch is None:
            return

        if self.previous_adjustment is None:
            self.previous_adjustment = np.zeros(layer.weights.shape)
        new_adjustment = (
            self.previous_adjustment * self.momentum_amount
            + de_dw_batch * self.learning_rate
        )
        layer.weights -= new_adjustment

        # Update previous_adjustment to get set up for the next iteration.
        self.previous_adjustment = new_adjustment