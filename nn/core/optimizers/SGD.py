from core.optimizers.base import BaseOptimizer

class SGD(BaseOptimizer):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

    def update(
            self,
            layer
    ):
        de_dw_batch = self.update_minibatch(layer)
        if de_dw_batch is None:
            return
        layer.weights -= de_dw_batch * self.learning_rate