from core.optimizers.base import BaseOptimizer

class Adam(BaseOptimizer):
    """
    Uses parameters minbatch_size, learning_rate, beta_1, beta_2, epsilon (thx brandon)
    https://arxiv.org/abs/1412.6980 (paper)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first_moment = 0
        self.second_moment = 0
        self.timestep = 0

    def update(self, layer):
        self.timestep += 1

        de_dw_batch = self.update_minibatch(layer)
        if de_dw_batch is None:
            return

        self.first_moment = (
            self.adam_beta_1 * self.first_moment
            + (1 - self.adam_beta_1) * de_dw_batch
        )
        self.second_moment = (
            self.adam_beta_2 * self.second_moment
            + (1 - self.adam_beta_2) * de_dw_batch ** 2
        )
        corrected_first_moment = self.first_moment / (
            1 - self.adam_beta_1 ** self.timestep)
        corrected_second_moment = self.second_moment / (
            1 - self.adam_beta_2 ** self.timestep)

        adjustment = self.learning_rate * corrected_first_moment / (
            corrected_second_moment ** .5 + self.epsilon)
        layer.weights -= adjustment