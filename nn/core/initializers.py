import numpy as np

class Glorot:
    @staticmethod
    def __str__():
        return 'Glorot'
    @staticmethod
    def initialize(
            n_rows,
            n_cols
    ):
        return np.random.normal(
            scale=np.sqrt(2 / (n_rows + n_cols)),
            size=(n_rows,n_cols)
        )


class He:
    @staticmethod
    def __str__():
        return 'He'
    @staticmethod
    def initialize(
            n_rows,
            n_cols
    ):
        return np.random.uniform(
            low=-np.sqrt(6 / n_rows),
            high=np.sqrt(6 / n_rows),
            size=(n_rows, n_cols)
        )


class Uniform:
    def __str__():
        return 'Uniform distribution on [-{}, {}]'.format(self.scale, self.scale)
    def __init__(
            self,
            scale=0.2
    ):
        self.scale = scale
    def initialize(
            self,
            n_rows,
            n_cols
    ):
        return np.random.uniform(
            low=-self.scale,
            high=self.scale,
            size=(n_rows, n_cols)
        )