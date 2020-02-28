import numpy as np


class BaseRegularizer:

    def __init__(
            self
    ):
        pass


    def pre_optim_update(
            self,
            layer
    ):
        pass

    def post_optim_update(
            self,
            layer
    ):
        pass