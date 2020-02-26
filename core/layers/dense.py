import numpy as np
from core.layers.base import BaseLayer
from core.optimizers.SGD import SGD
from core.initializers import Glorot

class Dense(BaseLayer):
    def __init__(
            self,
            n_outputs,
            activation_func,
            previous_layer=None,
            dropout_rate=0,
            optimizer=None,
            initializer=None
    ):
        self.previous_layer = previous_layer
        self.n_inputs = self.previous_layer.y.size
        self.n_outputs = n_outputs
        self.activation_func = activation_func
        self.dropout_rate = dropout_rate

        if initializer is None:
            self.initializer = Glorot()
        else:
            self.initializer = initializer
        if optimizer is None:
            self.optimizer = SGD(learning_rate=0.001)
        else:
            self.optimizer = optimizer

        self.weights = self.initializer.initialize(self.n_inputs + 1, self.n_outputs)

        self.reset()
        self.regularizers = []


    def add_regularizer(
            self,
            new_regularizer
    ):
        self.regularizers.append(new_regularizer)

    def reset(
            self
    ):
        self.x = np.zeros((1, self.n_inputs))
        self.y = np.zeros((1, self.n_outputs))
        self.de_dx = np.zeros((1, self.n_inputs))
        self.de_dy = np.zeros((1, self.n_outputs))


    def forward_pass(
            self,
            evaluating=False,
            **kwargs
    ):
        if self.previous_layer is not None:
            self.x += self.previous_layer.y

        if evaluating:
            dropout_rate = 0
        else:
            dropout_rate = self.dropout_rate

        self.i_dropout = np.zeros(self.x.size, dtype=bool)
        self.i_dropout[np.where(
            np.random.uniform(size=self.x.size) < dropout_rate)] = True
        self.x[:, self.i_dropout] = 0
        self.x[:, np.logical_not(self.i_dropout)] *= 1 / (1 - dropout_rate)
        #self.x[:, np.logical_not(self.i_dropout)] *= 1/(1- (np.sum(self.i_dropout)/self.x.size)) # brohrer test

        bias = np.ones((1,1))
        x_w_bias = np.concatenate((self.x, bias), axis=1)
        v = x_w_bias @ self.weights

        self.y = self.activation_func.calc(v)

    def backward_pass(
            self
    ):
        bias = np.ones((1,1))
        x_w_bias = np.concatenate((self.x, bias), axis=1)

        dy_dv = self.activation_func.calc_d(self.y)
        # v = self.x @ self.weights
        dv_dw = x_w_bias.transpose()
        dv_dx = self.weights.transpose()

        dy_dw = dv_dw @ dy_dv
        self.de_dw = self.de_dy * dy_dw

        for regularizer in self.regularizers:
            regularizer.pre_optim_update(self)

        self.optimizer.update(self)

        for regularizer in self.regularizers:
            regularizer.post_optim_update(self)

        self.de_dx = (self.de_dy * dy_dv) @ dv_dx
        de_dx_no_bias = self.de_dx[:, :-1]
        de_dx_no_bias[:, self.i_dropout] = 0

        self.previous_layer.de_dy += de_dx_no_bias
