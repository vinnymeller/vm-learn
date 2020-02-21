import numpy as np

class BaseLayer:


    def __init__(
        self,
        previous_layer
    ):
        self.previous_layer = previous_layer
        self.size = self.previous_layer.y.size
        self.reset()

    def reset(
        self
    ):
        self.x = np.zeros((1, self.size))
        self.y = np.zeros((1, self.size))
        self.de_dx = np.zeros((1, self.size))
        self.de_dy = np.zeros((1, self.size))

    def forward_pass(
        self,
        **kwargs
    ):
        self.x += self.previous_layer.y
        self.y = self.x

    def backward_pass(
        self
    ):
        self.de_dx = self.de_dy
        self.previous_layer.de_dy += self.de_dx




class Dense(BaseLayer):
    def __init__(
        self,
        n_outputs,
        activation_func=
        previous_layer=None,
        dropout_rate=0
    ):
        self.previous_layer = previous_layer
        self.n_inputs = self.previous_layer.y.size
        self.n_outputs = n_outputs
        self.activation_func = activation_func
        self.dropout_rate = dropout_rate

        self.learning_rate = 0.01

        self.weights = self.initial_weight_scale * (np.random.sample(
            size=(self.n_inputs + 1, self.n_outputs)) * 2 - 1)

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
        self.i_dropout[np.where(np.random.uniform(size=self.x.size) < dropout_rate)] = True
        self.x[:, self.i_dropout] = 0
        self.x[:, np.logical_not(self.i_dropout)] *= 1/(1-dropout_rate)

        bias = np.ones((1,1))
        x_w_bias = np.concaatenate((self.x, bias), axis=1)
        v = x_w_bias @ self.weights

        self.y = self.activation_func.calc(v)

    def back_prop(
        self,
        de_dy
    ):
        bias = np.ones((1,1))
        x_w_bias = np.concatenate((self.x, bias), axis=1)

        dy_dv = self.activation_func.calc_d(self.y)

        dv_dw = self.x_w_bias.transpose()
        dv_dx = self.weights.transpose()

        dy_dw = dv_dw @ dy_dv
        de_dw = de_dy * dy_dw
        self.weights -= de_dw * self.learning_rate
        for regularizer in self.regularizers:
            self.weights = regularizer.update(self)

        self.de_dx = (self.de_dy * dy_dv) @ dv_dx
        de_dx_no_bias = self.de_dx[:, :-1]
        de_dx_no_bias[:, self.i_dropout] = 0

        self.previous_layer.de_dy += de_dx_no_bias