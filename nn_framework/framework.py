import numpy as np
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')


class ANN:
    def __init__(
        self,
        model=None,
        error_func=None,
        printer=None
    ):

        self.layers = model
        self.error_func = error_func
        self.error_history = []
        self.n_iter_train = int(1e8)
        self.n_iter_eval = int(1e6)
        self.learning_rate = 0.01
        self.printer = printer
        self.viz_interval = 5000
        self.reporting_bin_size = 100
        self.report_min = 3
        self.report_max = 0
        self.expected_input_range = expected_input_range
        self.reports_path = 'reports'
        self.report_name = 'performance_history.png'
        try:
            os.makedirs(reports_path)
        except Exception:
            pass


    def train(self, training_set):
        for i in range(self.n_iter_train):
            x = self.normalize(next(training_set()).ravel())
            y = self.forward_pass(x)

            error = self.error_func.calc(y)
            error_d = self.error_func.calc_d(y)
            self.error_history.append(error)
            self.back_prop(error_d)
            if (i+1) % self.viz_interval == 0:
                print(i)
                self.printer.render(self, x, f'train_{i + 1:08d}')
                self.report()

    def evaluate(self, evaluation_set):
        for i in range(self.n_iter_eval):
            x = self.normalize(next(evaluation_set()).ravel())
            y = self.forward_pass(x, evaluating=True)

            error = self.error_func.calc(y)
            error_d = self.error_func.calc_d(y)
            self.error_history.append(error)

            if i+1 % self.viz_interval == 0:
                self.report()


    def forward_pass(
        self,
        x,
        evaluating=False,
        i_start_layer=None,
        i_stop_layer=None

    ):

        if i_start_layer is None:
            i_start_layer = 0
        if i_stop_layer is None:
            i_stop_leyer = len(self.layers)

        if i_sstart_layer >= i_stop_layer:
            return x


        self.layers[i_start_layer].x += x.ravel()[np.newaxis, :]

        for layer in self.layers[i_start_layer: i_stop_layer]:
            layer.forward_pass(evaluating=evaluating)

        return layer.y.ravel()

    def backward_pass(self, de_dy):
        self.layers[-1].de_dy += de_dy
        for layer in self.layers[::-1]:
            layer.backward_pass()



    def normalize(self, values):
        min_val = self.expected_input_range[0]
        max_val = self.expected_input_range[1]
        values -= min_val
        values /= (max_val - min_val)
        return values - 0.5


    def denormalize(self, normalized_values):
        min_val = self.expected_input_range[0]
        max_val = self.expected_input_range[1]
        normalized_values += 0.5
        normalized_values *= (max_val - min_val)
        return normalized_values + min_val

    def forward_prop_to_layer(self, x, i_layer):
        y = x.ravel()[np.newaxis, :]
        for layer in self.layers[:i_layer]:
            y = layer.forward_prop(y)
        return y.ravel()

    def forward_prop_from_layer(self, x, i_layer):
        y = x.ravel()[np.newaxis, :]
        for layer in self.layers[i_layer:]:
            y = layer.forward_prop(y)
        return y.ravel()


    def report(self):
        n_bins = int(len(self.error_history) // self.reporting_bin_size)
        smoothed_history = []
        for i_bin in range(n_bins):
            smoothed_history.append(np.mean(self.error_history[
                                            i_bin * self.reporting_bin_size:
                                            (i_bin + 1) * self.reporting_bin_size
                                            ]))
        error_history = np.log10(np.array(smoothed_history) + 1e-10)
        ymin = np.minimum(self.report_min, np.min(error_history))
        ymax = np.maximum(self.report_max, np.max(error_history))
        fig = plt.figure()
        ax = plt.gca()
        ax.plot(error_history)
        ax.set_xlabel(f"x{self.reporting_bin_size} iterations")
        ax.set_ylabel("log error")
        ax.set_ylim(ymin, ymax)
        ax.grid()
        fig.savefig(self.report_name)
        plt.close()