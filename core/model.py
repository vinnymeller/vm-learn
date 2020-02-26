import numpy as np
import os
import matplotlib.pyplot as plt
import datetime as dt
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
        self.printer = printer
        self.viz_interval = 500
        self.reporting_bin_size = 100
        self.report_min = -3
        self.report_max = 0
        time_dir = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.reports_path = os.path.join("reports", time_dir)
        self.performance_report_name = "performance_history.png"
        self.parameter_report_name = "model_parameters.txt"

        # Ensure that subdirectories exist.
        try:
            os.mkdir("reports")
        except Exception:
            pass
        try:
            os.mkdir(self.reports_path)
        except Exception:
            pass

        self.report_parameters()

    def __str__(self):
        str_parts = [
            "artificial neural network",
            "number of training iterations: " + str(self.n_iter_train),
            "number of evaluation iterations: " + str(self.n_iter_eval),
            "error_function:" + self.error_func.__str__()
        ]
        for i_layer, layer in enumerate(self.layers):
            str_parts.append(
                f"layer {i_layer}:" + layer.__str__()
            )
        return "\n".join(str_parts)

    def train(self, training_set):
        for i in range(self.n_iter_train):
            print(i)
            x = next(training_set()).ravel()
            print(x)
            y = self.forward_pass(x)
            print(y)

            error = self.error_func.calc(y)
            error_d = self.error_func.calc_d(y)
            self.error_history.append(error)
            self.backward_pass(error_d)
            if (i+1) % self.viz_interval == 0:
                self.printer.render(self, x, f'train_{i + 1:08d}')
                self.report()

    def evaluate(self, evaluation_set):
        for i in range(self.n_iter_eval):
            x = next(evaluation_set()).ravel()
            y = self.forward_pass(x, evaluating=True)

            error = self.error_func.calc(y)
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
            i_stop_layer = len(self.layers)

        if i_start_layer >= i_stop_layer:
            return x


        for layer in self.layers:
            layer.reset()

        self.layers[i_start_layer].x += x.ravel()[np.newaxis, :]

        for layer in self.layers[i_start_layer: i_stop_layer]:
            layer.forward_pass(evaluating=evaluating)

        return layer.y.ravel()

    def backward_pass(self, de_dy):
        self.layers[-1].de_dy += de_dy
        for layer in self.layers[::-1]:
            layer.backward_pass()


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
        fig.savefig(self.performance_report_name)
        plt.close()


    def report_parameters(self):
        """
        Create a human-readable summary of the model's parameters.
        """
        param_info = "type: " + self.__str__()
        with open(
            os.path.join(self.reports_path, self.parameter_report_name), "w"
        ) as param_file:
            param_file.write(param_info)
