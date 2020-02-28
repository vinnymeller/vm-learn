import numpy as np


class tanh:

    @staticmethod
    def calc(input):
        return np.tanh(input)

    @staticmethod
    def calc_d(input):
        return 1 - np.tanh(input)**2

class logistic:

    @staticmethod
    def calc(input):
        return 1 / (1+np.exp(-input))

    @staticmethod
    def calc_d(input):
        return calc(input) * (1-calc(input))

class relu:

    @staticmethod
    def calc(input):
        return maximum(0, input)

    def calc_d(input):
        if input > 0:
            return 1
        else:
            return 0