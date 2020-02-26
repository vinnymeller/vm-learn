import numpy as np



class sqr:
    @staticmethod
    def __str__():
        return 'Mean Squared Error'
    @staticmethod
    def calc(x):
        return np.mean(x**2)

    @staticmethod
    def calc_d(x):
        return 2 * x


class abs:
    @staticmethod
    def __str__():
        return 'Mean Absolute Error'

    @staticmethod
    def calc(x):
        return np.mean(np.abs(x))

    @staticmethod
    def calc_d(x):
        return np.sign(x)