import numpy as np
import pandas as pd



class gini:

    def __init__(
            self
    ):
        pass

    def calc(
            node
    ):
        fractions = node.data[node.y_col].value_counts().values / len(node.data)
        squared = fractions**2
        summed = np.sum(squared)
        return 1 - summed

