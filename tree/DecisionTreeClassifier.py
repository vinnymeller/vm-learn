import numpy as np
import pandas as pd
from core.node import DecisionTreeNode

class DecisionTreeClassifier:

    def __init__(
            self,
            data,
            y_col,
            max_depth
    ):
        self.data = data
        self.y_col = y_col
        self.max_depth = max_depth

        self.x_cols = [x for x in self.data.columns.values if x != y_col]
        self.column_values = {}
        self.column_indicies = {}
        for column in self.x_cols:
            self.column_values[column] = sorted(self.data[column].unique())
            self.column_indicies[column] = {'min': 1, 'max': len(self.column_values[column]) - 1 }

        self.head = DecisionTreeNode(
            depth=0,
            max_depth=self.max_depth,
            data=self.data,
            column_values=self.column_values,
            column_indicies=self.column_indicies,
            y_col=self.y_col)



    def fit(
            self
    ):
        self.head.split()
