import numpy as np
import pandas as pd
from core.criterion import gini

class DecisionTreeNode:

    def __init__(
            self,
            depth,
            max_depth,
            data,
            column_values,
            column_indicies,
            y_col,
    ):
        self.depth = depth
        self.max_depth = max_depth
        self.data = data
        self.column_values = column_values
        self.column_indicies = column_indicies.copy()
        self.y_col = y_col
        self.left = None
        self.right = None
        self.split_col = None
        self.split_val = None

    def split(
            self
    ):
        if self.depth > self.max_depth:
            return

        fractions = self.data[self.y_col].value_counts().values / len(self.data)
        squared = fractions**2
        summed = np.sum(squared)
        lowest_impurity = 1 - summed

        for column, values in self.column_values.items():
            for i in range(self.column_indicies[column]['min'], self.column_indicies[column]['max'] + 1):
                left_ind = np.where(self.data[column].values < values[i])
                right_ind = np.where(self.data[column].values >= values[i])

                left_frac1 = np.sum(self.data[self.y_col].values[left_ind])/len(left_ind[0])
                left_frac2 = 1-left_frac1
                left_gini = (1 - (left_frac1**2 + left_frac2**2)) * (len(left_ind[0])/len(self.data))

                right_frac1 = np.sum(self.data[self.y_col].values[right_ind])/len(right_ind[0])
                right_frac2 = 1-right_frac1
                right_gini = (1 - (right_frac1**2 + right_frac2**2)) * (len(right_ind[0])/len(self.data))

                gini = left_gini + right_gini

                if gini < lowest_impurity:
                    lowest_impurity = gini
                    self.split_col = column
                    self.split_val = i


        if self.split_val is not None:
            left_indicies = self.column_indicies.copy()
            left_indicies[self.split_col]['max'] = self.split_val - 1
            self.left = DecisionTreeNode(
                depth=self.depth+1,
                max_depth=self.max_depth,
                data=self.data.loc[self.data[self.split_col] < self.column_values[self.split_col][self.split_val]],
                column_values=self.column_values,
                column_indicies=left_indicies,
                y_col=self.y_col
            )
            self.left.split()
            right_indicies = self.column_indicies.copy()
            right_indicies[self.split_col]['min'] = self.split_val + 1
            self.right = DecisionTreeNode(
                depth=self.depth+1,
                max_depth=self.max_depth,
                data=self.data.loc[self.data[self.split_col] >= self.column_values[self.split_col][self.split_val]],
                column_values=self.column_values,
                column_indicies=right_indicies,
                y_col=self.y_col
            )
            self.right.split()

        # for column in self.x_cols:
        #     unique_col_values = self.data[column].unique()
        #     for val in unique_col_values:
        #         left = self.data.loc[self.data[column] < val]
        #         right = self.data.loc[self.data[column] >= val]
        #         temp_left_node = DecisionTreeNode(
        #             depth=self.depth+1,
        #             max_depth=self.max_depth,
        #             data=left,
        #             x_cols=self.x_cols,
        #             y_col=self.y_col
        #         )
        #         left_impurity = gini.calc(temp_left_node)
        #         temp_right_node = DecisionTreeNode(
        #             depth=self.depth + 1,
        #             max_depth=self.max_depth,
        #             data=right,
        #             x_cols=self.x_cols,
        #             y_col=self.y_col
        #         )
        #         right_impurity = gini.calc(temp_right_node)
        #         temp_impurity = (left_impurity * len(left) / len(self.data)) + (right_impurity * len(right) / len(self.data))
        #         if temp_impurity < low_score:
        #             low_score = temp_impurity
        #             self.left = temp_left_node
        #             self.right = temp_right_node
        #             self.split_col = column
        #             self.split_val = val
        #
        #
        # if self.left is not None:
        #     self.left.split()
        # if self.right is not None:
        #     self.right.split()
