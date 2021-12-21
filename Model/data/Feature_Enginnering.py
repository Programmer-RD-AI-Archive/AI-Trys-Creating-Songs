import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt


class Feature_Enginnering:
    def __init__(self, data_frame):
        self.data_frame = data_frame

    def remove_duplicates(self):
        self.data_frame = self.data_frame.drop_duplicates()
        return self.data_frame

    def encode(self, multiple_columns=False):
        for col_name, col_dtype in zip(
            self.data_frame.dtypes.to_dict().key(), self.data_frame.dtypes.to_dict().values()
        ):
            if col_dtype == type(str):
                if multiple_columns is True:
                    pd.get_dummies(self.data_frame[col_name])
                if multiple_columns is False:
                    dict_name_idx = {}
                    idx = -1
                    for name, number_of_occurrences in zip(
                        self.data_frame[col_name].value_counts().to_dict().keys(),
                        self.data_frame[col_name].value_counts().to_dict().values(),
                    ):
                        idx += 1
                        dict_name_idx[name] = idx
                    self.data_frame[col_name] = self.data_frame[col_name].replace(dict_name_idx)
        return self.data_frame
    
    
