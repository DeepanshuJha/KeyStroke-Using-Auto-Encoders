import pandas as pd
import numpy

class loader:

    def __init__(self):
        data = pd.read_csv('./dataset.csv', header=None)
        X = data.iloc[:, 2:]
        self.train_x = X.to_numpy()

    def load(self):
        return self.train_x, self.train_x