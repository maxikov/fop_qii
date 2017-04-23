#standard library
import time
from collections import defaultdict

#numpy library
import numpy as np

class HashTableRegression(object):
    def __init__(self):
        pass

    def train(self, data):
        """
        data: RDD of LabeledPoint
        """
        self.table = defaultdict(list)
        for lp in data.collect():
            self.table[tuple(lp.features)].append(lp.label)
        self.table = {k: np.median(v) for k, v in self.table.items()}
        all_median = np.mean(self.table.values())
        self.table = defaultdict(lambda: all_median, self.table)
        return self

    def predict(self, data):
        """
        data: RDD of feature lists
        """
        res = data.map(lambda x: self.table[tuple(x)])
        return res

def train(data):
    return HashTableRegression().train(data)
