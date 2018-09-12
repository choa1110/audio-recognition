from sklearn.externals import joblib
import numpy as np


class Model:

    def __init__(self):
        self.i2c = {}
        self.model = joblib.load("./model/3audiomodel.sav")
        self.X_fromcsv = np.genfromtxt("./data/train_features.csv", delimiter=',')
        self.y_fromcsv = np.genfromtxt("./data/train_labels.csv", delimiter=',')
        self.labels = ['Applause', 'Laughter', 'Scream']
        self.num_class = len(self.labels)
        self.c2i = {}
        self.i2c = {}
        for i, c in enumerate(self.labels):
            self.c2i[c] = i
            self.i2c[i] = c


class Metrics(Model):

    def __init__(self):
        super().__init__()

    @staticmethod
    def proba2labels(preds, i2c, k=3):
        ans = []
        ids = []
        for p in preds:
            idx = np.argsort(p)[::-1]
            ids.append([i for i in idx[:k]])
            ans.append(' '.join([i2c[i] for i in idx[:k]]))

        return ans, ids
