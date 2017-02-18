import numpy as np
from numpy.linalg import inv

class LLS(object):

    def __init__(self):
        pass

    def train(self, data, labels):
        # data is an n*d np array (n is the number of data points, d is the number of features)
        # labels is an np array of size n containing the corresponding labels (integers from 0 (inclusive) to c (exclusive))
        # where c is the number of classes
        n = data.shape[0]
        d = data.shape[1]
        c = len(set(labels))
        x = np.ones((n, d + 1), dtype=np.float32)
        x[:, :d] = data
        inverse = inv(np.dot(x.T, x))
        self.w = np.empty((c, d + 1))
        for i in range(c):
            t = np.array([1.0 if i == y else 0.0 for y in labels])[:, np.newaxis]
            self.w[i] = np.dot(inverse, np.dot(x.T, t))[:,0]

    def predict(self, data):
        # data is an n*d np array (n is the number of data points, d is the number of features)
        n = data.shape[0]
        y_pred = np.empty(n, dtype=np.uint8)
        x = np.append(data.T, np.ones((1, n), dtype=np.float32), axis=0)
        f = np.dot(self.w, x).T
        for i in range(f.shape[0]):
            scores = f[i]
            y_pred[i] = np.argmax(scores)
        return y_pred
