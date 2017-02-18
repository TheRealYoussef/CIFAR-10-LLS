import numpy as np
from numpy.linalg import inv

class LLS(object):

    def __init__(self):
        pass

    def train(self, data, labels):
        x = np.ones((data.shape[0], data.shape[1] + 1), dtype=np.float32)
        x[:, :data.shape[1]] = data
        inverse = inv(np.dot(x.T, x))
        self.w = np.empty((10, data.shape[1] + 1))
        for i in range(10):
            t = np.array([1.0 if i == y else 0.0 for y in labels])[:, np.newaxis]
            self.w[i] = np.dot(inverse, np.dot(x.T, t))[:,0]

    def predict(self, data):
        y_pred = np.empty(data.shape[0], dtype=np.uint8)
        x = np.append(data.T, np.ones((1, data.shape[0]), dtype=np.float32), axis=0)
        f = np.dot(self.w, x).T
        for i in range(f.shape[0]):
            scores = f[i]
            y_pred[i] = np.argmax(scores)
        return y_pred
