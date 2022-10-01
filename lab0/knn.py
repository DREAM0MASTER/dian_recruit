import numpy as np
from tqdm import tqdm


class Knn(object):

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):

        # TODO Predict the label of X by
        # the k nearest neighbors.

        # Input:
        # X: np.array, shape (n_samples, n_features)

        # Output:
        # y: np.array, shape (n_samples,)

        # Hint:
        # 1. Use self.X and self.y to get the training data.
        # 2. Use self.k to get the number of neighbors.
        # 3. Use np.argsort to find the nearest neighbors.

        # YOUR CODE HERE
        # raise NotImplementedError
        ...
        m=self.X.shape[0]
        n=X.shape[0]
        X=X.reshape(n,28*28)
        self.X=self.X.reshape(m,28*28)
        y_pred = np.zeros(n)
        for i in tqdm(range(n), desc='It\'s a test'):
            distance = np.zeros(m)
            for j in range(self.X.shape[0]):
                distance[j] =np.sqrt(np.dot(X[i]-self.X[j], (X[i]-self.X[j]).T))
            index = np.argsort(distance)
            score = np.zeros(10)
            for k in range(3):
                score[self.y[index[k]]] = score[self.y[index[k]]] + np.exp(distance[index[k]])
            Index = np.argsort(score)
            y_pred[i] = int(Index[-1])
        return y_pred


        # End of todo
