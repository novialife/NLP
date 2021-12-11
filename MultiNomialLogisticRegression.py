import numpy as np
import matplotlib.pyplot as plt
import random


class MultiNomialLogisticRegression(object):

    def __init__(self, x, y):
        self.LEARNING_RATE = 0.001  # The learning rate.
        self.CONVERGENCE_MARGIN = 0.001  # The convergence criterion.
        self.MAX_ITERATIONS = 5000
        # Number of datapoints.
        self.DATAPOINTS = len(x)

        # Number of features.
        self.FEATURES = len(x[0]) + 1

        # Encoding of the data points (as a DATAPOINTS x FEATURES size array).
        self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(x)), axis=1)

        # Correct labels for the datapoints.
        self.y = np.array(y)

        # The weights we want to learn in the training phase.
        self.theta = np.random.uniform(-1, 1, size=(6,self.FEATURES))

        # The current gradient.
        self.gradient = np.zeros((6,self.FEATURES))


        print(np.shape(self.x), "shape of x")
        print(np.shape(self.y), "shape of y")
        print(np.shape(self.theta), "shape of theta")
        print(np.shape(self.gradient), "shape of gradient")

        self.fit()
        #self.classify_datapoints(self.x, self.y)

    def loss(self, x, y):
        total = 0
        for i in range(self.DATAPOINTS):
            for k in range(self.FEATURES):
                if k == y[i]:
                    total += -np.log(self.softmax(np.multiply(self.theta[:][k], self.x[i][:])))

        return total/self.DATAPOINTS

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(z.shape[0], 1)

    def conditional_prob(self, label, datapoint):
        prob = self.sigmoid(np.dot(self.x[datapoint][:], self.theta))
        if label == 1:
            return prob
        else:
            return 1 - prob

    def fit(self):
        cost = []
        i = 0
        for i in range(self.MAX_ITERATIONS):
            dot = np.dot(self.x, self.theta.T)
            step = self.softmax(dot)

            cost.append(-np.sum(self.y * np.log(step)) / self.DATAPOINTS)

            self.gradient = np.dot((step - self.y).T, self.x)
            delta = (self.LEARNING_RATE/self.DATAPOINTS) * self.gradient
            self.theta = self.theta - delta
            i = i+1

        plt.plot(cost)
        plt.show()

    def classify_datapoints(self, test_data, test_labels):
        pass