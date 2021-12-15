import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys


class MultiNomialLogisticRegression(object):

    def __init__(self, x, y, y_train, testdata, testY, y_test):
        self.LEARNING_RATE = 0.001  # The learning rate.
        self.CONVERGENCE_MARGIN = 0.001  # The convergence criterion.
        self.MAX_ITERATIONS = []
        # Number of datapoints.
        self.DATAPOINTS = len(x)

        # Number of features.
        self.FEATURES = len(x[0]) + 1
        # Encoding of the data points (as a DATAPOINTS x FEATURES size array).
        self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(x)), axis=1)

        self.y = np.array(y)

        self.testdataDATAPOINTS = len(testdata)
        self.testY = np.array(testY)
        self.testdata = np.concatenate((np.ones((self.testdataDATAPOINTS, 1)), np.array(testdata)), axis=1)

        self.y_train = np.array(y_train)
        self.y_test = np.array(y_test)

        # The weights we want to learn in the training phase.
        self.theta = np.random.uniform(-1, 1, size=(7, self.FEATURES))

        # The current gradient.
        self.gradient = np.zeros((6, self.FEATURES))

        for iteration in self.MAX_ITERATIONS:
            print("Iterations =", str(iteration))
            self.fit(iteration)
            self.compare_results()
            self.confusion()
            self.prescision()
            self.recall()

        # for iteration in self.MAX_ITERATIONS:
        #     f = open("outputs/" + str(iteration) + "_acc_pres_rec" + ".txt", 'w+')
        #     sys.stdout = f
        #     self.fit(iteration)
        #     self.compare_results()
        #     self.confusion()
        #     f.close()

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(z.shape[0], 1)

    def fit(self, iteration):
        cost = []
        self.init_plot(self.FEATURES)
        for i in range(iteration):
            dot = np.dot(self.x, self.theta.T)
            step = self.softmax(dot)

            cost.append(-np.sum(self.y * np.log(step)) / self.DATAPOINTS)

            self.gradient = np.dot((step - self.y).T, self.x)
            delta = (self.LEARNING_RATE / self.DATAPOINTS) * self.gradient
            self.theta = self.theta - delta
            self.update_plot(cost[i])

        # plt.plot(cost)
        # plt.show()

    def compare_results(self):
        probab = self.softmax(np.dot(self.testdata, self.theta.T))
        predict = np.argmax(probab, axis=1)

        from sklearn import linear_model
        from sklearn.metrics import mean_squared_error, explained_variance_score, accuracy_score, precision_score, \
            recall_score

        # train the model with training data
        regr = linear_model.LogisticRegression()
        regr.fit(self.x, self.y_train)

        # Predict our test data
        sklearn_predict = regr.predict(self.testdata)

        print('Sklearn')
        # Accuracy score
        print("Accuracy score: %.2f" % accuracy_score(sklearn_predict, self.y_test))
        # Precision Score
        # print("Precision: %.2f" % precision_score(sklearn_predict, self.y_test, average=None))
        # Recall Score
        # print("Recall Score: %.2f" % recall_score(sklearn_predict, self.y_test, average=None))
        # The mean squared error
        print("Mean squared error: %.2f" % mean_squared_error(sklearn_predict, self.y_test))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % explained_variance_score(self.y_test, sklearn_predict))

        print('\n')

        print('Our Model')
        print("Accuracy score: %.2f" % accuracy_score(predict, self.y_test))
        # Precision Score
        # print("Precision: %.2f" % precision_score(predict, self.y_test, average=None))
        # Recall Score
        # print("Recall Score: %.2f" % recall_score(predict, self.y_test, average=None))
        # The mean squared error
        print("Mean squared error: %.2f" % mean_squared_error(predict, self.y_test))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % explained_variance_score(self.y_test, predict))

    def confusion(self):
        probab = self.softmax(np.dot(self.testdata, self.theta.T))
        predict = np.argmax(probab, axis=1)

        class_dict = {1: "AGE", 2: "TIME", 3: "DATE", 4: "DISTANCE", 5: "AMOUNT", 6: "MONEY"}
        row_labels = col_labels = list(class_dict.values())
        data = np.zeros((self.FEATURES - 2, self.FEATURES - 2))
        df = pd.DataFrame(data, columns=col_labels, index=row_labels)

        for prediction in range(0, len(predict)):
            df[class_dict[predict[prediction]]][class_dict[self.y_test[prediction]]] += 1

        self.confusion_matrix = df
        print(df)

    def prescision(self):
        class_dict = {0: "AGE", 1: "TIME", 2: "DATE", 3: "DISTANCE", 4: "AMOUNT", 5: "MONEY"}
        for i in range(self.confusion_matrix.shape[1]):
            numerator = self.confusion_matrix.iat[i, i]
            denominator = self.confusion_matrix.iloc[i].sum()
            prescision = numerator / denominator
            print("Prescision for " + class_dict[i], prescision)

    def recall(self):
        class_dict = {0: "AGE", 1: "TIME", 2: "DATE", 3: "DISTANCE", 4: "AMOUNT", 5: "MONEY"}
        for i in range(self.confusion_matrix.shape[1]):
            numerator = self.confusion_matrix.iat[i, i]
            denominator = self.confusion_matrix.iloc[:, i].sum()
            recall = numerator / denominator
            print("Recall for " + class_dict[i], recall)

    # ------------------------------ PLOT FUNCTIONS ------------------------------------------------------#

    def init_plot(self, num_axes):
        """
        num_axes is the number of variables that should be plotted.
        """
        self.i = []
        self.val = []
        plt.ion()
        self.axes = plt.gca()
        self.lines = []

        for i in range(num_axes):
            self.val.append([])
            self.lines.append([])
            self.lines[i], = self.axes.plot([], self.val[0], '-', c=[random.random() for _ in range(3)], linewidth=1.5,
                                            markersize=4)

    def update_plot(self, *args):
        """
        Handles the plotting
        """
        if self.i == []:
            self.i = [0]
        else:
            self.i.append(self.i[-1] + 1)

        for index, val in enumerate(args):
            self.val[index].append(val)
            self.lines[index].set_xdata(self.i)
            self.lines[index].set_ydata(self.val[index])

        self.axes.set_xlim(0, max(self.i) * 1.5)
        self.axes.set_ylim(0, max(max(self.val)) * 1.5)

        plt.draw()
        plt.pause(1e-20)
