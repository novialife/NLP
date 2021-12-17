import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from ParseFunctions import *
from ParseXML import ParseXML
import random
import math
from sklearn.metrics import mean_squared_error, explained_variance_score, accuracy_score, precision_score, \
    recall_score


class RegEx:

    def __init__(self):
        self.parse = ParseXML()
        self.y_train = self.parse.y_train
        self.testData = self.parse.testData
        self.y_test = self.parse.y_test
        self.testY = self.parse.testY
        self.class_dict = {0: "AGE", 1: "TIME", 2: "DATE", 3: "DISTANCE", 4: "AMOUNT", 5: "MONEY"}
        self.FEATURES = len(self.testData[0])
        self.classify()
        f = open("outputs/" + "regex.txt", 'w+')
        sys.stdout = f
        self.confusion()
        self.prescision()
        self.recall()
        f.close()


    def classify(self):
        self.predict = []

        for point in self.testData:
            if point[0] == 1:
                point.remove(1)

            count = 0
            indexes = []
            i = 0
            for ones in point:
                if ones == 1:
                    indexes.append(i)
                i += 1

            if len(indexes) == 0:
                self.predict.append(random.randint(0, 5))
            else:
                self.predict.append(random.choice(indexes))
        print(np.unique(self.predict))
    def confusion(self):
        row_labels = col_labels = list(self.class_dict.values())
        data = np.zeros((self.FEATURES - 1, self.FEATURES - 1))
        df = pd.DataFrame(data, columns=col_labels, index=row_labels)

        for prediction in range(0, len(self.predict)):
            df[self.class_dict[self.predict[prediction]]][self.class_dict[self.y_test[prediction]]] += 1

        self.confusion_matrix = df
        print(df)

        print("Accuracy score: %.2f" % accuracy_score(self.predict, self.y_test))
        # The mean squared error
        print("Mean squared error: %.2f" % mean_squared_error(self.predict, self.y_test))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % explained_variance_score(self.y_test, self.predict))

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


def main():
    RegEx()


if __name__ == '__main__':
    main()
