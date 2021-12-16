import random
import sys
import xml.etree.ElementTree as ET
import numpy as np
from MultiNomialLogisticRegression import MultiNomialLogisticRegression
from ParseFunctions import *


class ParseXML:

    def __init__(self, sentence=None):
        self.labelsNUMEX = {
            "FRQ": 4,
            "DST": 3,
            "AGE": 0,
            "CUR": 5
        }

        args = sys.argv
        self.train_file = args[1]
        self.test_file = args[2]

        self.x, self.y, self.y_train = self.readXML(self.train_file)
        self.testData, self.testY, self.y_test = self.readXML(self.test_file)

    def evaluate(self, token, label, x, y, y_num):
        token = token.split(" ")
        datapoint = [has_number(token), age_feature(token), date_feature(token),
                     time_feature(token),
                     distance_feature(token),
                     quantity_feature(token), money_feature(token)]
        x.append(datapoint)
        _ = np.zeros(6)
        _[label] = 1
        y.append(_)
        y_num.append(label)
        return x, y, y_num

    def readXML(self, file):
        root = ET.parse(file)
        # Iteration when training

        x = []
        y = []
        y_num = []
        for wordElement in root.iter():  # Iterate through each wordElement
            try:
                token = wordElement.attrib["name"]
                if wordElement.attrib["subtype"] in self.labelsNUMEX:  # If we are in NUMEX
                    x, y, y_num = self.evaluate(token, self.labelsNUMEX[wordElement.attrib["subtype"]], x, y, y_num)
                elif wordElement.attrib["subtype"] == "DAT":  # If we are in TIMEX
                    if date_feature(token.split(" ")) == 0:
                        x.append([has_number(token.split(" ")), 0, 0, 1, 0, 0, 0])
                        index = 2
                    else:
                        x.append([has_number(token.split(" ")), 0, 1, 0, 0, 0, 0])
                        index = 1

                    _ = np.zeros(6)
                    _[index] = 1
                    y.append(_)
                    y_num.append(index)
            except KeyError:  # Only check for expressions
                pass

        return x, y, y_num


def main():
    temp = ParseXML()
    MultiNomialLogisticRegression(temp.x, temp.y, temp.y_train, temp.testData, temp.y_test)


if __name__ == '__main__':
    main()
