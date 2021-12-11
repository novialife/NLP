import random
import sys
import xml.etree.ElementTree as ET
import numpy as np
from MultiNomialLogisticRegression import MultiNomialLogisticRegression

class ParseXML:

    def __init__(self):
        args = sys.argv
        train_file = args[1]
        test_file = args[2]

        self.labelsNUMEX = {
            "FRQ": 4,
            "DST": 3,
            "AGE": 0,
            "CUR": 5
        }

        self.labelsTIMEX = {
            "DATE": 1,
            "TIME": 2
        }

        self.x = []
        self.y = []
        self.testdata = []

        #self.readTraining(train_file)
        self.readTest(test_file)

    def has_number(self, token):
        for elem in token:
            try:
                if elem in ["en", "ett", "två", "tre", "fyra", "fem", "sex", "sju", "åtta", "nio", "tio"]:
                    return 1
                float(elem)
                return 1
            except ValueError:
                pass
        return 0

    def age_feature(self, token):
        if any(x in token for x in ["gammal", "ung", "år", "månader", "gamla", "årsåldern", "åldern"]):
            return 1
        else:
            return 0

    def distance_feature(self, token):
        if any(x in token for x in ["km", "m", "meter", "kilometer", "mil", "cm", "mm", "centimeter"]):
            return 1
        else:
            return 0

    def date_feature(self, token):
        if any(x in ["-", "_", "/", "talet", "januari", "februari", "mars", "april", "maj", "juni", "juli", "augusti",
                     "september", "oktober", "november", "december", "månad", "år"] for x in token):
            return 1
        else:
            return 0

    def time_feature(self, token):
        if any(x in [",", ":", "timme", "minuter", "timmar", "minuter"] for x in token):
            return 1
        else:
            return 0

    def quantity_feature(self, token):
        if any(x in token for x in ["gång", "gånger", "gången", "stycken", "par", "per"]):
            return 1
        else:
            return 0

    def money_feature(self, token):
        if any(x in token for x in
               ["Kronor", "kronor", "tusen", "spänn", "miljoner", "miljarder", "SEK", "sek", "kr", "Kr", "öre", "rubel",
                "dollar", "euro"]):
            return 1
        else:
            return 0

    def evaluate(self, token, label):
        token = token.split(" ")
        datapoint = [self.has_number(token), self.age_feature(token), self.date_feature(token), self.time_feature(token),
                     self.distance_feature(token),
                     self.quantity_feature(token), self.money_feature(token)]
        self.x.append(datapoint)
        _ = np.zeros(6)
        _[label] = 1
        self.y.append(_)

    def readTraining(self, training_file):
        root = ET.parse(training_file)
        # Iteration when training
        for wordElement in root.iter():  # Iterate through each wordElement
            try:
                token = wordElement.attrib["name"]
                if wordElement.attrib["subtype"] in self.labelsNUMEX:  # If we are in NUMEX
                    self.evaluate(token, self.labelsNUMEX[wordElement.attrib["subtype"]])
                elif wordElement.attrib["subtype"] == "DAT":  # If we are in TIMEX
                    if self.date_feature(token.split(" ")) == 0:
                        self.x.append([self.has_number(token.split(" ")), 0, 0, 1, 0, 0, 0])
                        index = 3
                    else:
                        self.x.append([self.has_number(token.split(" ")), 0, 1, 0, 0, 0, 0])
                        index = 2
                    _ = np.zeros(6)
                    _[index] = 1
                    self.y.append(_)
            except KeyError:  # Only check for expressions
                pass



    def readTest(self, test_file):
        root = ET.parse(test_file)
        for wordElement in root.iter("ex"):  # Iterate through each wordElement
            print(wordElement.text)
            quit()




def main():
    temp = ParseXML()
    MultiNomialLogisticRegression(temp.x, temp.y)


if __name__ == '__main__':
    main()
