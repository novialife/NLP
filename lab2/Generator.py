import math
import argparse
import codecs
from collections import defaultdict
import numpy as np
import random

"""
This file is part of the computer assignments for the course xx at KTH.
Created 2018 by Johan Boye and Patrik Jonell.
"""


class Generator(object):
    """
    This class generates words from a language model.
    """

    def __init__(self):

        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = {}

        # The bigram log-probabilities.
        self.bigram_prob = defaultdict(dict)

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        # The average log-probability (= the estimation of the entropy) of the test corpus.
        # Important that it is named self.logProb for the --check flag to work
        self.logProb = 0

        # The identifier of the previous word processed in the test corpus. Is -1 if the last word was unknown.
        self.last_index = -1

        # The fraction of the probability mass given to unknown words.
        self.lambda3 = 0.000001

        # The fraction of the probability mass given to unigram probabilities.
        self.lambda2 = 0.01 - self.lambda3

        # The fraction of the probability mass given to bigram probabilities.
        self.lambda1 = 0.99

        # The number of words processed in the test corpus.
        self.test_words_processed = 0

    def read_model(self, filename):
        """
        Reads the contents of the language model file into the appropriate data structures.

        :param filename: The name of the language model file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """
        try:
            with codecs.open(filename, 'r', 'utf-8') as f:
                self.unique_words, self.total_words = map(int, f.readline().strip().split(' '))
                # REUSE YOUR CODE FROM BigramTester.py here
                for row in f:
                    row = row.strip().split(" ")
                    if len(row) == 1:
                        break

                    if row[2].isdigit():
                        self.index[row[1]] = row[0]
                        self.word[row[0]] = row[1]
                        self.unigram_count[row[1]] = row[2]
                    else:
                        self.bigram_prob[self.word[row[0]]][self.word[row[1]]] = row[2]
                        self.logProb += float(row[2])

                self.logProb /= len(self.bigram_prob)
                return True
        except IOError:
            print("Couldn't find bigram probabilities file {}".format(filename))
            return False

    def generate(self, w, n):
        """
        Generates and prints n words, starting with the word w, and following the distribution
        of the language model.
        """
        # YOUR CODE HERE
        finalstr = ""
        curr = w
        for i in range(n):
            curr_dict = self.bigram_prob[curr]
            # print(curr, curr_dict)
            curr_dict_vals = list(curr_dict.values())

            if all(v == '0.000000000000000' for v in curr_dict_vals):
                unique_words = list(self.unigram_count.keys())
                curr = unique_words[random.randint(0, len(unique_words) - 1)]
                print(curr, end=" ")

            else:
                not_log_percents = []
                for y in list(curr_dict_vals):
                    not_log_percents.append(math.e.__pow__(float(y)))

                not_log_percents = np.array(not_log_percents)
                not_log_percents /= not_log_percents.sum()

                word = np.random.choice(list(curr_dict.keys()), 1,
                                        p=not_log_percents)
                curr = word[0]
                print(curr, end=" ")


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BigramTester')
    parser.add_argument('--file', '-f', type=str, required=True, help='file with language model')
    parser.add_argument('--start', '-s', type=str, required=True, help='starting word')
    parser.add_argument('--number_of_words', '-n', type=int, default=100)

    arguments = parser.parse_args()

    generator = Generator()
    generator.read_model(arguments.file)
    generator.generate(arguments.start, arguments.number_of_words)


if __name__ == "__main__":
    main()
