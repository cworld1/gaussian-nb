# -*- coding: utf-8 -*-
"""
@Author: CWorld
@Date: 2023-01-08 17:29:45
@Last Modified by:   CWorld
@Last Modified time: 2023-01-08 18:50:45
"""
import os

os.chdir(os.path.split(os.path.realpath(__file__))[0])

import sys

sys.path.append(os.path.abspath("."))

from utils.gaussian_nb import GaussianNB, load_data
from utils.model_selection import train_test_split, _get_acc
from utils.utils import run_time


@run_time
def main():
    """Tesing the performance of Gaussian NaiveBayes."""

    print("Tesing the performance of Gaussian NaiveBayes...")
    # Load data
    data, label = load_data("breast_cancer")
    # Split data randomly, train set rate 70%
    data_train, data_test, label_train, label_test = train_test_split(
        data, label, random_state=100
    )
    # Train model
    clf = GaussianNB()
    clf.fit(data_train, label_train)
    # Model evaluation
    y_hat = clf.predict(data_test)
    acc = _get_acc(label_test, y_hat)
    print("Accuracy is %.3f" % acc)


if __name__ == "__main__":
    main()
