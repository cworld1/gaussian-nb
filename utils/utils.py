# from copy import copy

import numpy as np
from time import time

# from typing import List

import os

os.chdir(os.path.split(os.path.realpath(__file__))[0])
BASE_PATH = os.path.abspath(".")


def load_data(file_name):
    """Read csv file.

    Arguments:
        file_name {str} -- csv file name

    Returns:
        X {ndarray} -- 2d array object with int or float
        y {ndarray} -- 1d array object with int or float
    """

    path = os.path.join(BASE_PATH, "%s.csv" % file_name)
    data = np.loadtxt(path, delimiter=",")
    X, y = data[:, :-1], data[:, -1]
    return X, y


def run_time(fn):
    """Decorator for calculating function runtime.Depending on the length of time,
    seconds, milliseconds, microseconds or nanoseconds are used.

    Arguments:
        fn {function}

    Returns:
        function
    """

    def inner():
        start = time()
        fn()
        ret = time() - start
        if ret < 1e-6:
            unit = "ns"
            ret *= 1e9
        elif ret < 1e-3:
            unit = "us"
            ret *= 1e6
        elif ret < 1:
            unit = "ms"
            ret *= 1e3
        else:
            unit = "s"
        print("Total run time is %.1f %s\n" % (ret, unit))

    return inner
