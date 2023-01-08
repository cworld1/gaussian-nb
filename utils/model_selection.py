from itertools import chain
import numpy as np
from numpy.random import choice, seed
from random import random


def train_test_split(data, label=None, prob=0.7, random_state=None):
    """Split data, label into train set and test set.

    Arguments:
        data {ndarray} -- Training data.

    Keyword Arguments:
        label {ndarray} -- Target values.
        prob {float} -- Train data expected rate between 0 and 1.
        (default: {0.7})
        random_state {int} -- Random seed. (default: {None})

    Returns:
        data_train {ndarray}
        data_test {ndarray}
        label_train {ndarray}
        y_test {ndarray}
    """

    # Set random state.
    if random_state is not None:
        seed(random_state)

    # Split data
    n_rows, _ = data.shape
    k = int(n_rows * prob)
    train_indexes = choice(range(n_rows), size=k, replace=False)
    test_indexes = np.array([i for i in range(n_rows) if i not in train_indexes])
    data_train = data[train_indexes]
    data_test = data[test_indexes]

    # Split label.
    if label is not None:
        label_train = label[train_indexes]
        label_test = label[test_indexes]
        ret = (data_train, data_test, label_train, label_test)
    else:
        ret = (data_train, data_test)

    # Cancel random state.
    if random_state is not None:
        seed(None)

    return ret


def train_test_split_list(X, y, prob=0.7, random_state=None):
    """Split X, y into train set and test set.

    Arguments:
        X {list} -- 2d list object with int or float.
        y {list} -- 1d list object with int or float.

    Keyword Arguments:
        prob {float} -- Train data expected rate between 0 and 1.
        (default: {0.7})
        random_state {int} -- Random seed. (default: {None})

    Returns:
        X_train {list} -- 2d list object with int or float.
        X_test {list} -- 2d list object with int or float.
        y_train {list} -- 1d list object with int 0 or 1.
        y_test {list} -- 1d list object with int 0 or 1.
    """

    if random_state is not None:
        seed(random_state)
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for i in range(len(X)):
        if random() < prob:
            X_train.append(X[i])
            y_train.append(y[i])
        else:
            X_test.append(X[i])
            y_test.append(y[i])
    # Make the fixed random_state random again
    seed()
    return X_train, X_test, y_train, y_test


def _clf_input_check(y, y_hat):
    m = len(y)
    n = len(y_hat)
    elements = chain(y, y_hat)
    valid_elements = {0, 1}
    assert m == n, "Lengths of two arrays do not match!"
    assert m != 0, "Empty array!"
    assert all(
        element in valid_elements for element in elements
    ), "Array values have to be 0 or 1!"


def _get_acc(y, y_hat):
    """Calculate the prediction accuracy.

    Arguments:
        y {ndarray} -- 1d array object with int.
        y_hat {ndarray} -- 1d array object with int.

    Returns:
        float
    """

    _clf_input_check(y, y_hat)
    return (y == y_hat).sum() / len(y)
