#!/usr/bin/python
"""
Author: Sineatos
Date: 2017-03-07
"""

from DecisionTree.DecisionTree import ID3DTree, C45DTree
from pprint import pprint as pp
from numpy import *
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


def test_ID3Tree():
    dtree = ID3DTree()
    dataset_path = "dataset.dat"
    save_path = "data.tree"
    labels = ['age', 'revenue', 'student', 'credit']
    dtree.load_data_set(dataset_path, labels)
    dtree.train()
    # ID3DTree.store_tree(dtree.tree, save_path)
    # tree = ID3DTree.grab_tree(save_path)
    # pp(tree)
    vector = ['0', '1', '0', '0']  # no
    pp(ID3DTree.predict(dtree.tree, labels, vector))


def test_C45DTree():
    dataset_path = "dataset.dat"
    labels = ['age', 'revenue', 'student', 'credit']
    dtree = C45DTree()
    dtree.load_data_set(dataset_path, labels)
    dtree.train()
    pp(dtree.tree)
    vector = ['2', '1', '0', '0']  # yes
    pp(ID3DTree.predict(dtree.tree, labels, vector))


def plotfigure(X, X_test, y, yp):
    plt.figure()
    plt.scatter(X, y, c="k", label="data")
    plt.plot(X_test, yp, c="r", label="max_depth=5", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Decision Tree Regression")
    plt.legend()
    plt.show()


def test_CART():
    x = np.linspace(-5, 5, 200)
    siny = np.sin(x)
    X = mat(x).T
    y = siny + np.random.rand(1, len(siny)) * 1.5
    y = y.tolist()[0]
    clf = DecisionTreeRegressor(max_depth=4)
    clf.fit(X, y)

    X_test = np.arange(-5.0, 5.0, 0.05)[:, np.newaxis]
    yp = clf.predict(X_test)
    plotfigure(X, X_test, y, yp)

if __name__ == "__main__":
    # test_ID3Tree()
    # test_C45DTree()
    test_CART()
    pass