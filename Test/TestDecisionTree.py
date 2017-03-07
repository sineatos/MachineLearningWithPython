#!/usr/bin/python
"""
Author: Sineatos
Date: 2017-03-07
"""

from numpy import *
from DecisionTree.DecisionTree import ID3DTree
from pprint import pprint as pp


def test_ID3Tree():
    dtree = ID3DTree()
    dataset_path = "dataset.dat"
    save_path = "data.tree"
    labels = ['age', 'revenue', 'student', 'credit']
    dtree.load_data_set(dataset_path, labels)
    dtree.train()
    ID3DTree.store_tree(dtree.tree, save_path)
    tree = ID3DTree.grab_tree(save_path)
    pp(tree)


if __name__ == "__main__":
    test_ID3Tree()
