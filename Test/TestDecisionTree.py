#!/usr/bin/python
"""
Author: Sineatos
Date: 2017-03-07
"""

from DecisionTree.DecisionTree import ID3DTree, C45DTree
from pprint import pprint as pp


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


if __name__ == "__main__":
    # test_ID3Tree()
    test_C45DTree()