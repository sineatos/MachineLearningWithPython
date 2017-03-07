#!/usr/bin/python
"""
Author: Sineatos
Date: 2017-03-07
"""
from numpy import *
import math
import copy
import pickle


class ID3DTree:
    def __init__(self):
        self.tree = {}  # 生成的树
        self.data_set = []  # 数据集
        self.labels = []  # 标签集

    def load_data_set(self, path, labels):
        """
        从文件中导入数据，一个样本为一行，一行中每一个属性用tab分隔开，样本的最后一个值为其分类
        :param path: 数据文件路径
        :param labels: 属性标签
        :return:
        """
        fp = open(path, "r")  # 读取文件内容
        content = fp.read()
        fp.close()
        row_list = content.splitlines()
        record_list = [str(row).split('\t') for row in row_list if row.strip()]
        self.data_set = record_list
        self.labels = labels

    def train(self):
        """
        训练决策树
        """
        labels = copy.deepcopy(self.labels)
        self.tree = self.build_tree(self.data_set, labels)

    def build_tree(self, data_set, labels):
        """
        构造决策树
        :param data_set: 数据集，一行为一个样本
        :param labels: 数据集拥有的特征
        :return: 构造完成的决策树
        """
        cate_list = [data[-1] for data in data_set]  # 将数据集中的每一个样本的分类拿出来
        if cate_list.count(cate_list[0]) == len(cate_list):
            # 当前集合中所有样本都为一类，所以结束当前分支的构造，返回分类标识
            return cate_list[0]
        if len(cate_list) == 1:
            # 当前集合只有一个样本,返回这一个样本的类别
            return ID3DTree.max_cate(cate_list)
        best_feat = ID3DTree.get_best_feat(data_set)  # 求出当前集合的最优划分特征的下标
        best_feat_label = labels[best_feat]
        tree = {best_feat_label: {}}
        del labels[best_feat]  # 删除最优划分特征
        unique_vals = set([data[best_feat] for data in data_set])
        for value in unique_vals:
            sub_labels = labels[:]  # 复制当前的特征列表
            split_data_set = ID3DTree.split_data_set(data_set, best_feat, value)
            sub_tree = self.build_tree(split_data_set, sub_labels)
            tree[best_feat_label][value] = sub_tree
        return tree

    @staticmethod
    def max_cate(cate_list):
        """
        返回列表中出现最多的类别标识
        :param cate_list: 类别标识列表
        :return: 列表中出现最多的类别标识
        """
        # 使用出现频数作为字典的key,如果有相同频数的类别则选择其中一个作为value
        items = dict([(cate_list.count(i), i) for i in cate_list])
        return items[max(items.keys())]

    @staticmethod
    def store_tree(input_tree, filename):
        """
        存储树到文件中
        :param input_tree: 树的字符串
        :param filename: 文件路径
        """
        fw = open(filename, 'wb')
        pickle.dump(input_tree, fw)
        fw.close()

    @staticmethod
    def grab_tree(filename):
        """
        从文件抓取树
        :return: 一个字符串
        """
        with open(filename, 'rb') as fr:
            return pickle.load(fr)

    @staticmethod
    def get_best_feat(data_set):
        """
        求出当前集合最好的划分特征
        :param data_set:数据集
        :return: 最好的划分特征的下标
        """
        num_features = len(data_set[0]) - 1  # 获取特征的数量
        base_entropy = ID3DTree.compute_entropy(data_set)  # 计算当前集合的信息熵
        best_info_gain = 0.0
        best_feature = -1
        for i in range(num_features):
            unique_vals = set([data[i] for data in data_set])  # 提取数据集中第i个属性在数据集中出现过的取值
            new_entropy = 0.0
            for value in unique_vals:  # 求根据特征轴划分数据集以后各个子集合的信息熵的加权平均值
                sub_data_set = ID3DTree.split_data_set(data_set, i, value)  # 求特征轴取值为value的数据集
                prob = len(sub_data_set) / float(len(data_set))
                new_entropy += prob * ID3DTree.compute_entropy(sub_data_set)
            info_gain = base_entropy - new_entropy
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = i
        return best_feature

    @staticmethod
    def compute_entropy(data_set):
        """
        计算并返回数据集的加权信息熵
        :param data_set: 数据集
        :return: 数据集的加权信息熵
        """
        data_len = float(len(data_set))  # 数据集中样本的总数
        cate_list = [data[-1] for data in data_set]
        items = dict([(i, cate_list.count(i)) for i in cate_list])  # 将每一个分类值出现的频数转换为一个字典{分类值:频数}
        info_entropy = 0.0
        for key in items:
            prob = float(items[key]) / data_len  # 求出分类在数据集中的出现的频率
            info_entropy -= prob * math.log(prob, 2)  # 求出数据集的加权信息熵
        return info_entropy

    @staticmethod
    def split_data_set(data_set, axis, value):
        """
        返回删除数据集中的特征轴的数据的新数据集，其中新数据集中的数据特征轴的取值为value
        :param data_set: 数据集
        :param axis: 特征轴的下标(指定的属性)
        :param value: 特征轴的取值
        :return: 删除特征轴数据以后的数据集
        """
        rtn_list = []
        for feat_vec in data_set:
            if feat_vec[axis] == value:
                r_feat_vec = feat_vec[:axis]
                r_feat_vec.extend(feat_vec[axis + 1:])
                rtn_list.append(r_feat_vec)
        return rtn_list