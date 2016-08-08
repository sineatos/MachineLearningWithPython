# -*- encoding:utf-8 -*-
"""
@file: __init__.py.py
@author: Sineatos
@time: 2016/7/31 16:40
@contact: sineatos@gmail.com
"""

import sys
import os
import jieba
import cPickle as pickle
from sklearn.datasets.base import Bunch
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer  # TF-IDF向量转换类
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF向量生成类

reload(sys)
sys.setdefaultencoding('utf-8')

# 全局变量
train_word_bag = "train_word_bag"
train_set_file = "train_set.dat"
if not os.path.exists(train_word_bag) or not os.path.isdir(train_word_bag):
    os.makedirs(train_word_bag)
wordbag_path = os.path.join(train_word_bag, train_set_file)  # 分词预料Bunch对象持久化文件路径
# Bunch类提供一种key,value的对象形式
# target_name:所有分类集名称列表
# label:每个文件的分类标签列表
# filenames:文件路径
# contents:分词后文件词向量形式
bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])

corpus_path = "train_corpus_small"  # 未分词分类语料库路径
seg_path = "train_corpus_seg"  # 分词后分类语料库路径


###########################################################################



# 将字符串保存到文件里面
def savefile(savepath, content):
    fp = open(savepath, "wb")
    fp.write(content)
    fp.close()


# 从文件中读取字符串
def readfile(path):
    fp = open(path, "rb")
    content = fp.read()
    fp.close()
    return content


def cut_words():
    """
    将未分词分类语料库里面的所有文本进行分词处理
    """
    catelist = os.listdir(corpus_path)  # 获取corpus_path下的所有子目录
    for mydir in catelist:
        class_path = os.path.join(corpus_path, mydir)
        seg_dir = os.path.join(seg_path, mydir)
        if not os.path.exists(seg_dir) or not os.path.isdir(seg_dir):
            os.makedirs(seg_dir)
        file_list = os.listdir(class_path)
        for file_path in file_list:
            fullname = os.path.join(class_path, file_path)
            content = readfile(fullname).strip()
            content = content.replace("\r\n", "").strip()  # 删除换行和多余的空格
            content_seg = jieba.cut(content)
            savefile(os.path.join(seg_dir, file_path), " ".join(content_seg))
    print "中文分词结束!!"


def save_into_bunch():
    catelist = os.listdir(seg_path)
    bunch.target_name.extend(catelist)
    for mydir in catelist:
        class_path = os.path.join(seg_path, mydir)
        file_list = os.listdir(class_path)
        for file_path in file_list:
            fullname = os.path.join(class_path, file_path)
            bunch.label.append(mydir)  # 保存当前文件的分类标签
            bunch.filenames.append(fullname)  # 保存当前文件的文件路径
            bunch.contents.append(readfile(fullname).strip())

    file_obj = open(wordbag_path, "wb")
    pickle.dump(bunch, file_obj)
    file_obj.close()
    print "构建文本对象结束！！！"


# 读取Bunch对象
def read_bunch_obj(path):
    file_obj = open(path, "rb")
    bunch_obj = pickle.load(file_obj)
    file_obj.close()
    return bunch_obj


# 写入Bunch对象
def write_bunch_obj(path, bunch_obj):
    file_obj = open(path, "wb")
    pickle.dump(bunch_obj, file_obj)
    file_obj.close()


if __name__ == "__main__":
    # cut_words()
    save_into_bunch()
    pass
