# -*- encoding:utf-8 -*-
"""
@file: testjieba.py
@author: Sineatos
@time: 2016/8/1 10:27
@contact: sineatos@gmail.com
"""

import jieba

sec = "小明1995年毕业于北京清华大学"

if __name__ == "__main__":
    seg_list = jieba.cut(sec, cut_all=False)
    print("Default Mode:", " ".join(seg_list))

    seg_list = jieba.cut(sec)
    print(" ".join(seg_list))

    seg_list = jieba.cut(sec, cut_all=True)
    print("Full Mode:", "/ ".join(seg_list))

    # 搜索引擎模式
    seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")
    print("/ ".join(seg_list))
