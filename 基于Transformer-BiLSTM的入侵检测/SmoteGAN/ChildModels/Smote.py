"""
-*- coding:utf-8 -*-
@Time : 2022/3/1 11:09
@Author : 597778912@qq.com
@File : Smote.py
"""

import random

import pandas as pd
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import OneSidedSelection
from sklearn.datasets import make_classification  # 用来生成不平衡数据
from collections import Counter  # 查看样本分布
import matplotlib.pyplot as plt
import numpy as np
import math
import ChildModels.Parameters as Parameters
import Data.DataParameters as DP
import torch
import os


class OSSborderlinesmote(object):
    # 固定0是多数类，1-4是少数类
    def __init__(self):
        self.oss_ss = Parameters.oss_sampling_strategy  # 构造OSS所需的sampling_strategy
        self.blsmo_ss = Parameters.smote_sampling_strategy  # 构造border line smote所需的sampling_strategy
        self.x = None
        self.y = None

    def fit(self, x, y):
        self.x = x
        self.y = y
        # 对sampling_strategy进行一次拆分
        oss = OneSidedSelection(sampling_strategy=self.oss_ss,
                                random_state=Parameters.random_state)  # OSS无法控制欠采样的数量，sampling_strategy是一个需要欠采样的标签的数组
        self.x, self.y = oss.fit_resample(x, y)
        print("经过OSS之后的数据：")
        self.__print_xy_info()

        blsmo = BorderlineSMOTE(sampling_strategy=self.blsmo_ss, random_state=Parameters.random_state)
        self.x, self.y = blsmo.fit_resample(self.x, self.y)
        print("经过blsmo之后的数据：")
        self.__print_xy_info()
        return self.x, self.y

    def __print_xy_info(self):
        print("train_data,total:" + str(self.y.shape[0]))
        printdict = {}
        for k, v in DP.labels.items():
            num = np.equal(self.y, v).sum()
            # print(k + ":" + str(num) + ":" + str(num / self.y.size()[0]))
            printdict[k] = num
        print(printdict)

    def save_xy(self, x_file_name, y_file_name):
        x = pd.DataFrame(self.x)
        y = pd.DataFrame(self.y)
        save_path = os.path.dirname(__file__)
        x.to_csv(save_path + "/temp_save/" + x_file_name, header=False, index=False)
        y.to_csv(save_path + "/temp_save/" + y_file_name, header=False, index=False)

# x, y = make_classification(n_classes=3, class_sep=2,
#                            weights=[0.5, 0.3, 0.2], n_informative=2,
#                            n_redundant=0, flip_y=0,
#                            n_features=2, n_clusters_per_class=1,
#                            n_samples=30, random_state=10)
# print("样本分布情况:")
# print(x.shape, y.shape)
# print(Counter(y))
#
# smo = BorderlineSMOTE(sampling_strategy={2: 10, 1: 10, 0: 15})  # 使用ratio(ratio不用了,变成sampling_strategy)参数指定对应类别要生成的数量
# x_smo, y_smo = smo.fit_resample(x, y)  # 默认生成1：1
#
# plt.scatter(x[:, 0], x[:, 1], c=y)
# plt.show()
# plt.scatter(x_smo[:, 0], x_smo[:, 1], c=y_smo)
# plt.show()
# print("smote之后的样本分布情况：\n")
# print(Counter(y_smo))


# 自己写SMOTE算法
# class Smote(object):
#     def __init__(self, N=50, k=5, r=2, seed=None):
#         # r=2，算法采用欧式距离
#         # r=1，曼哈顿距离
#         # r=其他值，明斯基距离
#         # 使用smote算法合成样本数量占原样本数量的百分比N%
#         self.N = N
#         # 最近邻个数k,和近邻算法的距离决定因子r
#         self.k = k
#         self.r = r
#         # newindex用于记录smote算法已经合成的样本个数
#         self.newindex = 0
#
#         # seed用于设置随机数种子
#         self.seed = seed
#         self.T = None
#         self.samples = None
#         self.numattrs = None
#         self.synthetic = None
#
#     # 构建训练函数
#     def fit(self, samples):
#         # 初始化self.samples, self.T, self.numattrs
#         self.samples = samples
#         # self.T是少数类样本个数，self.numattrs是少数类样本的特征个数
#
#         self.T, self.numattrs = self.samples.shape
#
#         # 查看T是否不大于近邻数k
#         if self.T <= self.k:
#             # 若是, k更新为T-1
#             self.k = self.T - 1
#
#         # N表示一共应生成多少个新的样本,math.ceil表示向上取整例如：4.1为5
#         N = math.ceil(self.N * self.T / 100)
#         # 创建保存合成样本的数组
#         self.synthetic = np.zeros((N, self.numattrs))
#
#         # 调用并设置k近邻函数
#         neighbors = NearestNeighbors(n_neighbors=self.k + 1,
#                                      algorithm="ball_tree",
#                                      p=self.r).fit(self.samples)
#
#         # 利用并行方法提高效率
#         nnarray = neighbors.kneighbors(self.samples.reshape((self.T, -1)), return_distance=False)
#         # nnarray[0][0]表示samples中第0个数据的第0个最近邻的索引，
#         # nnarray[i][j]表示samples中第i个数据的第j个最近邻的索引
#         self.__populate(N, nnarray)
#
#         return self.synthetic
#
#     def __populate(self, N, nnarray):
#         if self.seed:
#             np.random.seed(self.seed)
#         n = int(N / self.T)  # 要生成的新的样本是原始样本的n倍
#         d = int(N % self.T)  # 要生成新的样本是原始样本的n倍余d个
#         for i in range(n):
#             # 使用矩阵运算提高效率
#             nn = np.random.randint(1, self.k, self.T)  # 生成一个随机数列，其值在1~k之间，用于选择每个点的最近邻索引
#             indexs = nnarray[range(self.T), nn]  # 通过随机数列nn对每个点选择的随机索引号
#             arr = self.samples[indexs]
#             diff = arr - self.samples  # 计算差值
#             gaps = np.random.uniform(0, 1, self.T).reshape((self.T, -1))  # 生成T个随机数，在0~1之间
#             self.synthetic[self.newindex:self.newindex + self.T, :] = self.samples + diff * gaps
#             self.newindex += self.T
#         if d != 0:
#             rd = np.random.randint(0, self.T, d)  # 生成一个随机数列，其值在0~T之间，用于随机选择点
#             nnarr = nnarray[rd]
#             samples = self.samples[nnarr[:, 0]]  # 将选中的样本点提取出来
#             nn = np.random.randint(1, self.k, d)  # 生成一个随机数列，其值在1~k之间，用于选择每个点的最近邻索引
#             indexs = nnarr[range(d), nn]  # 通过随机数列nn对每个点选择的随机索引号
#             arr = self.samples[indexs]
#             diff = arr - samples  # 计算差值
#             gaps = np.random.uniform(0, 1, d).reshape((d, -1))  # 生成T个随机数，在0~1之间
#             self.synthetic[self.newindex:self.newindex + d, :] = samples + diff * gaps
#             self.newindex += d
