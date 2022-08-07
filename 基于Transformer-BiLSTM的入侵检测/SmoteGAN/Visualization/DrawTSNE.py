"""
-*- coding:utf-8 -*-
@Time : 2022/6/24 19:14
@Author : 597778912@qq.com
@File : DrawTSNE.py
"""
import numpy as np
import Data.DataPreprocess as DPP
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.manifold import TSNE
from Data.DataPreprocess import *
from joblib import dump, load
import seaborn as sns
from matplotlib import font_manager
import os
import time


class DrawTSNE(object):
    def __init__(self):
        self.font = font_manager.FontProperties(fname='./msyhl.ttc')
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决无法显示符号问题

        sns.set(palette='muted', color_codes=True)
        sns.set(font="SimHei", font_scale=0.8)  # 美化图像，显示中文
        sns.set_style('white')
        self.palette = sns.color_palette("bright", 5)
        self.labels = {0: "normal", 1: "dos", 2: "probe", 3: "r2l", 4: "u2r"}
        self.labels_name = ["normal", "dos", "probe", "r2l", "u2r"]

        # 输入X，Y为numpy类型

    def draw_TSNE_2d(self, X, Y, isSave=True, saveName="img"):
        data = X.squeeze()
        label = Y.squeeze()
        label = [self.labels.get(i) for i in label]
        tns = TSNE(method="barnes_hut", random_state=0)
        # tns = TSNE(random_state=0)
        print("开始转换！\n")
        X_embedded = tns.fit_transform(data)
        print("开始画图！")
        sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=label, hue_order=self.labels_name, legend='full',
                        palette=self.palette)
        if isSave:
            plt.savefig(os.path.dirname(__file__) + "/img" + saveName + str(int(time.time())), dpi=600,
                        bbox_inches='tight')

        plt.show()
