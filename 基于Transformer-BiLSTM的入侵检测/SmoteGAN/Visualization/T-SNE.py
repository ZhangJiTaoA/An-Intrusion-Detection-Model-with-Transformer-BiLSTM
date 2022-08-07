"""
-*- coding:utf-8 -*-
@Time : 2022/3/15 14:24
@Author : 597778912@qq.com
@File : T-SNE.py
sklearn实现t-SNE
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

font = font_manager.FontProperties(fname='./msyhl.ttc')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决无法显示符号问题

sns.set(palette='muted', color_codes=True)
sns.set(font="SimHei", font_scale=0.8)  # 美化图像，显示中文
sns.set_style('white')
palette = sns.color_palette("bright", 5)


# 对样本进行预处理并画图
def plot_embedding_2d(data, label, title):
    """
    :param data:数据集 shape（多少条数据，每个数据的特征数）
    :param label: 样本标签 shape（多少条数据）
    :param title: 图像标题
    :return: 图像
    """
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    print(x_max)
    print(x_min)
    data = (data - x_min) / (x_max - x_min)  # 归一化
    fig = plt.figure()  # 创建图形实例
    # ax = plt.subplot(111)  # 创建子图
    # 遍历所有样本
    for i in range(data.shape[0]):
        # 在图中为每个数据点画出标签
        # plt.scatter(data[i, 0], data[i, 1], label[i])
        plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i] / 10),
                 fontdict={"weight": "bold", "size": 10}
                 )
    plt.xticks()  # 指定坐标的刻度
    plt.yticks()
    plt.title(title, fontsize=14)

    return fig


def plot_embedding_3d(data, label, title):
    """
    :param data:数据集 shape（多少条数据，每个数据的特征数）
    :param label: 样本标签 shape（多少条数据）
    :param title: 图像标题
    :return: 图像
    """
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)  # 归一化
    fig = plt.figure()  # 创建图形实例
    ax = Axes3D(fig)  # 创建子图

    # 遍历所有样本
    for i in range(data.shape[0]):
        # 在图中为每个数据点画出标签
        ax.text(data[i, 0], data[i, 1], data[i, 2], str(label[i]), color=plt.cm.Set1(label[i] / 10),
                fontdict={"weight": "bold", "size": 10})

    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.title(title, fontsize=14)

    return fig


# 主函数，执行降维操作
def main(data, label):
    # data, label = get_data()
    print("Starting compute t-SNE Embedding...")
    ts = TSNE(n_components=3, init="pca", random_state=0)
    # t-SNE降维
    result = ts.fit_transform(data)
    print("Ending t-SNE, Starting draw img!")
    # 调用函数，绘制图像
    fig = plot_embedding_3d(result, label, "t-SNE Embedding of digits")
    # 显示图像
    plt.show()


if __name__ == '__main__':
    labels = {0: "normal", 1: "dos", 2: "probe", 3: "r2l", 4: "u2r"}
    labels_name = ["normal", "dos", "probe", "r2l", "u2r"]
    data = DPP.DataPreprocess()
    data.data_init()
    data.print_data_info()
    X_train, X_test, Y_train, Y_test = data.X_train,data.X_test,data.Y_train_5,data.Y_test_5
    data = X_train.detach().numpy().squeeze()
    label = Y_train.detach().numpy().squeeze()
    label = [labels.get(i) for i in label]
    # main(data, label)
    tns = TSNE(method="barnes_hut", random_state=0)
    # tns = TSNE(random_state=0)
    print("开始转换！\n")
    X_embedded = tns.fit_transform(data)
    print("开始画图！")
    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=label, hue_order=labels_name, legend='full', palette=palette)
    plt.show()
