"""
-*- coding:utf-8 -*-
@Time : 2022/6/26 9:42
@Author : 597778912@qq.com
@File : PreteatData.py
本脚本的目的是将KDDTrain+_20Percent.txt中的全部数据以及KDDTrain+.txt中的少数类数据进行合并以初步扩充少数类数据
KDDTrain+_20Percent.txt：KDDTrain+中前20%的记录
{'normal': 13449, 'dos': 9234, 'probe': 2289, 'r2l': 209, 'u2r': 11}
1. 首先读取KDDTrain+_20Percent.txt中的数据并将其中的r2l,u2r,probe删除
2. 读取KDDTrain+。txt中的数据仅保留r2l和u2r标签数据。
3. 将1和2获得的数据进行合并生成新的数据文件
"""
import pandas as pd
import DataParameters as DP


def main():
    trainData1 = pd.read_csv("./KDDTrain+_20Percent.txt", header=None, index_col=None)
    for key, value in DP.labels_classify.items():  # 将大类是r2l和u2r的进行替换
        if value == 'r2l' or value == 'u2r' or value == 'probe':
            trainData1.iloc[trainData1[41] == key, 41] = value
    print(trainData1.shape)
    trainData1 = trainData1[trainData1[41] != 'r2l']
    print(trainData1.shape)
    trainData1 = trainData1[trainData1[41] != 'u2r']
    print(trainData1.shape)
    trainData1 = trainData1[trainData1[41] != 'probe']
    print(trainData1.shape)

    trainData2 = pd.read_csv("./KDDTrain+.txt", header=None, index_col=None)
    for key, value in DP.labels_classify.items():  # 将大类是r2l和u2r的进行替换
        if value != 'r2l' and value != 'u2r' and value != 'probe':
            print(value)
            trainData2.iloc[trainData2[41] == key, 41] = value
    print(trainData2.shape)
    trainData2 = trainData2[trainData2[41] != "normal"]
    print(trainData2.shape)
    trainData2 = trainData2[trainData2[41] != "dos"]
    print(trainData2.shape)
    trainData = pd.concat([trainData1, trainData2], ignore_index=True, axis=0)  # 上下拼接
    trainData.to_csv("MyKDDTrain.csv", header=False, index=False)


if __name__ == '__main__':
    main()
