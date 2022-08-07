import pandas as pd
import numpy as np
import torch
import Data.DataParameters as DP


# 尽量都使用numpy格式，pd在函数封装里面使用
class DataPreprocess(object):
    def __init__(self, s_features=DP.select_features):
        # 离散属性所在的列, 0开头, 正常的列数减一得到0开头的列数
        # 求得选中的特征和离散属性的交集，即选中特征中的离散属性
        self.s_features = s_features
        self.discrete_arr = list(set(self.s_features).intersection(DP.discrete_arr))  # 不对
        self.continuous_arr = list(set(self.s_features).intersection(DP.continuous_arr))
        self.discrete_arr.sort()
        self.continuous_arr.sort()

        # 经过处理后均为tensor类型
        self.X_train = None
        self.Y_train_5 = None
        self.Y_train_2 = None
        self.X_test = None
        self.X_test_5 = None
        self.X_test_2 = None

    def data_init(self):
        train_data = pd.read_csv(DP.train_data_path, header=None, index_col=None)
        test_data = pd.read_csv(DP.test_data_path, header=None, index_col=None)
        Y_train, Y_test = train_data[41], test_data[41]
        # 根据特征选择数组选择数列
        train_data = train_data.iloc[:, self.s_features]
        train_data.columns = range(len(self.s_features))
        test_data = test_data.iloc[:, self.s_features]
        test_data.columns = range(len(self.s_features))
        # 获取特征选择后数组中离散数列所在的索引
        discrete_columns = [self.s_features.index(i) for i in self.discrete_arr]
        X_train, X_test = self.__deal_discrete(train_data, test_data, discrete_columns)
        self.X_train = torch.FloatTensor(X_train.reshape((X_train.shape[0], X_train.shape[1])))
        self.X_test = torch.FloatTensor(X_test.reshape((X_test.shape[0], X_test.shape[1])))

        print("X_train.size():", self.X_train.size())
        print("X_test.size():", self.X_test.size())

        self.Y_train_5, self.Y_test_5 = self.__five_classify(Y_train, Y_test)

        self.Y_train_2, self.Y_test_2 = self.__two_classify(Y_train, Y_test)

    # 输入整个数据DataFrame格式，和离散的属性列，返回将离散数据转为one-hat的训练集和测试集
    def __deal_discrete(self, train_data, test_data, discrete_columns):
        train_num = train_data.shape[0]  # 训练集个数
        data = pd.concat([train_data, test_data], ignore_index=True, axis=0)  # 上下拼接
        new_data = pd.DataFrame()  # 保存处理后的数据
        for i in range(data.shape[1]):
            if i not in discrete_columns:
                new_data = pd.concat([new_data, data[i]], ignore_index=True, axis=1)
            else:
                col_one_hat = pd.get_dummies(data[i])
                new_data = pd.concat([new_data, col_one_hat], ignore_index=True, axis=1)
                # print("第%d列=>%d个维度"%(i+1,col_one_hat.shape[1]))

        train = new_data.iloc[:train_num, :]
        test = new_data.iloc[train_num:, :]
        return train.to_numpy(), test.to_numpy()

    def __five_classify(self, Y_train, Y_test):
        train_num = Y_train.shape[0]  # 训练集个数
        Y = pd.concat([Y_train, Y_test], ignore_index=True, axis=0)
        for key, value in DP.labels_classify.items():
            Y[Y == key] = value

        for key, value in DP.labels.items():
            Y[Y == key] = value

        Y = Y.to_numpy().astype(float)
        Y_train, Y_test = Y[:train_num], Y[train_num:]
        Y_train, Y_test = torch.LongTensor(Y_train), torch.LongTensor(Y_test)
        return Y_train, Y_test

    def __two_classify(self, Y_train, Y_test):
        train_num = Y_train.shape[0]  # 训练集个数
        Y = pd.concat([Y_train, Y_test], ignore_index=True, axis=0)
        Y[Y != 'normal'] = 1
        Y[Y == "normal"] = 0

        Y = Y.to_numpy().astype(float)
        Y_train, Y_test = Y[:train_num], Y[train_num:]
        Y_train, Y_test = torch.LongTensor(Y_train), torch.LongTensor(Y_test)
        return Y_train, Y_test

    def print_data_info(self):
        train_data = pd.read_csv(DP.train_data_path, header=None, index_col=None)
        test_data = pd.read_csv(DP.test_data_path, header=None, index_col=None)
        Y_train, Y_test = train_data[41], test_data[41]
        Y_train, Y_test = self.__five_classify(Y_train, Y_test)
        print("train_data,total:" + str(Y_train.size()[0]))
        printdict = {}
        for k, v in DP.labels.items():
            num = torch.eq(Y_train, v).sum().cpu().detach().numpy()
            # print(k + ":" + str(num) + ":" + str(num / Y_train.size()[0]))
            printdict[k] = int(num)
        print(printdict)
        printdict = {}
        print("test_data,total:" + str(Y_test.size()[0]))
        for k, v in DP.labels.items():
            num = torch.eq(Y_test, v).sum().cpu().detach().numpy()
            # print(k + ":" + str(num) + ":" + str(num / Y_test.size()[0]))
            printdict[k] = int(num)
        print(printdict)