import pandas as pd
import numpy as np
import torch
import os

# 自己设置的参数
train_data_path = os.path.dirname(__file__)+"/KDDTrain+_20Percent.txt"
test_data_path = os.path.dirname(__file__)+"/KDDTest+.txt"
# 离散属性所在的列
discrete_arr = [1, 2, 3, 6, 11, 13, 14, 20, 21]
continuous_arr = [0, 4, 5, 7, 8, 9, 10, 12, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                  36, 37, 38, 39, 40]
labels_classify = {'neptune': 'dos', 'warezclient': 'r2l', 'ipsweep': 'probe', 'portsweep': 'probe', 'teardrop': 'dos',
                   'nmap': 'probe', 'satan': 'probe', 'smurf': 'dos', 'pod': 'dos', 'back': 'dos',
                   'guess_passwd': 'r2l',
                   'ftp_write': 'r2l', 'multihop': 'r2l', 'rootkit': 'u2r', 'buffer_overflow': 'u2r', 'imap': 'r2l',
                   'warezmaster': 'r2l', 'phf': 'r2l', 'land': 'dos', 'loadmodule': 'u2r', 'spy': 'r2l',
                   'saint': 'probe',
                   'mscan': 'probe', 'apache2': 'dos', 'snmpgetattack': 'r2l', 'processtable': 'dos',
                   'httptunnel': 'u2r',
                   'ps': 'u2r', 'snmpguess': 'r2l', 'mailbomb': 'dos', 'named': 'r2l', 'sendmail': 'r2l',
                   'xterm': 'u2r',
                   'worm': 'r2l', 'xlock': 'r2l', 'perl': 'u2r', 'xsnoop': 'r2l', 'sqlattack': 'u2r', 'udpstorm': 'dos'}

labels = {"normal": 0, "dos": 1, "probe": 2, "r2l": 3, "u2r": 4}


# 尽量都使用numpy格式，pd在函数封装里面使用

# 输入是一列或一行都行
def get_one_hat(x):
    x = pd.DataFrame(x)
    x = pd.get_dummies(x)
    return x


# 输入整个数据DataFrame格式，和离散的属性列，返回将离散数据转为one-hat的训练集和测试集
def deal_discrete(train_data, test_data, discrete_columns):
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


# 连续类型离散化,输入整个数据DataFrame格式，和连续的属性列，返回将连续数据转为离散属性的训练集和测试集
# def deal_continuous(train_data, test_data, continuous_columns):
#     train_num = train_data.shape[0]  # 训练集个数
#     data = pd.concat([train_data, test_data], ignore_index=True, axis=0)  # 上下拼接
#     new_data = pd.DataFrame()  # 保存处理后的数据
#     for i in range(data.shape[1]):
#         if i not in continuous_columns:
#             new_data = pd.concat([new_data, data[i]], ignore_index=True, axis=1)
#         else:
#             # 进行离散化操作
#             col_one_hat = pd.get_dummies(data[i])
#             new_data = pd.concat([new_data, col_one_hat], ignore_index=True, axis=1)
#             # print("第%d列=>%d个维度"%(i+1,col_one_hat.shape[1]))
#
#     train = new_data.iloc[:train_num, :]
#     test = new_data.iloc[train_num:, :]
#     return train.to_numpy(), test.to_numpy()

def five_classify(Y_train, Y_test):
    train_num = Y_train.shape[0]  # 训练集个数
    Y = pd.concat([Y_train, Y_test], ignore_index=True, axis=0)
    for key, value in labels_classify.items():
        Y[Y == key] = value

    for key, value in labels.items():
        Y[Y == key] = value

    Y = Y.to_numpy().astype(float)
    Y_train, Y_test = Y[:train_num], Y[train_num:]
    Y_train, Y_test = torch.LongTensor(Y_train), torch.LongTensor(Y_test)
    return Y_train, Y_test


def two_classify(Y_train, Y_test):
    train_num = Y_train.shape[0]  # 训练集个数
    Y = pd.concat([Y_train, Y_test], ignore_index=True, axis=0)
    Y[Y != 'normal'] = 1
    Y[Y == "normal"] = 0

    Y = Y.to_numpy().astype(float)
    Y_train, Y_test = Y[:train_num], Y[train_num:]
    Y_train, Y_test = torch.LongTensor(Y_train), torch.LongTensor(Y_test)
    return Y_train, Y_test


def get_NSLKDD(classify=40):  # 40/5/2
    train_data = pd.read_csv(train_data_path, header=None, index_col=None)
    test_data = pd.read_csv(test_data_path, header=None, index_col=None)

    X_train, X_test = deal_discrete(train_data.iloc[:, :41], test_data.iloc[:, :41], discrete_arr)
    X_train = torch.FloatTensor(X_train.reshape((X_train.shape[0], X_train.shape[1], -1)))
    X_test = torch.FloatTensor(X_test.reshape((X_test.shape[0], X_test.shape[1], -1)))

    Y_train, Y_test = train_data[41], test_data[41]

    if classify == 40:
        pass
    elif classify == 5:
        Y_train, Y_test = five_classify(Y_train, Y_test)
    elif classify == 2:
        Y_train, Y_test = two_classify(Y_train, Y_test)

    return X_train, X_test, Y_train, Y_test


def print_data_info():
    train_data = pd.read_csv("KDDTrain+_20Percent.txt", header=None, index_col=None)
    test_data = pd.read_csv("KDDTest+.txt", header=None, index_col=None)
    Y_train, Y_test = train_data[41], test_data[41]
    Y_train, Y_test = five_classify(Y_train, Y_test)
    print("train_data,total:" + str(Y_train.size()[0]))
    for k, v in labels.items():
        num = torch.eq(Y_train, v).sum().cpu().detach().numpy()
        print(k + ":" + str(num) + ":" + str(num / Y_train.size()[0]))
    print("test_data,total:" + str(Y_test.size()[0]))
    for k, v in labels.items():
        num = torch.eq(Y_test, v).sum().cpu().detach().numpy()
        print(k + ":" + str(num) + ":" + str(num / Y_test.size()[0]))


def draw_img():  # 可以进行特征选择
    pass


if __name__ == '__main__':
    train_data = pd.read_csv("./MyKDDTrain.csv",header=None,index_col=None)
    test_data = pd.read_csv("./KDDTest+.txt",header=None,index_col=None)
    X_train, X_test = deal_discrete(train_data.iloc[:, :41], test_data.iloc[:, :41], discrete_arr)

    Y_train, Y_test = train_data[41], test_data[41]
    Y_train, Y_test = five_classify(Y_train, Y_test)

    X_train = pd.DataFrame(X_train)
    X_train.to_csv("tempXTrain.csv",header=False,index=False)

    Y_train = pd.DataFrame(Y_train)
    Y_train.to_csv("tempYTrain.csv",header=False,index=False)

    pass
