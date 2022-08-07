import Data.DataPreprocess as DPP
import ChildModels.Parameters as Par
import Data.DataParameters as DP
from imblearn.under_sampling import OneSidedSelection
import numpy as np
from Visualization.DrawTSNE import DrawTSNE
from ChildModels.Smote import OSSborderlinesmote
import pandas as pd
from ChildModels.GAN import Gan
from ChildModels.WGAN import WGan
import torch
import torch.utils.data
import torch.optim
from Config import *
from ChildModels.Parameters import *
from ComparionModel.all import *
from analyse_result import analyseResult


def main1():
    data = DPP.DataPreprocess()
    data.data_init()
    data.print_data_info()

    X_train = data.X_train.detach().numpy()
    Y_train = data.Y_train_5.detach().numpy()
    drawTSNE = DrawTSNE()

    # drawTSNE.draw_TSNE_2d(X_train, Y_train, isSave=False)

    OSSblsmo = OSSborderlinesmote()
    x, y = OSSblsmo.fit(data.X_train, data.Y_train_5)

    # drawTSNE.draw_TSNE_2d(x, y, isSave=False)
    OSSblsmo.save_xy("Xtrain.csv", "Ytrain.csv")


def main2():
    X = pd.read_csv("./ChildModels/temp_save/Xtrain.csv", index_col=None, header=None)
    Y = pd.read_csv("./ChildModels/temp_save/Ytrain.csv", index_col=None, header=None)
    X_Y = pd.concat([X, Y], ignore_index=True, axis=1)
    X_probe = X_Y[X_Y[95] == 2].iloc[:, 0:95]
    print(X_probe.shape)
    X_r2l = X_Y[X_Y[95] == 3].iloc[:, 0:95]
    print(X_r2l.shape)
    X_u2r = X_Y[X_Y[95] == 4].iloc[:, 0:95]
    print(X_u2r.shape)

    probGAN = Gan(X_probe.to_numpy(), name_n=2)
    probGAN.train()
    probGAN.fit(4000)
    r2lGAN = Gan(X_r2l.to_numpy(), name_n=3)
    r2lGAN.train()
    r2lGAN.fit(3000)
    u2rGAN = Gan(X_u2r.to_numpy(), name_n=4)
    u2rGAN.train()
    u2rGAN.fit(2000)


def main22():
    X = pd.read_csv("./ChildModels/temp_save/Xtrain.csv", index_col=None, header=None)
    Y = pd.read_csv("./ChildModels/temp_save/Ytrain.csv", index_col=None, header=None)
    X_Y = pd.concat([X, Y], ignore_index=True, axis=1)
    X_probe = X_Y[X_Y[95] == 2].iloc[:, 0:95]
    print(X_probe.shape)
    X_r2l = X_Y[X_Y[95] == 3].iloc[:, 0:95]
    print(X_r2l.shape)
    X_u2r = X_Y[X_Y[95] == 4].iloc[:, 0:95]
    print(X_u2r.shape)

    probGAN = WGan(X_probe.to_numpy(), name_n=2)
    probGAN.train()
    probGAN.fit(4000)
    r2lGAN = WGan(X_r2l.to_numpy(), name_n=3)
    r2lGAN.train()
    r2lGAN.fit(3000)
    u2rGAN = WGan(X_u2r.to_numpy(), name_n=4)
    u2rGAN.train()
    u2rGAN.fit(2000)


def main3():
    X = pd.read_csv("./ChildModels/temp_save/Xtrain.csv", index_col=None, header=None)
    Y = pd.read_csv("./ChildModels/temp_save/Ytrain.csv", index_col=None, header=None)
    prob_X = pd.read_csv("./ChildModels/temp_save/Sam_2.csv", index_col=None, header=None)
    prob_label = pd.DataFrame(np.zeros(prob_X.shape[0]) + 2)
    X = pd.concat([X, prob_X], ignore_index=True, axis=0)
    Y = pd.concat([Y, prob_label], ignore_index=True, axis=0)

    r2l_X = pd.read_csv("./ChildModels/temp_save/Sam_3.csv", index_col=None, header=None)
    r2l_label = pd.DataFrame(np.zeros(r2l_X.shape[0]) + 3)
    X = pd.concat([X, r2l_X], ignore_index=True, axis=0)
    Y = pd.concat([Y, r2l_label], ignore_index=True, axis=0)

    u2r_X = pd.read_csv("./ChildModels/temp_save/Sam_4.csv", index_col=None, header=None)
    u2r_label = pd.DataFrame(np.zeros(u2r_X.shape[0]) + 4)
    X = pd.concat([X, u2r_X], ignore_index=True, axis=0)
    Y = pd.concat([Y, u2r_label], ignore_index=True, axis=0)

    X.to_csv("./ChildModels/temp_save/Xtrain_gen.csv", index=False, header=False)
    Y.to_csv("./ChildModels/temp_save/Ytrain_gen.csv", index=False, header=False)
    # drawTSNE = DrawTSNE()
    # drawTSNE.draw_TSNE_2d(X, Y, isSave=False)


def get_1_dataLoader():  # 获取原始数据的dataloader
    data = DPP.DataPreprocess()
    data.data_init()
    data.print_data_info()
    X_train = data.X_train.unsqueeze(dim=-1)
    Y_train = data.Y_train_5
    X_test = data.X_test.unsqueeze(dim=-1)
    Y_test = data.Y_test_5
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=MODEL_BATCH_SIZE,
        shuffle=True
    )

    test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=MODEL_BATCH_SIZE,
        shuffle=False
    )
    print("数据装载完成")
    print("数据准备完成：------------------------------------------------------------")
    return train_loader, test_loader, (X_train, X_test, Y_train, Y_test)


def get_2_dataLoader():  # 获取数据平衡后的数据的dataloader
    X_train = pd.read_csv("./ChildModels/temp_save/Xtrain_gen.csv", header=None, index_col=None)
    Y_train = pd.read_csv("./ChildModels/temp_save/Ytrain_gen.csv", header=None, index_col=None)
    X_train = torch.FloatTensor(X_train.to_numpy()).unsqueeze(dim=-1)
    Y_train = torch.LongTensor(Y_train.to_numpy().squeeze())

    printdict = {}
    print("train_data,total:" + str(Y_train.size()[0]))
    for k, v in DP.labels.items():
        num = torch.eq(Y_train, v).sum().cpu().detach().numpy()
        # print(k + ":" + str(num) + ":" + str(num / Y_test.size()[0]))
        printdict[k] = int(num)
    print(printdict)

    data = DPP.DataPreprocess()
    data.data_init()
    # data.print_data_info()
    X_test = data.X_test.unsqueeze(dim=-1)
    Y_test = data.Y_test_5
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=MODEL_BATCH_SIZE,
        shuffle=True
    )

    test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=MODEL_BATCH_SIZE,
        shuffle=False
    )
    print("数据装载完成")
    print("数据准备完成：------------------------------------------------------------")
    return train_loader, test_loader, (X_train, X_test, Y_train, Y_test)


# 训练过程中进行测试，由于共用一个测试集，所以可以共用
def test_report(test_loader, model, loss_func):
    test_loss = 0
    num_correct = 0
    pred_arr = []
    for step, (batch_x, batch_y) in enumerate(test_loader):
        batch_x, batch_y = batch_x.to(Device), batch_y.to(Device)
        model.eval()
        prediction = model(batch_x)
        loss = loss_func(prediction, batch_y)

        test_loss += loss.cpu().detach().numpy()
        pred = prediction.argmax(dim=1)
        pred_arr.extend(pred.cpu().detach().numpy().tolist())
        num_correct += torch.eq(pred, batch_y).sum().cpu().detach().numpy()
    test_loss, test_acc = test_loss / len(test_loader), num_correct / len(test_loader.dataset)
    return test_loss, test_acc, pred_arr


def train_BiLSTM_1():  # 根据原始数据进行训练
    train_loader, test_loader, (_, _, _, Y_test) = get_1_dataLoader()

    model = BiLSTM().to(Device)
    optim = torch.optim.Adam(model.parameters(), lr=MODEL_LR)
    loss_func = torch.nn.CrossEntropyLoss().to(Device)
    train_loss = 0
    num_correct = 0
    for epoch in range(MODEL_EPOCH):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(Device), batch_y.to(Device)
            model.train()
            prediction = model(batch_x)
            loss = loss_func(prediction, batch_y)

            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss += loss.cpu().detach().numpy()
            pred = prediction.argmax(dim=1)
            num_correct += torch.eq(pred, batch_y).sum().cpu().detach().numpy()
        train_loss, train_acc = train_loss / len(train_loader), num_correct / len(train_loader.dataset)
        test_loss, test_acc, test_pred = test_report(test_loader, model, loss_func)
        print(
            "Epoch:%d => train_loss:%.10f => train_acc:%.10f%% => test_loss: %.10f => test_acc: %.10f%%=>lr: %.10f" % (
                epoch + 1, train_loss, train_acc * 100, test_loss, test_acc * 100,
                optim.state_dict()['param_groups'][0]['lr']), end=":::::")
        analyseResult(Y_test, test_pred, "BiLSTM_1").print_analyse_result()
        train_loss, num_correct = 0, 0


def train_BiLSTM_2():  # 根据处理过数据平衡问题的数据进行训练
    train_loader, test_loader, (_, _, _, Y_test) = get_2_dataLoader()

    model = BiLSTM().to(Device)
    optim = torch.optim.Adam(model.parameters(), lr=MODEL_LR)
    loss_func = torch.nn.CrossEntropyLoss().to(Device)
    train_loss = 0
    num_correct = 0
    for epoch in range(MODEL_EPOCH):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(Device), batch_y.to(Device)
            model.train()
            prediction = model(batch_x)
            loss = loss_func(prediction, batch_y)

            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss += loss.cpu().detach().numpy()
            pred = prediction.argmax(dim=1)
            num_correct += torch.eq(pred, batch_y).sum().cpu().detach().numpy()
        train_loss, train_acc = train_loss / len(train_loader), num_correct / len(train_loader.dataset)
        test_loss, test_acc, test_pred = test_report(test_loader, model, loss_func)
        print(
            "Epoch:%d => train_loss:%.10f => train_acc:%.10f%% => test_loss: %.10f => test_acc: %.10f%%=>lr: %.10f" % (
                epoch + 1, train_loss, train_acc * 100, test_loss, test_acc * 100,
                optim.state_dict()['param_groups'][0]['lr']), end=":::::")
        analyseResult(Y_test, test_pred, "BiLSTM_2").print_analyse_result()
        train_loss, num_correct = 0, 0


def train_LSTM_1():
    train_loader, test_loader, (_, _, _, Y_test) = get_1_dataLoader()

    model = LSTM().to(Device)
    optim = torch.optim.Adam(model.parameters(), lr=MODEL_LR)
    loss_func = torch.nn.CrossEntropyLoss().to(Device)
    train_loss = 0
    num_correct = 0
    for epoch in range(MODEL_EPOCH):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(Device), batch_y.to(Device)
            model.train()
            prediction = model(batch_x)
            loss = loss_func(prediction, batch_y)

            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss += loss.cpu().detach().numpy()
            pred = prediction.argmax(dim=1)
            num_correct += torch.eq(pred, batch_y).sum().cpu().detach().numpy()
        train_loss, train_acc = train_loss / len(train_loader), num_correct / len(train_loader.dataset)
        test_loss, test_acc, test_pred = test_report(test_loader, model, loss_func)
        print(
            "Epoch:%d => train_loss:%.10f => train_acc:%.10f%% => test_loss: %.10f => test_acc: %.10f%%=>lr: %.10f" % (
                epoch + 1, train_loss, train_acc * 100, test_loss, test_acc * 100,
                optim.state_dict()['param_groups'][0]['lr']), end=":::::")
        analyseResult(Y_test, test_pred, "LSTM_1").print_analyse_result()
        train_loss, num_correct = 0, 0


def train_LSTM_2():
    train_loader, test_loader, (_, _, _, Y_test) = get_2_dataLoader()

    model = LSTM().to(Device)
    optim = torch.optim.Adam(model.parameters(), lr=MODEL_LR)
    loss_func = torch.nn.CrossEntropyLoss().to(Device)
    train_loss = 0
    num_correct = 0
    for epoch in range(MODEL_EPOCH):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(Device), batch_y.to(Device)
            model.train()
            prediction = model(batch_x)
            loss = loss_func(prediction, batch_y)

            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss += loss.cpu().detach().numpy()
            pred = prediction.argmax(dim=1)
            num_correct += torch.eq(pred, batch_y).sum().cpu().detach().numpy()
        train_loss, train_acc = train_loss / len(train_loader), num_correct / len(train_loader.dataset)
        test_loss, test_acc, test_pred = test_report(test_loader, model, loss_func)
        print(
            "Epoch:%d => train_loss:%.10f => train_acc:%.10f%% => test_loss: %.10f => test_acc: %.10f%%=>lr: %.10f" % (
                epoch + 1, train_loss, train_acc * 100, test_loss, test_acc * 100,
                optim.state_dict()['param_groups'][0]['lr']), end=":::::")
        analyseResult(Y_test, test_pred, "LSTM_2").print_analyse_result()
        train_loss, num_correct = 0, 0


def train_RNN_1():
    pass


def train_RNN_2():
    pass


def train_DNN_1():
    pass


def train_DNN_2():
    pass


def train_ml_1():
    pass


def train_ml_2():
    pass


if __name__ == '__main__':
    # main1()  # 经过OSS-blsmo处理后的数据存放在temp_save
    # main2()  # 训练出来GAN网络并生成数据进行存储
    main22()  # 训练出来WGan网络并生成数据进行存储
    main3()  # 将生成的数据以及本来的数据进行整合保存
    train_BiLSTM_1()
    train_BiLSTM_2()
    train_LSTM_1()
    train_LSTM_2()
