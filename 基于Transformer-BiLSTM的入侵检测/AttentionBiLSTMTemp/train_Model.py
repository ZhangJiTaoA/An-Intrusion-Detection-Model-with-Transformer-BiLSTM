import torch
import torch.nn

import analyse_result
from Data.data_handler import *
from MyModel.EncoderBiLSTMDNN_five_classify import *
from MyModel.EncoderBiLSTMDNN_two_classify import *
from MyModel.MultiAttentiomBiLSTMDNN_five_classify import *
from MyModel.MultiAttentiomBiLSTMDNN_two_classify import *
from MyModel.BiLSTMDNN_two_classify import BiLSTMDNN_Two_Classify
from MyModel.BiLSTMDNN_five_classify import BiLSTMDNN_Five_Classify
from MyModel.PositionEncoderDNN_two_classify import PositionEncoderDNN_Two_Classify
from MyModel.TransformerDNN_five_classify import EncoderDNN_Five_Classify
from MyModel.TransformerDNN_two_classify import EncoderDNN_Two_Classify
from MyModel.PositionEncoderDNN_five_classify import PositionEncoderDNN_Five_Classify
import torch.optim as optimizer
import torch.utils.data as data
from config import *

def get_2_dataLoader(classify=5):  # 获取数据平衡后的数据的dataloader
    X_train, X_test, Y_train, Y_test = get_NSLKDD(classify)
    if X_test.shape[0] == Y_test.shape[0]:
        print('测试集处理成功！测试集维度为：', X_test.size(), "测试标签维度为：", Y_test.size())
    # 装载数据到loader里面
    test_dataset = data.TensorDataset(X_test, Y_test)
    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=BatchSize,
        shuffle=False
    )

    X_train = pd.read_csv("./Data2/Xtrain_gen.csv", header=None, index_col=None)
    Y_train = pd.read_csv("./Data2/Ytrain_gen.csv", header=None, index_col=None)
    if classify==2:
        Y_train[Y_train!=0]=1
    X_train = torch.FloatTensor(X_train.to_numpy()).unsqueeze(dim=-1)
    Y_train = torch.LongTensor(Y_train.to_numpy().squeeze())
    if classify==5:
        printdict = {}
        print("train_data,total:" + str(Y_train.size()[0]))
        for k, v in labels.items():
            num = torch.eq(Y_train, v).sum().cpu().detach().numpy()
            # print(k + ":" + str(num) + ":" + str(num / Y_test.size()[0]))
            printdict[k] = int(num)
        print(printdict)

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BatchSize,
        shuffle=True
    )

    print("数据装载完成")
    print("数据准备完成：------------------------------------------------------------")
    return train_loader, test_loader, (X_train, X_test, Y_train, Y_test)


def get_dataset_loader(classify):
    # 准备数据
    X_train, X_test, Y_train, Y_test = get_NSLKDD(classify)

    if X_train.size()[0] == Y_train.size()[0]:
        print("训练集处理成功！训练集维度为：", X_train.size(), "训练标签维度为：", Y_train.size())
    if X_test.shape[0] == Y_test.shape[0]:
        print('测试集处理成功！测试集维度为：', X_test.size(), "测试标签维度为：", Y_test.size())

    # 装载数据到loader里面
    train_dataset = data.TensorDataset(X_train, Y_train)
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=BatchSize,
        shuffle=True
    )
    test_dataset = data.TensorDataset(X_test, Y_test)
    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=BatchSize,
        shuffle=False
    )
    print("数据装载完成")
    print("数据准备完成：------------------------------------------------------------")
    return train_loader, test_loader, (X_train, X_test, Y_train, Y_test)


def train(train_loader, model, loss_func, optimizer):
    train_loss = 0
    num_correct = 0
    for step, (batch_x, batch_y) in enumerate(train_loader):
        batch_x, batch_y = batch_x.to(Device), batch_y.to(Device)
        model.train()
        prediction = model(batch_x)
        loss = loss_func(prediction, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.cpu().detach().numpy()
        pred = prediction.argmax(dim=1)
        num_correct += torch.eq(pred, batch_y).sum().cpu().detach().numpy()
    train_loss, train_acc = train_loss / len(train_loader), num_correct / len(train_loader.dataset)
    return train_loss, train_acc


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


def train_model_two_classify(model, count=0):
    # train_loader, test_loader, (X_train, X_test, Y_train, Y_test) = get_dataset_loader(TwoClassify)
    train_loader, test_loader, (X_train, X_test, Y_train, Y_test) = get_2_dataLoader(TwoClassify)
    print(X_train.size())
    my_model = model.to(Device)
    if (IsInitParams):
        initNetParams(my_model, w=0.1)

    optim = optimizer.Adam(my_model.parameters(), lr=TwoClassifyLR)
    #    scheduler = optimizer.lr_scheduler.StepLR(optim,step_size=20,gamma=0.1)
    scheduler = optimizer.lr_scheduler.MultiStepLR(optim, milestones=TwoClassifyMilestones, gamma=0.1)
    loss_func = nn.CrossEntropyLoss().to(Device)

    print("模型的架构：---------------------------------------------------------------")
    print(my_model)
    print("模型的基本参数：---------------------------------------------------------------")
    print("基本参数：%d分类,epoch:%d,batch_size:%d,损失函数：%s,优化器：%s" % (
        TwoClassify, TwoClassifyEpoch, BatchSize, loss_func.__str__(), optim.__str__()))
    print("接下来开始训练：--------------------------------------------------------------")

    save_acc = 0
    test_acc_array = []
    train_acc_array = []
    train_loss_array = []
    test_loss_array = []
    for epoch in range(TwoClassifyEpoch):
        train_loss, train_acc = train(train_loader, my_model, loss_func, optim)
        test_loss, test_acc, test_pred = test_report(test_loader, my_model, loss_func)

        print(
            "Epoch:%d => train_loss:%.10f => train_acc:%.10f%% => test_loss: %.10f => test_acc: %.10f%%=>lr: %.10f" % (
                epoch + 1, train_loss, train_acc * 100, test_loss, test_acc * 100,
                optim.state_dict()['param_groups'][0]['lr']))

        scheduler.step()
        test_acc_array.append(test_acc)
        train_acc_array.append(train_acc)
        test_loss_array.append(test_loss)
        train_loss_array.append(train_loss)

        if test_acc > save_acc:
            save_acc = test_acc
            torch.save(my_model, model._get_name() + ".pkl")
            print("Epoch:%d,已经保存" % (epoch + 1))
            ### 改这里！
            # analyse_result.save_analyse_result(Y_test, test_pred, "./Result/" + model._get_name() + str(count),
            #                                    classify="binary",
            #                                    dic={"best_epoch": epoch + 1})
            analyse_result.save_analyse_result(Y_test, test_pred, "./Result2/" + model._get_name() + str(count),
                                               classify="binary",
                                               dic={"best_epoch": epoch + 1})

    np.save("./Result2/two_Classify_test_acc_arr" + str(count) + ".np", test_acc_array)
    np.save("./Result2/two_Classify_train_acc_arr" + str(count) + ".np", train_acc_array)
    # np.save("./Result/isInitParameter/two_Classify_true_test_acc_arr" + str(count) + ".np", test_acc_array)
    # np.save("./Result/isInitParameter/two_Classify_true_train_acc_arr" + str(count) + ".np", train_acc_array)
    # np.save("./Result/isInitParameter/two_Classify_true_test_loss_arr" + str(count) + ".np", test_loss_array)
    # np.save("./Result/isInitParameter/two_Classify_true_train_loss_arr" + str(count) + ".np", train_loss_array)


def train_model_five_classify(model, count=0):
    train_loader, test_loader, (X_train, X_test, Y_train, Y_test) = get_2_dataLoader(FiveClassify)
    # train_loader, test_loader, (X_train, X_test, Y_train, Y_test) = get_dataset_loader(FiveClassify)
    my_model = model.to(Device)
    if (IsInitParams):
        initNetParams(my_model)

    optim = optimizer.Adam(my_model.parameters(), lr=FiveClassifyLR)
    #    scheduler = optimizer.lr_scheduler.StepLR(optim,step_size=20,gamma=0.1)
    scheduler = optimizer.lr_scheduler.MultiStepLR(optim, milestones=FiveClassifyMilestones, gamma=0.1)
    loss_func = nn.CrossEntropyLoss().to(Device)

    print("模型的架构：---------------------------------------------------------------")
    print(my_model)
    print("模型的基本参数：---------------------------------------------------------------")
    print("基本参数：%d分类,epoch:%d,batch_size:%d,损失函数：%s,优化器：%s" % (
        FiveClassify, FiveClassifyEpoch, BatchSize, loss_func.__str__(), optim.__str__()))
    print("接下来开始训练：--------------------------------------------------------------")
    save_acc = 0
    test_acc_array = []
    train_acc_array = []
    test_loss_array = []
    train_loss_array = []
    for epoch in range(FiveClassifyEpoch):
        train_loss, train_acc = train(train_loader, my_model, loss_func, optim)
        test_loss, test_acc, test_pred = test_report(test_loader, my_model, loss_func)
        print(
            "Epoch:%d => train_loss:%.10f => train_acc:%.10f%% => test_loss: %.10f => test_acc: %.10f%%=>lr: %.10f" % (
                epoch + 1, train_loss, train_acc * 100, test_loss, test_acc * 100,
                optim.state_dict()['param_groups'][0]['lr']))

        scheduler.step()
        test_acc_array.append(test_acc)
        train_acc_array.append(train_acc)
        test_loss_array.append(test_loss)
        train_loss_array.append(train_loss)
        if test_acc > save_acc:
            save_acc = test_acc
            torch.save(my_model, model._get_name() + "pkl")
            print("Epoch:%d,已经保存" % (epoch + 1))
            # analyse_result.save_analyse_result(Y_test, test_pred, "./Result/" + model._get_name() + str(count),
            #                                    classify="macro",
            #                                    dic={"best_epoch": epoch + 1})
            analyse_result.save_analyse_result(Y_test, test_pred, "./Result2/" + model._get_name() + str(count),
                                               classify="macro",
                                               dic={"best_epoch": epoch + 1})
    np.save("./Result2/five_Classify_test_acc_arr" + str(count) + ".np", test_acc_array)
    np.save("./Result2/five_Classify_train_acc_arr" + str(count) + ".np", train_acc_array)
    # np.save("./Result/isInitParameter/five_Classify_true_test_acc_arr" + str(count) + ".np", test_acc_array)
    # np.save("./Result/isInitParameter/five_Classify_true_train_acc_arr" + str(count) + ".np", train_acc_array)
    # np.save("./Result/isInitParameter/five_Classify_true_test_loss_arr" + str(count) + ".np", test_loss_array)
    # np.save("./Result/isInitParameter/five_Classify_true_train_loss_arr" + str(count) + ".np", train_loss_array)


if __name__ == '__main__':
    # train_model_five_classify(BiLSTMDNN_Five_Classify())
    # print("AttentionNum:", AttentionHeadNum, "是否参数初始化：", IsInitParams)
    # for i in range(10):
    #     print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "第 %d 次" % (i + 1))
    #     train_model_two_classify(EncoderDNN_Two_Classify(), i + 1)

    for i in range(10):
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "第 %d 次" % (i + 1))
        train_model_five_classify(EncoderBiLSTMDNN_Five_Classify(), i + 1)

    # # train_model_two_classify(MultiAttentionBiLSTMDNN_Two_Classify())
    # for i in range(10):
    #     print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "第 %d 次" % (i + 1))
    #     train_model_two_classify(MultiAttentionBiLSTMDNN_Two_Classify())
    # # train_model_five_classify(MultiAttentionBiLSTMDNN_Five_Classify())
    # for i in range(10):# 多分类效果不尽人意
    #     print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "第 %d 次" % (i + 1))
    #     train_model_five_classify(MultiAttentionBiLSTMDNN_Five_Classify())
