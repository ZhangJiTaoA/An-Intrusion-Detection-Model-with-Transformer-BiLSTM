import itertools
from Data.data_handler import *
import torch, torchvision
from config import *
from torchsummary import summary
from ChildModel.DNN import DNN
from MyModel.EncoderBiLSTMDNN_two_classify import *
from MyModel.EncoderBiLSTMDNN_five_classify import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
from analyse_result import *
from matplotlib import font_manager

font = font_manager.FontProperties(fname='./msyhl.ttc')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决无法显示符号问题

sns.set(palette='muted', color_codes=True)
sns.set(font="SimHei", font_scale=0.8)  # 美化图像，显示中文
sns.set_style('white')

train_acc = np.load("./Result/two_Classify_train_acc_arr.np.npy") * 100
test_acc = np.load("./Result/two_Classify_test_acc_arr.np.npy") * 100
# plt.axis([0,30])


# plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter())
my_style = ['-+', '-o', '-*', '-.', '-x', '-s', '-p', '-h', '-d', '-^']
test_acc1 = np.load("./Result/isInitParameter/five_Classify_false_train_loss_arr1.np.npy")
test_acc2 = np.load("./Result/isInitParameter/five_Classify_true_train_loss_arr1.np.npy")
# test_acc3 = np.load("./Result/differentLR/two_Classify_lr_0.0001_test_acc_arr1.np.npy")
# test_acc4 = np.load("./Result/differentLR/two_Classify_lr_0.00001_test_acc_arr1.np.npy")
test_acc = [test_acc1[::2], test_acc2[::2]]

for i in range(10):
    # train_acc = np.load("./Result/five_Classify_train_acc_arr"+str(i+1)+".np.npy") * 100
    test_acc = np.load("./Result/MultiAttentionBiLSTMDNN/two_Classify_test_acc_arr" + str(i + 1) + ".np.npy") * 100
    # plt.axis([0,20,40,100])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("epochs", fontsize=15)
    plt.ylabel('test_acc/%', fontsize=15)
    # plt.plot(np.arange(1, 41,2).astype(dtype=np.str), train_acc[::2],'-o')
    plt.plot(np.arange(1, 21).astype(dtype=np.str), test_acc, my_style[i])
# plt.legend(["第"+str(i+1)+"次"])
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.legend(["第" + str(i + 1) + "次" for i in range(10)], prop=font)
# leg = ["对比实验","本文模型"]
# plt.legend(["对比实验","本文模型"],prop=font)
# plt.legend(leg,prop={"size":15},loc=7,bbox_to_anchor=(1.1,0.3))
sns.despine()
plt.savefig('./img/11two_classify_MABD_10.jpg', dpi=600, bbox_inches='tight')
plt.show()

model = EncoderBiLSTMDNN_Five_Classify()
binary_result_arr = []
for i in range(10):
    dic = load_analyse_result("./Result/" + model._get_name() + str(i + 1) + "_multi_result")
    binary_result_arr.append(dic)

all_accuracy = 0
all_recall = 0
all_precision = 0
all_f1 = 0
all_cm = np.zeros((5, 5))
for i in range(10):
    all_accuracy += binary_result_arr[i]["accuracy"]
    all_precision += binary_result_arr[i]["precision"]
    all_recall += binary_result_arr[i]["recall"]
    all_f1 += binary_result_arr[i]['f1']
    all_cm += binary_result_arr[i]['cm']
print("average_acc:" + str(all_accuracy / 10))
print("average_precision:" + str(all_precision / 10))
print("average_recall:" + str(all_recall / 10))
print("average_f1:" + str(all_f1 / 10))


def plot_confusion_matrix(cm, classes, normalize=False, title='State transition matrix', cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.axis('equal')

    ax = plt.gca()

    left, right = plt.xlim()
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor('white')

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:,2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(j, i, num, verticalalignment='center', horizontalalignment='center',
                 color='white' if num > thresh else 'black')

    plt.ylabel('Slef patt')
    plt.xlabel('Transition patt')
    plt.tight_layout()
    plt.savefig('./img/TBD.jpg', dpi=600, bbox_inches='tight')
    plt.show()


from MyModel.MultiAttentiomBiLSTMDNN_two_classify import MultiAttentionBiLSTMDNN_Two_Classify
from MyModel.MultiAttentiomBiLSTMDNN_five_classify import MultiAttentionBiLSTMDNN_Five_Classify


def plot_bar_fig():
    # model = MultiAttentionBiLSTMDNN_Five_Classify()
    # binary_result_arr = []
    # for i in range(10):
    #     dic = load_analyse_result("./Result/MultiAttentionBiLSTMDNN/" + model._get_name() + str(i + 1) + "_multi_result")
    #     binary_result_arr.append(dic)
    #
    # all_accuracy = 0
    # all_recall = 0
    # all_precision = 0
    # all_f1 = 0
    # all_cm = np.zeros((5, 5))
    # for i in range(10):
    #     all_accuracy += binary_result_arr[i]["accuracy"]
    #     all_precision += binary_result_arr[i]["precision"]
    #     all_recall += binary_result_arr[i]["recall"]
    #     all_f1 += binary_result_arr[i]['f1']
    # #    all_cm += binary_result_arr[i]['cm']
    # print("average_acc:" + str(all_accuracy / 10))
    # print("average_precision:" + str(all_precision / 10))
    # print("average_recall:" + str(all_recall / 10))
    # print("average_f1:" + str(all_f1 / 10))
    # result = {"model": model._get_name(),
    #           "accuracy": all_accuracy / 10,
    #           "recall": all_recall / 10,
    #           "precision": all_precision / 10,
    #           "f1": all_f1/10
    #           # ,"tpr": tpr
    #           # ,"fpr": fpr
    #           }
    # with open("./Result/MultiAttentionBiLSTMDNN/"+model._get_name() + "average_multi_result", "w") as f:
    #     json.dump(result, f, indent=1)

    # 开始画图
    name_list = ["TBD", "BD", "MABD", "TD", "PTD", "SVM", "RF", "DT"]
    TBD = load_analyse_result("./Result/EncoderBiLSTMDNN_Five_Classifyaverage_multi_result")
    BD = load_analyse_result("./Result/BiLSTMDNN/BiLSTMDNN_Five_Classify1_multi_result")
    MABD = load_analyse_result(
        "./Result/MultiAttentionBiLSTMDNN/MultiAttentionBiLSTMDNN_Five_Classifyaverage_multi_result")
    TD = load_analyse_result("./Result/TransformerDNN/PositionEncoderDNN_Five_Classify1_multi_result")
    PTD = load_analyse_result("./Result/PositionTransformerDNN/PositionEncoderDNN_Five_Classify1_multi_result")
    SVM = load_analyse_result("./Result/07SVM_multi_result")
    RF = load_analyse_result("./Result/06RF_multi_result")
    DT = load_analyse_result("./Result/04DT_multi_result")
    accuracy = [TBD["accuracy"], BD["accuracy"], MABD["accuracy"], TD["accuracy"], PTD["accuracy"], SVM["accuracy"],
                RF["accuracy"], DT["accuracy"]]
    recall = [TBD["recall"], BD["recall"], MABD["recall"], TD["recall"], PTD["recall"], SVM["recall"], RF["recall"],
              DT["recall"]]
    precision = [TBD["precision"], BD["precision"], MABD["precision"], TD["precision"], PTD["precision"],
                 SVM["precision"], RF["precision"], DT["precision"]]
    f1 = [TBD["f1"], BD["f1"], MABD["f1"], TD["f1"], PTD["f1"], SVM["f1"], RF["f1"], DT["f1"]]
    sns.set(palette='muted', color_codes=True)
    sns.set(font="SimHei", font_scale=0.8)  # 美化图像，显示中文
    sns.set_style('white')
    x = np.arange(8)
    total_width, n = 0.7, 4
    width = total_width / n
    x = x - (total_width - width) / 2
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("models", fontsize=10)
    plt.ylabel('value', fontsize=10)
    # plt.bar(x, accuracy, width=width, label='accuracy')
    # plt.bar(x + width, recall, width=width, label='recall')
    # plt.bar(x + width * 2, precision, width=width, label='precision', tick_label=name_list)
    # plt.bar(x + width * 3, f1, width=width, label='f1-score')
    plt.bar(x, accuracy, width=width, label='accuracy',color='w',edgecolor='k')
    plt.bar(x + width, recall, width=width, label='recall',color='w',edgecolor='k',hatch="***")
    plt.bar(x + width * 2, precision, width=width, label='precision', tick_label=name_list,color='w',edgecolor='k',hatch="...")
    plt.bar(x + width * 3, f1, width=width, label='f1-score',color='w',edgecolor='k',hatch="///")
    plt.legend(ncol=4,prop={"size":10},bbox_to_anchor=(1,1.05),framealpha=0)
    sns.despine()
    plt.savefig('./img/16five_classify_everyone_bar.jpg', dpi=600, bbox_inches='tight')
    plt.show()

def plot_bar_two_classify_fig():
    # model = MultiAttentionBiLSTMDNN_Five_Classify()
    # binary_result_arr = []
    # for i in range(10):
    #     dic = load_analyse_result("./Result/MultiAttentionBiLSTMDNN/" + model._get_name() + str(i + 1) + "_multi_result")
    #     binary_result_arr.append(dic)
    #
    # all_accuracy = 0
    # all_recall = 0
    # all_precision = 0
    # all_f1 = 0
    # all_cm = np.zeros((5, 5))
    # for i in range(10):
    #     all_accuracy += binary_result_arr[i]["accuracy"]
    #     all_precision += binary_result_arr[i]["precision"]
    #     all_recall += binary_result_arr[i]["recall"]
    #     all_f1 += binary_result_arr[i]['f1']
    # #    all_cm += binary_result_arr[i]['cm']
    # print("average_acc:" + str(all_accuracy / 10))
    # print("average_precision:" + str(all_precision / 10))
    # print("average_recall:" + str(all_recall / 10))
    # print("average_f1:" + str(all_f1 / 10))
    # result = {"model": model._get_name(),
    #           "accuracy": all_accuracy / 10,
    #           "recall": all_recall / 10,
    #           "precision": all_precision / 10,
    #           "f1": all_f1/10
    #           # ,"tpr": tpr
    #           # ,"fpr": fpr
    #           }
    # with open("./Result/MultiAttentionBiLSTMDNN/"+model._get_name() + "average_multi_result", "w") as f:
    #     json.dump(result, f, indent=1)

    # 开始画图
    name_list = ["TBD", "BD", "MABD", "TD", "PTD", "SVM", "RF", "DT"]
    TBD = load_analyse_result("./Result/EncoderBiLSTMDNN_Two_Classifyaverage_binary_result")
    BD = load_analyse_result("./Result/BiLSTMDNN/BiLSTMDNN_Two_Classify1_binary_result")
    MABD = load_analyse_result(
        "./Result/MultiAttentionBiLSTMDNN/MultiAttentionBiLSTMDNN_Two_Classifyaverage_binary_result")
    TD = load_analyse_result("./Result/TransformerDNN/PositionEncoderDNN_Two_Classify1_binary_result")
    PTD = load_analyse_result("./Result/PositionTransformerDNN/PositionEncoderDNN_Two_Classify1_binary_result")
    SVM = load_analyse_result("./Result/07SVM_binary_result")
    RF = load_analyse_result("./Result/06RF_binary_result")
    DT = load_analyse_result("./Result/04DT_binary_result")
    accuracy = [TBD["accuracy"], BD["accuracy"], MABD["accuracy"], TD["accuracy"], PTD["accuracy"], SVM["accuracy"],
                RF["accuracy"], DT["accuracy"]]
    recall = [TBD["recall"], BD["recall"], MABD["recall"], TD["recall"], PTD["recall"], SVM["recall"], RF["recall"],
              DT["recall"]]
    precision = [TBD["precision"], BD["precision"], MABD["precision"], TD["precision"], PTD["precision"],
                 SVM["precision"], RF["precision"], DT["precision"]]
    f1 = [TBD["f1"], BD["f1"], MABD["f1"], TD["f1"], PTD["f1"], SVM["f1"], RF["f1"], DT["f1"]]
    sns.set(palette='muted', color_codes=True)
    sns.set(font="SimHei", font_scale=0.8)  # 美化图像，显示中文
    sns.set_style('white')
    x = np.arange(8)
    total_width, n = 0.7, 4
    width = total_width / n
    x = x - (total_width - width) / 2
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("models", fontsize=10)
    plt.ylabel('value', fontsize=10)
    # plt.bar(x, accuracy, width=width, label='accuracy')
    # plt.bar(x + width, recall, width=width, label='recall')
    # plt.bar(x + width * 2, precision, width=width, label='precision', tick_label=name_list)
    # plt.bar(x + width * 3, f1, width=width, label='f1-score')
    plt.bar(x, accuracy, width=width, label='accuracy',color='w',edgecolor='k')
    plt.bar(x + width, recall, width=width, label='recall',color='w',edgecolor='k',hatch="***")
    plt.bar(x + width * 2, precision, width=width, label='precision', tick_label=name_list,color='w',edgecolor='k',hatch="...")
    plt.bar(x + width * 3, f1, width=width, label='f1-score',color='w',edgecolor='k',hatch="///")
    plt.legend(ncol=4,prop={"size":10},bbox_to_anchor=(1,1.05),framealpha=0)
    sns.despine()
    plt.savefig('./img/15two_classify_everyone_bar.jpg', dpi=600, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    #RF = load_analyse_result("./Result/06RF_multi_result")
    #DT = load_analyse_result("./Result/04DT_multi_result")

    #plot_confusion_matrix(np.array(RF['cm']),labels.keys())
    #plot_confusion_matrix(np.array(DT['cm']),labels.keys())
    # plot_confusion_matrix(all_cm/10,labels.keys())
    plot_bar_two_classify_fig()

    pass
