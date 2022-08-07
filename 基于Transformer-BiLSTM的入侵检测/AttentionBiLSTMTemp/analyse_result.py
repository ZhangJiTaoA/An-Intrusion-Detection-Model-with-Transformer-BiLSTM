import json
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def get_analyse_result(expected, predicted, model, classify='binary', dic={}):  # 如果有追加写入.classify=binary/macro
    accuracy = accuracy_score(expected, predicted)
    if classify != 'binary':
        classify = "macro"
    recall = recall_score(expected, predicted, average=classify)
    precision = precision_score(expected, predicted, average=classify)
    f1 = f1_score(expected, predicted, average=classify)
    cm = confusion_matrix(expected, predicted).tolist()
    # tpr = float(cm[0][0]) / np.sum(cm[0])
    # fpr = float(cm[1][1]) / np.sum(cm[1])
    result = {"model": model,
              "accuracy": accuracy,
              "recall": recall,
              "precision": precision,
              "f1": f1, "cm": cm
              # ,"tpr": tpr
              # ,"fpr": fpr
              }
    result.update(dic)
    return result


def print_analyse_result(expected, predicted, model, classify="binary", dic={}):
    result = get_analyse_result(expected, predicted, model, classify=classify, dic=dic)
    print(result)
    return result


def save_analyse_result(expected, predicted, model, classify="binary", dic={}):
    result = print_analyse_result(expected, predicted, model, classify=classify, dic=dic)
    if classify == "binary":
        with open(model + "_binary_result", "w") as f:
            json.dump(result, f, indent=1)
    else:
        with open(model + "_multi_result", "w") as f:
            json.dump(result, f, indent=1)


def load_analyse_result(file_path):
    with open(file_path, 'r') as f:
        result = json.load(f)
    return result
