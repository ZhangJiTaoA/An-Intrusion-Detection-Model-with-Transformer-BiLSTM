"""
-*- coding:utf-8 -*-
@Time : 2022/6/28 10:59
@Author : 597778912@qq.com
@File : analyse_result.py
"""
import json
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import os
import logging


class analyseResult(object):
    def __init__(self, expected, predicted, model_name, dic={}):
        self.expected = expected
        self.predicted = predicted
        self.model_name = model_name
        self.dic = dic
        self.classify = "macro"
        self.file_path = os.path.dirname(__file__) + "/Result/"

    def __get_analyse_result(self):  # 如果有追加写入.classify=binary/macro
        accuracy = accuracy_score(self.expected, self.predicted)
        recall = recall_score(self.expected, self.predicted, average=self.classify)
        precision = precision_score(self.expected, self.predicted, average=self.classify, zero_division=1)
        f1 = f1_score(self.expected, self.predicted, average=self.classify)
        cm = confusion_matrix(self.expected, self.predicted).tolist()
        # tpr = float(cm[0][0]) / np.sum(cm[0])
        # fpr = float(cm[1][1]) / np.sum(cm[1])
        result = {"model": self.model_name,
                  "accuracy": accuracy,
                  "recall": recall,
                  "precision": precision,
                  "f1": f1, "cm": cm
                  # ,"tpr": tpr
                  # ,"fpr": fpr
                  }
        result.update(self.dic)
        return result

    def print_analyse_result(self):
        result = self.__get_analyse_result()
        logging.basicConfig(
            level=logging.DEBUG,
            filename="./Result/00MyLog.log",
            format='%(asctime)s %(filename)s[line:%(lineno)d ] %(levelname)s %(message)s',
            datefmt='%d %b %Y,%a %H:%M:%S',  # 日 月 年 ，星期 时 分 秒
            filemode='w'
        )
        logging.info(result)
        print(result)
        return result

    def save_analyse_result(self):
        result = self.__get_analyse_result()
        with open(self.file_path + self.model_name + "_result", "w") as f:
            json.dump(result, f, indent=1)

    def load_analyse_result(self):
        with open(self.file_path + self.model_name + "_result", 'r') as f:
            result = json.load(f)
        return result
