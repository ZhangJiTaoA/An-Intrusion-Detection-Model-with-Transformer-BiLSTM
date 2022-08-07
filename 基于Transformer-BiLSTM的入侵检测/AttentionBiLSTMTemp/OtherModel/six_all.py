from Data.data_handler import *
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import analyse_result
import time

classify = 2
train_X, test_X, train_Y, test_Y = get_NSLKDD(classify)
# 建立回归模型并训练
start = time.time()
model = LogisticRegression()
model.fit(train_X, train_Y)
end = time.time()
# 做预测
expected = test_Y
predicted = model.predict(test_X)
# 分析结果并保存
expend_time = {"expend_time": end - start}
analyse_result.save_analyse_result(expected, predicted, "../Result/01LR", expend_time)

# 建立朴素贝叶斯模型并训练
start = time.time()
model = GaussianNB()
model.fit(train_X, train_Y)
end = time.time()
# 做预测
expected = test_Y
predicted = model.predict(test_X)
# 分析结果并保存
expend_time = {"expend_time": end - start}
analyse_result.save_analyse_result(expected, predicted, "../Result/02GNB", expend_time)

# 建立K近邻模型并训练
start = time.time()
model = KNeighborsClassifier()
model.fit(train_X, train_Y)
end = time.time()
# 做预测
expected = test_Y
predicted = model.predict(test_X)
# 分析结果并保存
expend_time = {"expend_time": end - start}
analyse_result.save_analyse_result(expected, predicted, "../Result/03KNN", expend_time)

# 建立决策树模型并训练
start = time.time()
model = DecisionTreeClassifier()
model.fit(train_X, train_Y)
end = time.time()
# 做预测
expected = test_Y
predicted = model.predict(test_X)
# 分析结果并保存
expend_time = {"expend_time": end - start}
analyse_result.save_analyse_result(expected, predicted, "../Result/04DT", expend_time)

# 建立集成学习模型并训练
start = time.time()
model = AdaBoostClassifier()
model.fit(train_X, train_Y)
end = time.time()
# 做预测
expected = test_Y
predicted = model.predict(test_X)
# 分析结果并保存
expend_time = {"expend_time": end - start}
analyse_result.save_analyse_result(expected, predicted, "../Result/05AB", expend_time)

# 建立随机森林模型并训练
start = time.time()
model = RandomForestClassifier()
model.fit(train_X, train_Y)
end = time.time()
# 做预测
expected = test_Y
predicted = model.predict(test_X)
# 分析结果并保存
expend_time = {"expend_time": end - start}
analyse_result.save_analyse_result(expected, predicted, "../Result/06RF", expend_time)
