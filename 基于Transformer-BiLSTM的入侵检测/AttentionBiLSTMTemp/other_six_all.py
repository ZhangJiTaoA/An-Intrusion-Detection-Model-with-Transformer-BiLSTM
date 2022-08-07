from Data.data_handler import *
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer
import analyse_result
from config import *

Classify = "binary"
# train_X, test_X, train_Y, test_Y = get_NSLKDD(FiveClassify)
train_X, test_X, train_Y, test_Y = get_NSLKDD(TwoClassify)
train_X = train_X.view((train_X.size()[0], train_X.size()[1]))
test_X = test_X.reshape((test_X.size()[0], test_X.size()[1]))
print(test_X.shape)
train_X = pd.read_csv("./Data2/Xtrain_gen.csv",header=None,index_col=None).to_numpy()
train_Y = pd.read_csv("./Data2/Ytrain_gen.csv",header=None,index_col=None)
train_Y[train_Y!=0]=1

scaleNorm = Normalizer().fit(train_X)
train_X = scaleNorm.transform(train_X)
test_X = scaleNorm.transform(test_X)

# 建立回归模型并训练
model = LogisticRegression()
model.fit(train_X, train_Y)
# 做预测
expected = test_Y
predicted = model.predict(test_X)
# 分析结果并保存
analyse_result.save_analyse_result(expected, predicted, "./Result2/01LR", classify=Classify,dic={})

# 建立朴素贝叶斯模型并训练
model = GaussianNB()
model.fit(train_X, train_Y)
# 做预测
expected = test_Y
predicted = model.predict(test_X)
# 分析结果并保存
analyse_result.save_analyse_result(expected, predicted, "./Result2/02GNB", classify=Classify)

# 建立K近邻模型并训练
model = KNeighborsClassifier()
model.fit(train_X, train_Y)
# 做预测
expected = test_Y
predicted = model.predict(test_X)
# 分析结果并保存
analyse_result.save_analyse_result(expected, predicted, "./Result2/03KNN", classify=Classify)

# 建立决策树模型并训练
model = DecisionTreeClassifier()
model.fit(train_X, train_Y)
# 做预测
expected = test_Y
predicted = model.predict(test_X)
# 分析结果并保存
analyse_result.save_analyse_result(expected, predicted, "./Result2/04DT", classify=Classify)

# 建立集成学习模型并训练
model = AdaBoostClassifier()
model.fit(train_X, train_Y)
# 做预测
expected = test_Y
predicted = model.predict(test_X)
# 分析结果并保存
analyse_result.save_analyse_result(expected, predicted, "./Result2/05AB", classify=Classify)

# 建立随机森林模型并训练
model = RandomForestClassifier()
model.fit(train_X, train_Y)
# 做预测
expected = test_Y
predicted = model.predict(test_X)
# 分析结果并保存
analyse_result.save_analyse_result(expected, predicted, "./Result2/06RF", classify=Classify)

# 建立支持向量机模型并进行训练
model = SVC()
model.fit(train_X, train_Y)
# 做预测
expected = test_Y
predicted = model.predict(test_X)
# 分析结果并保存
analyse_result.save_analyse_result(expected, predicted, "./Result2/07SVM", classify=Classify)
