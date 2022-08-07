import pandas as pd
import numpy as np
import Data.DataParameters as DP
Y = pd.read_csv("./ChildModels/temp_save/Ytrain_gen.csv", header=None,index_col=None)
al = Y.shape[0]
print(Y.shape)
printdict = {}
for k, v in DP.labels.items():
    num = np.equal(Y, v).sum()
    # print(k + ":" + str(num) + ":" + str(num / Y_train.size()[0]))
    printdict[k] = int(num)
    print(k,num/al)
print(printdict)
