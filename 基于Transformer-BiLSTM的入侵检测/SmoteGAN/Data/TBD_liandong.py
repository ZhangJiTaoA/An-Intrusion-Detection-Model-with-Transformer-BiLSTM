import pandas as pd
from ChildModels.Smote import *
from ChildModels.WGAN import *
X = pd.read_csv("tempXTrain.csv",header=None,index_col=None)
Y = pd.read_csv("tempYTrain.csv",header=None,index_col=None)
X,Y = X.to_numpy() ,Y.to_numpy()
print(X.shape)

OSSblsmo = OSSborderlinesmote()
x, y = OSSblsmo.fit(X, Y)

X,Y = pd.DataFrame(x),pd.DataFrame(y)
print(X.shape)

X_Y = pd.concat([X, Y], ignore_index=True, axis=1)
X_probe = X_Y[X_Y[126] == 2].iloc[:, 0:126]
print(X_probe.shape)
X_r2l = X_Y[X_Y[126] == 3].iloc[:, 0:126]
print(X_r2l.shape)
X_u2r = X_Y[X_Y[126] == 4].iloc[:, 0:126]
print(X_u2r.shape)

probGAN = WGan(X_probe.to_numpy(), name_n=2)
probGAN.train()
probGAN.fit(2000)
r2lGAN = WGan(X_r2l.to_numpy(), name_n=3)
r2lGAN.train()
r2lGAN.fit(3000)
u2rGAN = WGan(X_u2r.to_numpy(), name_n=4)
u2rGAN.train()
u2rGAN.fit(2000)

prob_X = pd.read_csv("../ChildModels/temp_save/WSam_2.csv", index_col=None, header=None)
prob_label = pd.DataFrame(np.zeros(prob_X.shape[0]) + 2)
X = pd.concat([X, prob_X], ignore_index=True, axis=0)
Y = pd.concat([Y, prob_label], ignore_index=True, axis=0)

r2l_X = pd.read_csv("../ChildModels/temp_save/WSam_3.csv", index_col=None, header=None)
r2l_label = pd.DataFrame(np.zeros(r2l_X.shape[0]) + 3)
X = pd.concat([X, r2l_X], ignore_index=True, axis=0)
Y = pd.concat([Y, r2l_label], ignore_index=True, axis=0)

u2r_X = pd.read_csv("../ChildModels/temp_save/WSam_4.csv", index_col=None, header=None)
u2r_label = pd.DataFrame(np.zeros(u2r_X.shape[0]) + 4)
X = pd.concat([X, u2r_X], ignore_index=True, axis=0)
Y = pd.concat([Y, u2r_label], ignore_index=True, axis=0)

X.to_csv("../ChildModels/temp_save/Xtrain_gen.csv", index=False, header=False)
Y.to_csv("../ChildModels/temp_save/Ytrain_gen.csv", index=False, header=False)