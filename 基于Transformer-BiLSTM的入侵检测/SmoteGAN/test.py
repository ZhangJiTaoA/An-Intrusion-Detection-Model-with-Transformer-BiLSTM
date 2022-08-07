"""
-*- coding:utf-8 -*-
@Time : 2022/5/31 15:42
@Author : 597778912@qq.com
@File : test.py
"""
import numpy as np
import matplotlib.pyplot as plt
import torch

import torch.utils.data as Data
import ChildModels.Parameters as Parameters
import ChildModels.GAN as GAN

# samples = np.array([[3, 6], [4, 3], [6, 2],
#                     [7, 4], [5, 5], [2, 2]])
#
# train_dataset = Data.TensorDataset(torch.from_numpy(samples))
# dataloader = Data.DataLoader(
#     dataset=train_dataset
#     ,batch_size=3
#     ,shuffle=True
# )
#
# for step,batch_x in enumerate(dataloader):
#     print("*"*34)
#     print("step:",step)
#     print("batch_x: ")
#     print(batch_x)
#
#
#
# smote = Smote(N=325, k=5, seed=2)
# synthetic_points = smote.fit(samples)
# print(synthetic_points)
# print(synthetic_points.shape)
# print(smote.newindex)
#
# plt.scatter(samples[:, 0], samples[:, 1], c="blue")
# plt.scatter(synthetic_points[:, 0], synthetic_points[:, 1], c="red")
# plt.show()
# discrete_arr = [1, 2, 3, 6, 11, 13, 14, 20, 21]
# continuous_arr = [0, 4, 5, 7, 8, 9, 10, 12, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
#                   36, 37, 38, 39, 40]
# select_features = [0, 1, 2, 3, 4, 5, 9, 16, 23, 28, 32, 33, 34, 38]
#
#
# l1 = list(set(select_features).intersection(discrete_arr))
# l2 = list(set(select_features).intersection(continuous_arr))
# l1.sort()
# l2.sort()
# print(l1)
# print(l2)
# print([select_features.index(i) for i in l2])

# [3.1555e-01],
#         [5.1361e-01],
#         [4.6226e-01],
#         [1.5532e-01],
#         [7.1420e-01],
#         [4.9349e-01],
#         [3.1232e-01],
print(3.1555e-01)