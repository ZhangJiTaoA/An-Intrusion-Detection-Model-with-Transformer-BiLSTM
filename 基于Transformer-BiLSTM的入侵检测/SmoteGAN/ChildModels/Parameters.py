"""
-*- coding:utf-8 -*-
@Time : 2022/6/1 19:00
@Author : 597778912@qq.com
@File : Parameters.py
"""
# oss-smote
# OSS
import os

import torch

random_state = 0
SEED = random_state
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 显卡
device_num = 1  # 使用第几块显卡


def get_device(device_num):
    cuda_condition = torch.cuda.is_available()
    cuda0 = torch.device('cuda:0' if cuda_condition else 'cpu')
    cuda1 = torch.device('cuda:1' if cuda_condition else 'cpu')
    device = cuda0 if device_num == 0 else cuda1
    print("当前使用的显卡为：", device)
    return device


Device = get_device(device_num)
# ----------------------------------------------------------


oss_sampling_strategy = [0, 2]

# smote
smote_sampling_strategy = {
    # 0: 10000,
    # 1: 5000,
    # 2: 3000,
    3: 3000,
    4: 500}

# Gan
# SEQ_LEN = 100   # 一条数据的长度,注释掉，将设置权力移交给程序，进行现场计算

Pkl_path = os.path.dirname(__file__) + "/temp_save/"
GAN_BATCH_SIZE = 64
GAN_EPOCH = 30
LR_G = 0.0001  # 生成器学习率
LR_D = 0.0001  # 判别器学习率
G_INPUT_DIM = 150  # 生成器的输入维度
SEQ_LEN = 41  # 目前没有用到
