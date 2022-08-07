"""
-*- coding:utf-8 -*-
@Time : 2022/2/28 11:28
@Author : 597778912@qq.com
@File : GAN.py
"""
import torch
import torch.nn as nn
import torch.functional as F
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import os
import ChildModels.Parameters as Parameters


class Gan(object):
    def __init__(self, x, name_n=0):
        self.x = torch.FloatTensor(x)
        # self.seq_len = Parameters.SEQ_LEN  # 多少列，即是一条x的长度
        self.seq_len = x.shape[1]
        self.generator = Generator(self.seq_len).to(Parameters.Device)  # 最终应当保存该生成器的模型结构和参数用于fit的生成数据
        self.discriminate = Discriminate(self.seq_len).to(Parameters.Device)  # 判别器，也保存了吧。用来判断fit生成样本的质量？

        self.G_name = "G_" + str(name_n) + ".pkl"
        self.D_name = "D_" + str(name_n) + ".pkl"
        self.Samples_name = "Sam_" + str(name_n) + ".csv"

    # 训练模型
    # input1(batch_size,G_INPUT_LEN):服从某种分布的随机数列
    # input2(batch_size,SEQ_LEN):需要扩充的某一类的数列
    # batch_size应该等于训练集数据的个数？
    def train(self):  # G_name,D_name为保存模型结构和参数的文件名
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=Parameters.LR_G)
        optimizer_D = torch.optim.Adam(self.discriminate.parameters(), lr=Parameters.LR_D)
        BCE_loss = nn.BCEWithLogitsLoss()
        train_dataset = Data.TensorDataset(self.x)
        train_loader = Data.DataLoader(
            dataset=train_dataset
            , batch_size=Parameters.GAN_BATCH_SIZE
            , shuffle=True
        )
        print("模型的结构：------------------------------------------------------------------------")
        print(self.generator)
        print(self.discriminate)
        print()
        print("基本参数：epoch:%d,batch_size:%d,优化器：%s" % (
            Parameters.GAN_EPOCH, Parameters.GAN_BATCH_SIZE, optimizer_G.__str__()))
        print("接下来开始训练：--------------------------------------------------------------")

        for epoch in range(Parameters.GAN_EPOCH):
            for step, (batch_x,) in enumerate(train_loader):
                batch_noise = torch.randn(batch_x.size()[0], Parameters.G_INPUT_DIM)
                batch_noise, batch_x = batch_noise.to(Parameters.Device), batch_x.to(Parameters.Device)
                self.generator.train()
                self.discriminate.train()

                g_data = self.generator(batch_noise)
                g_data_detach = g_data.detach()
                prob_real = self.discriminate(batch_x)
                prob_fake = self.discriminate(g_data_detach)

                real_label = torch.ones(batch_x.size()[0], 1, device=Parameters.Device)
                fake_label = torch.zeros(batch_x.size()[0], 1, device=Parameters.Device)
                d_loss_real = BCE_loss(prob_real, real_label)
                d_loss_fake = BCE_loss(prob_fake, fake_label)
                d_loss = torch.div(torch.add(d_loss_real, d_loss_fake), 2)
                # 很显然，mean里面的这部分是一个负值，如果想整体loss变小，必须要变成正直，加一个负号，否则会越来越大
                # d_loss = -torch.mean(torch.log(prob_real) + torch.log(1 - prob_fake))
                # d_loss = torch.mean(0.5 * (prob_fake + 1 - prob_real))
                optimizer_D.zero_grad()
                # d_loss.backward(retain_graph=True)  # 这个参数retain_graph=True，因为G网络的损失值使用了D网络计算之后的数据，所以保留其反向之后的参数给G进行计算
                d_loss.backward()
                optimizer_D.step()

                for i in range(3):
                    batch_noise = torch.randn(batch_x.size()[0], Parameters.G_INPUT_DIM).to(Parameters.Device)
                    g_data = self.generator(batch_noise)
                    prob_fake = self.discriminate(g_data)
                    # 而g的loss要使得discriminator的prob_fake尽可能小，这样才能骗过它，因此也要加一个负号
                    # g_loss = torch.mean(torch.log(1-prob_fake))
                    g_loss = BCE_loss(prob_fake, real_label)
                    optimizer_G.zero_grad()
                    g_loss.backward()
                    optimizer_G.step()
                print(
                    "Epoch:%d => step:%d => d_loss_real:%.10f => d_loss_fake:%.10f => d_loss:%.10f => g_loss:%.10f" % (
                        epoch + 1, step + 1, d_loss_real, d_loss_fake, d_loss, g_loss))

        torch.save(self.generator.state_dict(), Parameters.Pkl_path + self.G_name)
        torch.save(self.discriminate.state_dict(), Parameters.Pkl_path + self.D_name)

    # 生成新的数据
    def fit(self, g_samples_n=20):  # Samples_name为保存生成样本的文件名
        if not os.path.exists(Parameters.Pkl_path + self.G_name):
            self.train()
        self.generator.load_state_dict(torch.load(Parameters.Pkl_path + self.G_name))
        self.generator.eval()

        batch_noise = torch.randn(g_samples_n, Parameters.G_INPUT_DIM).to(Parameters.Device)
        g_data = self.generator(batch_noise)
        g_data = g_data.cpu().detach().numpy()
        np.savetxt(Parameters.Pkl_path+self.Samples_name, g_data, delimiter=',')
        return g_data


# input(batch_size,g_input_len)    output(batch_size,seq_len)
class Generator(nn.Module):
    def __init__(self, seq_len):  # 参数seq_len为生成器最终生成的维度
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(Parameters.G_INPUT_DIM, 256)
        self.linear1_act = nn.ReLU()
        self.linear1_drop = nn.Dropout()
        self.linear2 = nn.Linear(256, 128)
        self.linear2_act = nn.ReLU()
        self.linear2_drop = nn.Dropout()
        self.linear3 = nn.Linear(128, seq_len)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear1_act(x)
        x = self.linear1_drop(x)
        x = self.linear2(x)
        x = self.linear2_act(x)
        x = self.linear2_drop(x)
        x = self.linear3(x)
        return x


# input(batch_size,seq_len)   output(batch_size,1)
class Discriminate(nn.Module):
    def __init__(self, seq_len):  # 参数seq_len为判决器的输入维度
        super(Discriminate, self).__init__()
        self.linear1 = nn.Linear(seq_len, 256)
        self.linear1_act = nn.ReLU()
        self.linear1_drop = nn.Dropout()
        self.linear2 = nn.Linear(256, 128)
        self.linear2_act = nn.ReLU()
        self.linear2_drop = nn.Dropout()
        self.linear3 = nn.Linear(128, 1)
        self.linear3_act = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear1_act(x)
        x = self.linear1_drop(x)
        x = self.linear2(x)
        x = self.linear2_act(x)
        x = self.linear2_drop(x)
        x = self.linear3(x)
        x = self.linear3_act(x)
        return x
