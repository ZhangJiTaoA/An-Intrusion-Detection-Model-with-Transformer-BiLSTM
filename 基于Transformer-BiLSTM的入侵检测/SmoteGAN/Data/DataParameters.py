"""
-*- coding:utf-8 -*-
@Time : 2022/6/5 16:38
@Author : 597778912@qq.com
@File : DataParameters.py
"""
import os
# 自己设置的参数
train_data_path = os.path.dirname(__file__)+"/MyKDDTrain.csv"
test_data_path = os.path.dirname(__file__)+"/KDDTest+.txt"
# 离散属性所在的列, 0开头, 正常的列数减一得到0开头的列数
discrete_arr = [1, 2, 3, 6, 11, 13, 14, 20, 21]
continuous_arr = [0, 4, 5, 7, 8, 9, 10, 12, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                  36, 37, 38, 39, 40]
select_features = [0, 1, 2, 3, 4, 5, 9, 16, 23, 28, 32, 33, 34, 38]
labels_classify = {'neptune': 'dos', 'warezclient': 'r2l', 'ipsweep': 'probe', 'portsweep': 'probe', 'teardrop': 'dos',
                   'nmap': 'probe', 'satan': 'probe', 'smurf': 'dos', 'pod': 'dos', 'back': 'dos',
                   'guess_passwd': 'r2l',
                   'ftp_write': 'r2l', 'multihop': 'r2l', 'rootkit': 'u2r', 'buffer_overflow': 'u2r', 'imap': 'r2l',
                   'warezmaster': 'r2l', 'phf': 'r2l', 'land': 'dos', 'loadmodule': 'u2r', 'spy': 'r2l',
                   'saint': 'probe',
                   'mscan': 'probe', 'apache2': 'dos', 'snmpgetattack': 'r2l', 'processtable': 'dos',
                   'httptunnel': 'u2r',
                   'ps': 'u2r', 'snmpguess': 'r2l', 'mailbomb': 'dos', 'named': 'r2l', 'sendmail': 'r2l',
                   'xterm': 'u2r',
                   'worm': 'r2l', 'xlock': 'r2l', 'perl': 'u2r', 'xsnoop': 'r2l', 'sqlattack': 'u2r', 'udpstorm': 'dos'}

labels = {"normal": 0, "dos": 1, "probe": 2, "r2l": 3, "u2r": 4}
