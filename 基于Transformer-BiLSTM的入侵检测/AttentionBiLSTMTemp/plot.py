import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns


plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决无法显示符号问题

sns.set(palette='muted',color_codes=True)
sns.set(font="SimHei",font_scale=0.8)     # 美化图像，显示中文
sns.set_style('white')
my_style = ['-x', '-o', '-s', '-.', '-+', '-*', '-p', '-h', '-d', '-^']
my_style_color = ["black","white","black"]

# arr1 = np.load("./Result4/differentDropout/drop0/two_Classify_true_train_loss_arr1.np.npy")
arr2 = np.load("./Result4/differentLR/lr0.001/two_Classify_true_test_loss_arr1.np.npy")
arr3 = np.load("./Result4/differentLR/lr0.0001/two_Classify_true_test_loss_arr5.np.npy")
arr4 = np.load("./Result4/differentLR/lr0.00001/two_Classify_true_test_loss_arr1.np.npy")
loss_arrs=[]
# loss_arrs.append(arr1)
loss_arrs.append(arr2)
loss_arrs.append(arr3)
loss_arrs.append(arr4)

# plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter())
print(np.arange(0,0.8,0.1))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("epochs",fontsize=15)
plt.ylabel('test_loss',fontsize=15)


for i in range(len(loss_arrs)):
    plt.plot(np.arange(1, 21).astype(dtype=np.str),
             loss_arrs[i][:20],
             my_style[i],
             markerfacecolor=my_style_color[i],
             color='black')

plt.legend(["0.001","0.0001","0.00001"],fontsize=15)
sns.despine()
plt.savefig('./img/lr/test_2.jpg', dpi=600, bbox_inches='tight')
plt.show()



