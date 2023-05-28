import torch
from model import Transformer
from dataset import data_loader, text_ClS
from config import Config        # 参数包
from matplotlib import pyplot as plt
import numpy as np

cfg = Config()
data_path = "D:/桌面/Emotional_classification/data/simplifyweibo_4_moods.csv"
data_stop_path = "D:/桌面/Emotional_classification/data/hit_stopwords.txt"
dict_patch = "D:/桌面/Emotional_classification/data/dict"

dataset = text_ClS(data_path, data_stop_path, dict_patch)    # 实例化dataset类
train_size = int(len(dataset) * 0.7)                         # 训练集大小划分0.7
test_size = len(dataset) - train_size                        # 测试集大小0.3
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
test_dataloader = data_loader(test_dataset, cfg)

cfg.max_len = dataset.max_len_seq

model_text_cls = Transformer(cfg)                            # 实例化模型
model_text_cls.to(cfg.device)                          # GPU
model_text_cls.load_state_dict(torch.load("D:/桌面/Emotional_classification/Transformer/model/49.pth"))        # 加载模型参数

if torch.cuda.is_available() == True:
    print("Running on GPU")
else:
    print('Running on CPU')

from sklearn.metrics import confusion_matrix

#  绘制混淆矩阵
def plot_confusion_matrix(cm, labels_name, title, colorbar=False, cmap='GnBu'):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)  # 在特定的窗口上显示图像
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    if colorbar:
        plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name)  # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)  # 将标签印在y轴坐标上
    plt.title(title)  # 图像标题
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

a0, a1, a2, a3 = 0, 0, 0, 0
b0, b1, b2, b3 = 0, 0, 0, 0
c0, c1, c2, c3 = 0, 0, 0, 0
d0, d1, d2, d3 = 0, 0, 0, 0
rec = 0
acc = 0
pre = 0
F1 = 0
y_true = []
y_pred = []

for i, batch in enumerate(test_dataloader):
    label, data = batch
    data = torch.as_tensor(data).to(cfg.device)
    label = torch.as_tensor(label).to(cfg.device)

    pred_softmax = model_text_cls.forward(data)  # 前向函数
    # print(pred)
    # print(label)
    pred = torch.argmax(pred_softmax, dim=1)  # 拿到最终预测标签

    for j in range(cfg.batch_size):
        y_true.append(label[j].item())
        y_pred.append(pred[j].item())

Cm = confusion_matrix(y_true, y_pred)
a0 = Cm[(0, 0)]
a1 = Cm[(0, 1)]
a2 = Cm[(0, 2)]
a3 = Cm[(0, 3)]
b0 = Cm[(1, 0)]
b1 = Cm[(1, 1)]
b2 = Cm[(1, 2)]
b3 = Cm[(1, 3)]
c0 = Cm[(2, 0)]
c1 = Cm[(2, 1)]
c2 = Cm[(2, 2)]
c3 = Cm[(2, 3)]
d0 = Cm[(3, 0)]
d1 = Cm[(3, 1)]
d2 = Cm[(3, 2)]
d3 = Cm[(3, 3)]

rec = a0 / (a0 + b0 + c0 + d0)  # 召回率
acc = (a0 + b1 + c2 + d3) / (a0 + a1 + a2 + a3 + b0 + b1 + b2 + b3 + c0 + c1 + c2 + c3 + d0 + d1 + d2 + d3)  # 准确率
pre = a0 / (a0 + a1 + a2 + a3)  # 精确率
F1 = (2 * pre * rec) / (pre + rec)

print("rec: {}, acc: {}, pre: {}, F1: {}".format(rec, acc, pre, F1))

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文

plot_confusion_matrix(Cm, ["0喜悦", "1愤怒", "2厌恶", "3低落"], "Confusion Matrix")
plt.show()