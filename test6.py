import torch
from model import Model
from dataset6 import data_loader, text_ClS
from config import Config        # 参数包
from matplotlib import pyplot as plt
import numpy as np

cfg = Config()
data_path = "D:/桌面/Emotional_classification/data/usual_train.txt"
data_stop_path = "D:/桌面/Emotional_classification/data/hit_stopwords.txt"
dict_patch = "D:/桌面/Emotional_classification/data/dict6"

dataset = text_ClS(data_path, data_stop_path, dict_patch)    # 实例化dataset类
train_size = int(len(dataset) * 0.7)                         # 训练集大小划分0.7
test_size = len(dataset) - train_size                        # 测试集大小0.3
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_dataloader = data_loader(test_dataset, cfg)
cfg.max_len = dataset.max_len_seq

model_text_cls = Model(cfg)                            # 实例化模型
model_text_cls.to(cfg.device)                                # GPU
model_text_cls.load_state_dict(torch.load("D:/桌面/Emotional_classification/LSTM/model/89.pth"))

loss_func = torch.nn.CrossEntropyLoss()                      # 损失函数
optimizer = torch.optim.Adam(model_text_cls.parameters(), lr=cfg.lr)   # 优化器

if torch.cuda.is_available() == True:
    print("Running on GPU")
else:
    print('Running on CPU')

from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, labels_name, title, colorbar=False, cmap='GnBu'):
    #  绘制混淆矩阵
    plt.imshow(cm, interpolation='nearest', cmap=cmap)    # 在特定的窗口上显示图像
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    if colorbar:
        plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.title(title)    # 图像标题
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

rec = 0
acc = 0
pre = 0
F1 = 0

y_true = []
y_pred = []
for i, batch in enumerate(train_dataloader):
    label, data = batch
    data = torch.as_tensor(data).to(cfg.device)
    label = torch.as_tensor(label).to(cfg.device)

    pred_softmax = model_text_cls.forward(data)        # 前向函数

    # print(pred)
    # print(label)
    pred = torch.argmax(pred_softmax, dim=1)   # 拿到最终预测标签

    for j in range(cfg.batch_size):
        y_true.append(label[j].item())
        y_pred.append(pred[j].item())

Cm = confusion_matrix(y_true, y_pred)  # 每轮训练的混淆矩阵

rec = Cm[(0, 0)] / (Cm[(0, 0)] + Cm[(1, 0)] + Cm[(2, 0)] + Cm[(3, 0)] + Cm[(4, 0)] + Cm[(5, 0)])  # 召回率

acc = (Cm[(0, 0)] + Cm[(1, 1)] + Cm[(2, 2)] + Cm[(3, 3)] + Cm[(4, 4)] + Cm[(5, 5)]) / \
          (Cm[(0, 0)] + Cm[(0, 1)] + Cm[(0, 2)] + Cm[(0, 3)] + Cm[(0, 4)] + Cm[(0, 5)] +
           Cm[(1, 0)] + Cm[(1, 1)] + Cm[(1, 2)] + Cm[(1, 3)] + Cm[(1, 4)] + Cm[(1, 5)] +
           Cm[(2, 0)] + Cm[(2, 1)] + Cm[(2, 2)] + Cm[(2, 3)] + Cm[(2, 4)] + Cm[(2, 5)] +
           Cm[(3, 0)] + Cm[(3, 1)] + Cm[(3, 2)] + Cm[(3, 3)] + Cm[(3, 4)] + Cm[(3, 5)] +
           Cm[(4, 0)] + Cm[(4, 1)] + Cm[(4, 2)] + Cm[(4, 3)] + Cm[(4, 4)] + Cm[(4, 5)] +
           Cm[(5, 0)] + Cm[(5, 1)] + Cm[(5, 2)] + Cm[(5, 3)] + Cm[(5, 4)] + Cm[(5, 5)])  # 准确率

pre = Cm[(0, 0)] / (Cm[(0, 0)] + Cm[(0, 1)] + Cm[(0, 2)] + Cm[(0, 3)] + Cm[(0, 4)] + Cm[(0, 5)])  # 精确率
F1 = (2 * pre * rec) / (pre + rec)


print("rec: {}, acc: {}, pre: {}, F1: {}".format(rec, acc, pre, F1))

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文

cm = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cm, ["0喜悦", "1愤怒", "2悲伤", "3恐惧", "4惊喜", "5无情绪"], "Confusion Matrix")  # happy、angry、sad、fear、surprise、neutral
plt.show()