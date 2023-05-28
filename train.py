import torch
from model import Model
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
train_dataloader = data_loader(train_dataset, cfg)
cfg.pad_size = dataset.max_len_seq

model_text_cls = Model(cfg)                                  # 实例化模型
model_text_cls.to(cfg.device)                                # GPU

loss_func = torch.nn.CrossEntropyLoss()                      # 损失函数
optimizer = torch.optim.Adam(model_text_cls.parameters(), lr=cfg.lr)   # 优化器

if torch.cuda.is_available() == True:
    print("Running on GPU")
else:
    print('Running on CPU')

x = np.arange(0, 100)
loss_list = np.zeros(100)
rec_list = np.zeros(100)  # 召回率存储
acc_list = np.zeros(100)  # 准确率
pre_list = np.zeros(100)  # 精确率
F1_list = np.zeros(100)   # F1

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

for epoch in range(cfg.epoch):
    loss = 0
    loss_l = []  # 存储loss，方便拿个数求平均值

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
    for i, batch in enumerate(train_dataloader):
        label, data = batch
        data = torch.as_tensor(data).to(cfg.device)
        label = torch.as_tensor(label).to(cfg.device)

        optimizer.zero_grad()                      # 优化器参数清零
        pred_softmax = model_text_cls.forward(data)        # 前向函数
        loss_val = loss_func(pred_softmax, label)          # 损失值计算

        # print(pred)
        # print(label)

        loss_val.backward()                        # 反向传播
        optimizer.step()                           # 参数更新
        pred = torch.argmax(pred_softmax, dim=1)   # 拿到最终预测标签

        loss += loss_val  # loss总和
        loss_l.append(loss_val)


        for j in range(cfg.batch_size):
            y_true.append(label[j].item())
            y_pred.append(pred[j].item())

    Cm = confusion_matrix(y_true, y_pred)  # 每轮训练的混淆矩阵

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
    '''
    a0 = Cm[(0, 0)]
    a1 = Cm[(0, 1)]
    b0 = Cm[(1, 0)]
    b1 = Cm[(1, 1)]
    rec = a0 / (a0 + b0)
    acc = (a0 + b1) / (a0 + a1 + b0 + b1)
    pre = a0 / (a0 + a1)
    F1 = (2 * pre * rec) / (pre + rec)
    '''
    loss_list[epoch] = loss / len(loss_l)
    rec_list[epoch] = rec
    acc_list[epoch] = acc
    pre_list[epoch] = pre
    F1_list[epoch] = F1

    print("epoch: {}, loss: {}, rec: {}, acc: {}, pre: {}, F1: {}".format(epoch, loss / len(loss_l), rec, acc, pre, F1))

    # 存储参数
    if (epoch+1) % 10 == 0:
        torch.save(model_text_cls.state_dict(), "model/{}.pth".format(epoch))

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文

cm = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cm, ["0喜悦", "1愤怒", "2厌恶", "3低落"], "Confusion Matrix")  # "0喜悦", "1愤怒", "2厌恶", "3低落"
plt.show()

plt.subplot(2, 2, 1)
plt.plot(x, rec_list)
plt.xlabel("epoch")
plt.ylabel("rec")
plt.title("召回率")

plt.subplot(2, 2, 2)
plt.plot(x, acc_list)
plt.xlabel("epoch")
plt.ylabel("acc")
plt.title("准确率")

plt.subplot(2, 2, 3)
plt.plot(x, pre_list)
plt.xlabel("epoch")
plt.ylabel("pre")
plt.title("精确率")

plt.subplot(2, 2, 4)
plt.plot(x, F1_list)
plt.xlabel("epoch")
plt.ylabel("F1")
plt.title("F1")

plt.show()

plt.plot(x, loss_list)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("损失")
plt.show()