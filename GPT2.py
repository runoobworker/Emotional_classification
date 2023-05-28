from transformers import BertTokenizer
from transformers import GPT2Config, GPT2Model
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

logging.set_verbosity_error()

x_list = np.arange(0, 10)
loss_list = np.zeros(10)
rec_list = np.zeros(10)  # 召回率存储
acc_list = np.zeros(10)  # 准确率
pre_list = np.zeros(10)  # 精确率
F1_list = np.zeros(10)   # F1

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

df = pd.read_csv('D:/桌面/Emotional_classification/data/weibo_senti_100k.csv')
x = list(df['review'])
y = list(df['label'])
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
# GPT2Tokenizer，是以字节为单位的字节对编码，不是以中文的字或词为单位,故使用bert分词器
tokenizer = BertTokenizer.from_pretrained('uer/gpt2-chinese-cluecorpussmall')
train_encoding = tokenizer(x_train, truncation=True, padding=True, max_length=256)
test_encoding = tokenizer(x_test, truncation=True, padding=True, max_length=256)

# print(train_encoding.keys())
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encoding = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encoding.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_encoding, y_train)
test_dataset = NewsDataset(test_encoding, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, drop_last=True)

class CustomGPT2Model(torch.nn.Module):
    def __init__(self, num_labels):
        super(CustomGPT2Model, self).__init__()
        config = GPT2Config.from_pretrained('uer/gpt2-chinese-cluecorpussmall')
        self.gpt2 = GPT2Model.from_pretrained('uer/gpt2-chinese-cluecorpussmall', config=config)
        self.classifier = torch.nn.Linear(self.gpt2.config.n_embd, num_labels)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0][:, -1, :]
        logits = self.classifier(last_hidden_state)
        logits = self.softmax(logits)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = CustomGPT2Model(2).to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
total_steps = len(train_loader) * 1
scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=total_steps)  # 学习率预热：从0线性增加到优化器中的初始lr。

def train():
    model.train()
    total_train_loss = 0
    total_iter = len(train_loader)

    rec = 0
    acc = 0
    pre = 0
    F1 = 0

    y_true = []
    y_pred = []
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪

        optim.step()
        scheduler.step()
        pred = torch.argmax(outputs, dim=1)

        for j in range(16):
            y_true.append(labels[j].item())
            y_pred.append(pred[j].item())

    Cm = confusion_matrix(y_true, y_pred)  # 每轮训练的混淆矩阵

    rec = Cm[(0, 0)] / (Cm[(0, 0)] + Cm[(1, 0)])  # 召回率

    acc = (Cm[(0, 0)] + Cm[(1, 1)]) / \
          (Cm[(0, 0)] + Cm[(0, 1)] +
           Cm[(1, 0)] + Cm[(1, 1)])  # 准确率

    pre = Cm[(0, 0)] / (Cm[(0, 0)] + Cm[(0, 1)])  # 精确率
    F1 = (2 * pre * rec) / (pre + rec)
    print("epoch: {}, loss: {}, rec: {}, acc: {}, pre: {}, F1: {}".format(epoch, loss / len(train_loader), rec, acc, pre, F1))
    loss_list[epoch] = loss / len(train_loader)
    rec_list[epoch] = rec
    acc_list[epoch] = acc
    pre_list[epoch] = pre
    F1_list[epoch] = F1
    return Cm

def test():
    model.eval()
    total_loss = 0
    rec = 0
    acc = 0
    pre = 0
    F1 = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            # 正常传播
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)

            loss = criterion(outputs, labels)
            total_loss += loss
            pred = torch.argmax(outputs, dim=1)

            for j in range(100):
                y_true.append(labels[j].item())
                y_pred.append(pred[j].item())

        Cm = confusion_matrix(y_true, y_pred)  # 每轮训练的混淆矩阵

        rec = Cm[(0, 0)] / (Cm[(0, 0)] + Cm[(1, 0)])  # 召回率

        acc = (Cm[(0, 0)] + Cm[(1, 1)]) / \
              (Cm[(0, 0)] + Cm[(0, 1)] +
               Cm[(1, 0)] + Cm[(1, 1)])  # 准确率

        pre = Cm[(0, 0)] / (Cm[(0, 0)] + Cm[(0, 1)])  # 精确率
        F1 = (2 * pre * rec) / (pre + rec)
        print("epoch: {}, loss: {}, rec: {}, acc: {}, pre: {}, F1: {}".format(epoch, loss / len(train_loader), rec, acc,
                                                                              pre, F1))
        loss_list[epoch] = loss / len(train_loader)
        rec_list[epoch] = rec
        acc_list[epoch] = acc
        pre_list[epoch] = pre
        F1_list[epoch] = F1


for epoch in range(2):
    cm = train()

torch.save(model.state_dict(), "model/{}.pth".format(2))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文

plot_confusion_matrix(cm, ["0负面", "1正面"], "Confusion Matrix")
plt.show()

plt.subplot(2, 2, 1)
plt.plot(x_list, rec_list)
plt.xlabel("epoch")
plt.ylabel("rec")
plt.title("召回率")

plt.subplot(2, 2, 2)
plt.plot(x_list, acc_list)
plt.xlabel("epoch")
plt.ylabel("acc")
plt.title("准确率")

plt.subplot(2, 2, 3)
plt.plot(x_list, pre_list)
plt.xlabel("epoch")
plt.ylabel("pre")
plt.title("精确率")

plt.subplot(2, 2, 4)
plt.plot(x_list, F1_list)
plt.xlabel("epoch")
plt.ylabel("F1")
plt.title("F1")

plt.show()

plt.plot(x_list, loss_list)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("损失")
plt.show()