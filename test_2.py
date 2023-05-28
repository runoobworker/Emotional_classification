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
model.load_state_dict(torch.load("D:/桌面/Emotional_classification/GPT/model/2.pth"))
criterion = torch.nn.CrossEntropyLoss().to(device)
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
print("loss: {}, rec: {}, acc: {}, pre: {}, F1: {}".format(loss / len(train_loader), rec, acc,
                                                                              pre, F1))

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文

plot_confusion_matrix(Cm, ["0负面", "1正面"], "Confusion Matrix")
plt.show()