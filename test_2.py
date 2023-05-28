from transformers import BertTokenizer, BertModel, BertConfig
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
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
# df.info() 基本信息
# print(df['label'].value_counts())

x = list(df['review'])
y = list(df['label'])
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
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

class Bert_model(torch.nn.Module):
    def __init__(self, freeze_bert=False, hidden_size=192):
        super(Bert_model, self).__init__()
        config = BertConfig.from_pretrained('bert-base-chinese')
        config.update({'output_hidden_states': True})
        self.bert = BertModel.from_pretrained('bert-base-chinese', config=config)
        self.fc = torch.nn.Linear(hidden_size*4, 2)
        self.softmax = torch.nn.Softmax(dim=1)

        # 是否冻结bert
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        all_hidden_states = torch.stack(outputs[2])  # 输出转换成矩阵
        concat_last_4layers = torch.cat((all_hidden_states[-1], all_hidden_states[-2], all_hidden_states[-3], all_hidden_states[-4]), dim=1)  # 取最后四层输出
        cls_concat = concat_last_4layers[:, 0, :]
        result = self.fc(cls_concat)
        result = self.softmax(result)

        return result

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = Bert_model().to(device)
model.load_state_dict(torch.load("D:/桌面/Emotional_classification/Bert/model/2.pth"))
criterion = torch.nn.CrossEntropyLoss().to(device)

total_loss = 0
a0, a1 = 0, 0
b0, b1 = 0, 0

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

    a0 = Cm[(0, 0)]
    a1 = Cm[(0, 1)]
    b0 = Cm[(1, 0)]
    b1 = Cm[(1, 1)]

    rec = a0 / (a0 + b0)  # 召回率
    acc = (a0 + b1) / (a0 + a1 + b0 + b1)  # 准确率
    pre = a0 / (a0 + a1)  # 精确率
    F1 = (2 * pre * rec) / (pre + rec)

print("loss: {}, rec: {}, acc: {}, pre: {}, F1: {}".format( loss / len(train_loader), rec, acc,
                                                                              pre, F1))

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文

plot_confusion_matrix(Cm, ["0喜悦", "1愤怒", "2悲伤", "3恐惧", "4惊喜", "5无情绪"], "Confusion Matrix")
plt.show()