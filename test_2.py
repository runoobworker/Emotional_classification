import torch
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, TensorDataset
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

# 原始文本输入
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

# 加载 BERT 模型和 tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')

class Bert_LSTM_Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, freeze_bert):
        super(Bert_LSTM_Model, self).__init__()
        config = BertConfig.from_pretrained('bert-base-chinese')
        config.update({'output_hidden_states': True})
        self.bert = BertModel.from_pretrained('bert-base-chinese', config=config)
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size * 2, output_size)
        self.softmax = torch.nn.Softmax(dim=1)

        # 是否冻结bert
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        encoded_layers = outputs[0]  # 获取 BERT 输出的编码层
        lstm_input = encoded_layers.squeeze(0)  # 移除批次维度
        lstm_output, _ = self.lstm(lstm_input)
        lstm_output = lstm_output[:, -1, :]  # 取 LSTM 的最后一个时间步的输出作为句子级别的表示
        output = self.fc(lstm_output)
        output = self.softmax(output)
        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 初始化 LSTM 模型
input_size = 768  # BERT 的输出维度为 768
hidden_size = 128  # LSTM 的隐藏层维度
output_size = 2  # 情感分类的类别数
num_layers = 3
model = Bert_LSTM_Model(input_size, hidden_size, output_size, num_layers, False).to(device)
model.load_state_dict(torch.load("D:/桌面/Emotional_classification/Bert-LSTM/model/2.pth"))

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss().to(device)

total_train_loss = 0
total_iter = len(train_loader)

rec = 0
acc = 0
pre = 0
F1 = 0

y_true = []
y_pred = []
with torch.no_grad():
    for batch in train_loader:

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        total_train_loss += loss.item()

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
print("loss: {}, rec: {}, acc: {}, pre: {}, F1: {}".format(loss / len(train_loader), rec, acc, pre,
                                                                        F1))

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文

plot_confusion_matrix(Cm, ["0负面", "1正面"], "Confusion Matrix")
plt.show()