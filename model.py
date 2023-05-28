import torch
import torch.nn.functional as F
import numpy as np

class Model(torch.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = torch.nn.Embedding(config.n_vocab,  # 字典大小
                                            config.embedding_size,
                                            padding_idx=0)  # 填充id:<PAD>
        self.lstm = torch.nn.LSTM(config.embedding_size,    # 输入维度
                                  config.hidden_size,       # 隐藏层大小
                                  config.num_layers,        # 隐层层数
                                  bidirectional=True,       # 双向LSTM
                                  batch_first=True,         # input = (batch, seq, input_size)
                                  dropout=config.dropout)   # 随机抑制节点，防止过拟合
        self.maxpool = torch.nn.MaxPool1d(config.pad_size)  # 1维池化
        self.fc = torch.nn.Linear(config.hidden_size * 2 + config.embedding_size,  # 双向
                                  config.num_class)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        embed = self.embedding(x)  # (bacth_size, seq, embedding_size)
        out, _ = self.lstm(embed)
        out = F.relu(out)                   # 激活函数
        out = torch.cat((embed, out), 2)   # 拼接,inputs,dim
        out = out.permute(0, 2, 1)         # 交换维度（转置）
        out = self.maxpool(out).reshape(out.size()[0], -1)  # 转化为2维
        out = self.fc(out)                  # 全连接层
        out = self.softmax(out)             # 分类
        return out

if __name__ == "__main__":   # 测试
    from  config import Config
    cfg = Config()
    cfg.pad_size = 640
    model_textcls = Model(config=cfg)

    input_tensor = torch.tensor([i for i in range(640)]).reshape([1, 640])
    out_tensor = model_textcls.forward(input_tensor)
    print(out_tensor.size())
    print(out_tensor)
