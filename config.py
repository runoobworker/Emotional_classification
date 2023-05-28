import torch
class Config():
    def __init__(self):
        self.n_vocab = 1002          # 字典大小
        self.embedding_size = 128    # embedding大小
        self.hidden_size = 128       # 隐藏层大小
        self.num_layers = 5          # lstm层数
        self.dropout = 0.8           # lstm参数，防止过拟合
        self.num_class = 6          # 六分类
        self.pad_size = 32           # 池化大小
        self.batch_size = 256          # batch
        self.is_shuffle = True       # 打乱
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = 0.01              # 学习率
        self.epoch = 100            # 训练次数