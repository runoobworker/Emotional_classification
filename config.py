import torch
class Config():
    def __init__(self):
        # dim_feedforward,  dropout, max_len
        self.vocab_size = 10002  # 字典大小
        self.embedding_dim = 128  # embedding大小
        self.hidden_dim = 128  # 隐藏层大小
        self.d_model = 128  # 维度
        self.num_head = 4  # 头
        self.num_encoder_layer = 3  # encoder层数
        # self.num_decoder_layer = 3   # decoder层数
        self.dim_feedforward = 512  # 前向函数参数
        self.max_len = 0  # 序列最长值存储
        self.dropout = 0.5  # 参数，防止过拟合
        self.num_class = 6  # 六分类
        self.batch_size = 128  # batch
        self.is_shuffle = True  # 打乱
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = 0.001  # 学习率
        self.epoch = 100  # 训练次数