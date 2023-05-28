from torch.utils.data import DataLoader, Dataset
import jieba
import numpy as np
from config import Config

def read_dict(voc_dict_patth):
    # 读字典
    voc_dict = {}
    dict_list = open(voc_dict_patth, encoding="utf-8").readlines()
    for item in dict_list:
        item = item.split("，")
        voc_dict[item[0]] = int(item[1].strip())
    return voc_dict

def load_data(data_path, data_stop_path):
    #加载数据

    data_list = open(data_path, encoding='utf-8').readlines()[1:]  # 构建数据列表去掉第一行
    stops_word = open(data_stop_path, encoding='utf-8').readlines()  # 停用词
    stops_word = [line.strip() for line in stops_word]  # 去掉换行符,strip()移除字符串头尾指定的字符（默认为空格或换行符）
    stops_word.append(" ")  # 停用词添加空格
    stops_word.append("\n")  # 停用词添加换行符
    max_len_seq = 0
    data = []
    for item in data_list:
        label = item[0]
        content = item[2:].strip()  # 去掉换行符
        seg_list = jieba.cut(content, cut_all=False)  # jieba库进行分词，采用精确分词

        seg_res = []  # 存储分词结果
        for seg_item in seg_list:
            if seg_item in stops_word:  # 分词在
                continue
            seg_res.append(seg_item)

        if max_len_seq < len(seg_res):
            max_len_seq = len(seg_res) + 4  # 最大序列长度


        data.append([label, seg_item])
    # print(max_len_seq)
    return data, max_len_seq

class text_ClS(Dataset):
    def __init__(self, data_path, data_stop_path, voc_dict_patch):
        self.data_path = data_path
        self.data_stop_path = data_stop_path
        self.voc_dict = read_dict(voc_dict_patch)
        self.data, self.max_len_seq = load_data(self.data_path, self.data_stop_path)

        np.random.shuffle(self.data)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        label = int(data[0])
        word_list = data[1]
        input_idx = []
        # 在字典中则用相应值代替，不在则同<UNK>
        for word in word_list:
            if word in self.voc_dict.keys():
                input_idx.append(self.voc_dict[word])
            else:
                input_idx.append(self.voc_dict["<UNK>"])
        # 长度不一用<PAD>
        if len(input_idx) < self.max_len_seq:
            input_idx += [self.voc_dict["<PAD>"]
                          for _ in range(self.max_len_seq - len(input_idx))]

        data = np.array(input_idx)
        return label, data

def data_loader(dataset, config):
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=config.is_shuffle, drop_last=True)