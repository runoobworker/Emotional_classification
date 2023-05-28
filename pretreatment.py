import jieba

data_path = "D:/桌面/Emotional_classification/data/simplifyweibo_4_moods.csv"
data_stop_path = "D:/桌面/Emotional_classification/data/hit_stopwords.txt"
data_list = open(data_path, encoding='utf-8').readlines()[1:]  # 构建数据列表去掉第一行
stops_word = open(data_stop_path, encoding='utf-8').readlines()                  # 停用词
stops_word = [line.strip() for line in stops_word]             # 去掉换行符,strip()移除字符串头尾指定的字符（默认为空格或换行符）
stops_word.append(" ")                                         # 停用词添加空格
stops_word.append("\n")                                        # 停用词添加换行符


voc_dict = {}  # 定义空字典
min_seq = 1
top_n = 10000
UNK = "<UNK>"
PAD = "<PAD>"
for item in data_list:
    label = item[0]
    content = item[2:].strip()  # 去掉换行符
    seg_list = jieba.cut(content, cut_all=False)  # jieba库进行分词，采用精确分词

    seg_res = []  # 存储分词结果
    for seg_item in seg_list:
        if seg_item in stops_word:  # 分词在
            continue
        seg_res.append(seg_item)
        # 根据分词构建词典
        if seg_item in voc_dict.keys():
            voc_dict[seg_item] = voc_dict[seg_item] + 1  # 在字典中词频加1
        else:
            voc_dict[seg_item] = 1                       # 不在则置1

    # print(content)
    # print(seg_res)

voc_dict = sorted([_ for _ in voc_dict.items() if _[1] > min_seq],
                      key=lambda x:x[1],                 # 匿名函数
                      reverse=True)[:top_n]              # 降序排序，且只取排名3000的词


voc_dict = {word_count[0]: idx + 2 for idx, word_count in enumerate(voc_dict)}  # 根据排序重新构建词典
voc_dict.update({UNK:1, PAD:0})  # 字典以外的字符作为UNK，padding添0

# print(voc_dict)
# 存储词典
ff = open("D:/桌面/Emotional_classification/data/dict", "w", encoding='utf-8')
for i in voc_dict.keys():
    ff.writelines("{}，{}\n".format(i, voc_dict[i]))
ff.close()