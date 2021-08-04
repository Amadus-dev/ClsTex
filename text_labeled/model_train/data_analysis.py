import pandas as pd
import jieba
import os
from collections import Counter
from pylab import mpl
from itertools import chain

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def get_data_labels(csv_path):
    """获得训练数据和对应的标签, 以正负样本的csv文件路径为参数"""
    # pandas读取csv文件至内存
    df = pd.read_csv(csv_path, header=None, sep="\t")
    # 对句子进行分词处理并且过滤掉长度为1的词
    train_data = list(map(lambda x: list(filter(lambda x: len(x)>1, jieba.lcut(x))), df[1].values))
    # 取第0列的值作为训练标签
    train_labels = df[0].values
    return train_data, train_labels

def pic_show(pic, pic_path, pic_name):
    """用于图片显示，以图片对象和预保存的路径为参数"""
    if not os.path.exists(pic_path):os.mkdir(pic_path)
    pic.savefig(os.path.join(pic_path, pic_name))
    print("请通过地址http://localhost/text_labeled/model_train" + pic_path[1:] + pic_name + "查看.")

def get_labels_distribution(train_labels, pic_path, pic_name='ld.png'):
    """获取正负样本数量的基本分布情况"""
    # class_dict >>> {1: 3995, 0: 4418}
    class_dict = dict(Counter(train_labels))
    print(class_dict)
    df = pd.DataFrame(list(class_dict.values()), list(class_dict.keys()))
    pic = df.plot(kind='bar', title='类别分布图').get_figure()
    pic_show(pic, pic_path, pic_name)

def get_sentence_length_distribution(train_data, pic_path, pic_name='sld.png'):
    """该函数用于获得句子长度分布情况"""
    sentence_len_list = list(map(len, train_data))
    # len_dict >>> {38: 62, 58: 18, 40: 64, 35: 83,....}
    len_dict = dict(Counter(sentence_len_list))
    len_list = list(zip(len_dict.keys(), len_dict.values()))
    # len_list >>> [(1, 3), (2, 20), (3, 51), (4, 96), (5, 121), (6, 173), ...]
    len_list.sort(key=(lambda x: x[0]))
    df = pd.DataFrame(list(map(lambda x:x[1], len_list)), list(map(lambda x:x[0], len_list)))
    ax = df.plot(kind='bar', figsize=(18, 18), title='句子长度分布图')
    ax.set_xlabel('句子长度')
    ax.set_ylabel('该长度出现的次数')
    pic = ax.get_figure()
    pic_show(pic, pic_path, pic_name)

def get_word_frequency_distribution(train_data, pic_path, pic_name='wfd.png'):
    """该函数用于获得词频分布"""
    vocab_size = len(set(chain(*train_data)))
    print('所有样本共包含不同词汇数量为：', vocab_size)
    # 获取常见词分布字典，以便进行绘图
    # common_word_dict >>> {'电影': 1548, '自己': 968, '一个': 850, '导演': 757, '现场': 744, ...}
    common_word_dict = dict(Counter(chain(*train_data)).most_common(50))
    df = pd.DataFrame(list(common_word_dict.values()), list(common_word_dict.keys()))
    pic = df.plot(kind='bar', figsize=(18, 18), title='常见词分布图').get_figure()
    pic_show(pic, pic_path, pic_name)


if __name__ == '__main__':
    # 样本csv文件路径
    csv_path = "./star/sample.csv"
    train_data, train_labels = get_data_labels(csv_path)
    # get_data_labels函数得到的train_labe
    # 图片的存储路径
    pic_path = './star/'
    # 图片的名字， 默认是ld.png
    pic_name = 'wld.png'
    #get_labels_distribution(train_labels, pic_path, pic_name)
    #get_sentence_length_distribution(train_data, pic_path, pic_name)
    get_word_frequency_distribution(train_data, pic_path, pic_name)