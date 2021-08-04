# 导入用于对象保存与加载的joblib
import joblib
# 导入keras中的词汇映射器Tokenizer
import tensorflow.keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
# 导入从样本csv到内存的get_data_labels函数
from data_analysis import get_data_labels
import numpy as np
# 首先导入keras构建模型的必备工具包
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
# 导入作图工具包matplotlib
import matplotlib.pyplot as plt


cutlen=70
# 根据样本集最大词汇数选择最大特征数，应大于样本集最大词汇数
max_features = 25000
# n-gram特征的范围，一般选择为2
ngram_range = 2
# 定义词嵌入维度为50
embedding_dims = 50
# 最大对其长度，即输入矩阵中每条向量的长度
maxlen = 139
# 最大特征数, 即输入矩阵中元素的最大值
new_max_features = 159393
# batch_size是每次进行参数更新的样本数量
batch_size = 32

# epochs将全部数据遍历训练的次数
epochs = 40

def word_map(csv_path, tokenizer_path, cut_num):
    """进行词汇映射，以训练数据的csv路径和映射器存储路径以及截断数为参数"""
    # 使用get_data_labels函数获取简单处理后的训练数据和标签
    train_data, train_labels = get_data_labels(csv_path)
    # 进行正负样本均衡切割, 使其数量比例为1:1
    train_data = train_data[:-cut_num]
    train_labels = train_labels[:-cut_num]
    # 实例化一个词汇映射器对象
    t = Tokenizer(num_words=None, char_level=False)
    # 使用映射器拟合现有文本数据
    t.fit_on_texts(train_data)
    # 使用joblib工具保存映射器
    joblib.dump(t, tokenizer_path)
    # 使用映射器转化现有文本数据
    x_train = t.texts_to_sequences(train_data)
    # 获得标签数据
    y_train = train_labels
    return x_train, y_train

def padding(x_train, cutlen=70):
    return sequence.pad_sequences(x_train, cutlen)

def create_ngram_set(input_list, ngram_value=2):
    """
    从列表中提取n-gram特征
    #>>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))

def get_ti_and_nmf(x_train, ti_path, ngram_range):
    """从训练数据中获得token_indice和新的max_features"""
    # >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    # 创建一个盛装n-gram特征的集合.
    ngram_set = set()
    # 遍历每一个数值映射后的列表
    for input_list in x_train:
        # 遍历可能存在2-gram, 3-gram等
        for i in range(2, ngram_range + 1):
            # 获得对应的n-gram表示
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            # 更新n-gram集合
            ngram_set.update(set_of_ngram)

    # 去除掉(0, 0)这个2-gram特征
    ngram_set.discard(tuple([0] * ngram_range))
    # 将n-gram特征映射成整数.
    # 为了避免和之前的词汇特征冲突，n-gram产生的特征将从max_features+1开始
    start_index = max_features + 1
    # 得到对n-gram表示与对应特征值的字典
    token_indice = {v:k + start_index for k, v in enumerate(ngram_set)}
    # 将token_indice写入文件以便预测时使用
    with open(ti_path, "w") as f:
        f.write(str(token_indice))
    # token_indice的反转字典，为了求解新的最大特征数
    indice_token = {token_indice[k]: k for k in token_indice}
    # 获得加入n-gram之后的最大特征数
    new_max_features = np.max(list(indice_token.keys())) + 1
    return token_indice, new_max_features

def add_ngram(sequences, token_indice, ngram_range=2):
    """
    将n-gram特征加入到训练数据中
    如: adding bi-gram
    #>>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    #>>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    #>>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
    """

    new_sequences = []
    # 遍历序列列表中的每一个元素作为input_list, 即代表一个句子的列表
    for input_list in sequences:
        # copy一个new_list
        new_list = input_list[:].tolist()
        # 遍历n-gram的value，至少从2开始
        for ngram_value in range(2, ngram_range + 1):
            # 遍历各个可能的n-gram长度
            for i in range(len(new_list) - ngram_value + 1):
                # 获得input_list中的n-gram表示
                ngram = tuple(new_list[i : i+ngram_value])
                # 如果在token_indice中，则追加相应的数值特征
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])

        new_sequences.append(new_list)

    return np.array(new_sequences)

def align(x_train):
    """用于向量按照最长长度进行补齐"""
    # 获得所有句子长度的最大值
    maxlen = max(list(map(lambda x: len(x), x_train)))
    # 调用padding函数
    x_train = padding(x_train, maxlen)
    return x_train, maxlen


def model_build(maxlen, new_max_features):
    """该函数用于模型结构构建"""
    # 在函数中，首先初始化一个序列模型对象
    model = Sequential()
    # 然后首层使用Embedding层进行词向量映射
    model.add(Embedding(new_max_features, embedding_dims, input_length=maxlen))
    # 然后用构建全局平均池化层，减少模型参数，防止过拟合
    model.add(GlobalAveragePooling1D())
    # 最后构建全连接层 + sigmoid层来进行分类.
    model.add(Dense(1, activation='sigmoid'))

    return model

def model_compile(model):
    """用于选取模型的损失函数和优化方法"""
    # 使用model自带的compile方法，选择预定义好的二分类交叉熵损失函数，Adam优化方法，以及准确率评估指标.
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def model_fit(model, x_train, y_train):
    """用于模型训练"""
    history = model.fit(x=x_train,
                        y=y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.1)
    return history


def plot_loss_acc(history:tensorflow.keras.Sequential, acc_png_path, loss_png_path):
    """用于绘制模型的损失和acc对照曲线, 以模型训练历史为参数"""
    # 首先获得模型训练历史字典，
    # 形如{'val_loss': [0.8132099324259264, ..., 0.8765081824927494],
    #    'val_acc': [0.029094827586206896,...,0.13038793103448276],
    #     'loss': [0.6650978644232184,..., 0.5267722122513928],
    #     'acc': [0.5803400383141762, ...,0.8469827586206896]}

    history_dict = history.history
    # 取出需要的的各个key对应的value，准备作为纵坐标
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    # 取epochs的递增列表作为横坐标
    epochs = range(1, len(acc) + 1)

    # 绘制训练准确率的点图
    plt.plot(epochs, acc, 'bo', label='Training acc')
    # 绘制验证准确率的线图
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    # 增加标题
    plt.title('Training and validation accuracy')
    # 增加横坐标名字
    plt.xlabel('Epochs')
    # 增加纵坐标名字
    plt.ylabel('Accuracy')
    # 将上面的图放在一块画板中
    plt.legend()
    # 保存图片
    plt.savefig(acc_png_path)

    # 清空面板
    plt.clf()
    # 绘制训练损失的点图
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # 绘制验证损失的线图
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    # 添加标题
    plt.title('Training and validation loss')
    # 添加横坐标名字
    plt.xlabel('Epochs')
    # 添加纵坐标名字
    plt.ylabel('Loss')
    # 把两张图放在一起
    plt.legend()
    # 保存图片
    plt.savefig(loss_png_path)

def model_save(save_path, model):
    """模型保存函数"""
    # 使用model.save对模型进行保存.
    model.save(save_path)
    print('Model saved')
    return

def model_load(save_path, sample):
    """模型加载与预测函数"""
    # 使用load_model方法进行加载
    model = load_model(save_path)
    # 使用predict方法进行预测
    return  model.predict(sample)

if __name__ == '__main__':
    # 对应的样本csv路径
    csv_path = './movie/sample.csv'
    # 词汇映射器保存的路径
    tokenizer_path = './movie/Tokenizer'
    # 截断数
    cut_num = 422

    # token_indice的保存路径
    ti_path = "./movie/token_indice"
    # 模型训练的历史对象history

    # 准确率对照曲线存储路径
    acc_png_path = "./movie/acc.png"

    # 损失对照曲线存储路径
    loss_png_path = "./movie/loss.png"

    # 模型的保存路径
    save_path = "./movie/model.h5"



    x_train, y_train = word_map(csv_path, tokenizer_path, cut_num)
    x_train = padding(x_train)
    token_indice, new_max_features = get_ti_and_nmf(x_train, ti_path, ngram_range)
    new_sequences = add_ngram(x_train, token_indice, ngram_range)
    x_train, maxlen = align(new_sequences)
   # print(x_train, '\n', maxlen, '\n', new_max_features)

    model = model_build(maxlen=maxlen, new_max_features=new_max_features)
    model = model_compile(model)
    history = model_fit(model, x_train, y_train)
    plot_loss_acc(history=history, acc_png_path=acc_png_path, loss_png_path=loss_png_path)
    model_save(save_path=save_path, model=model)

    # 一个实例: 训练数据的第一条
    sample = np.array([x_train[0]])
    result = model_load(save_path, sample)
    print(result)
