import os
import time
import json
import threading
import requests
import joblib
from movie_model_train import add_ngram
from movie_model_train import padding, model_load
from tensorflow.keras.models import load_model
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
import tensorflow as tf
if tf.executing_eagerly():
    tf.compat.v1.disable_eager_execution()



'''def to_savedmodel(h5_model_path, pb_model_path):
    """将h5模型转化成tensorflow的pb格式模型用于微服务"""

    model = load_model(h5_model_path)
    builder = saved_model_builder.SavedModelBuilder(pb_model_path)

    signature = predict_signature_def(inputs={'input': model.input[0]}, outputs={'income': model.input[0]})
    with tf.compat.v1.keras.backend.get_session() as sess:
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tag_constants.SERVING],
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
            }
        )
        builder.save()'''


# 定义模型配置路径，它指向一个json文件
model_config_path = './model_config.json'
# model_config.json形如 ：
# {"影视": ["/data/django-uwsgi/text_labeled/model_train/movie/Tokenizer", 60, 2,
#           "/data/django-uwsgi/text_labeled/model_train/movie/token_indice", 119,
#           "http://localhost:8501/v1/models/movie:predict"],
# "美妆": ["/data/django-uwsgi/text_labeled/model_train/beauty/Tokenizer", 75, 2,
#           "/data/django-uwsgi/text_labeled/model_train/beauty/token_indice", 119,
#           "http://localhost:8502/v1/models/beauty:predict"]}
# json文件中是一个字典，字典中的每个key是我们标签的中文字符，每个value是一个列表，
# 列表的第一项是特征处理词汇映射器的存储地址
# 第二项是特征处理语料的截断长度
# 第三项是n-gram取得n值
# 第四项是n-gram特征中token_indice的保存路径
# 第五项是最后的最大的对齐长度
# 第六项是该模型保存的地址

# 最终的模型预测结果

model_prediction = []

def fea_process(word_list, config_list):
    """对输入进行类似与训练前的特征处理过程"""
    # 读取设定好的配置
    tokenizer_path = config_list[0]
    cutlen = config_list[1]
    ngram_range = config_list[2]
    ti_path = config_list[3]
    maxlen = config_list[4]

    # 加载分词映射器
    t = joblib.load(tokenizer_path)
    x_train = t.texts_to_sequences([word_list])
    # 进行截断对齐
    x_train = padding(x_train, cutlen)
    # 获得n-gram映射文件
    with open(ti_path, "r", encoding='utf-8') as f:
        token_indice = eval(f.read())
    # 添加n-gram特征
    x_train = add_ngram(x_train, token_indice, ngram_range)
    # 进行最大长度对齐
    x_train = padding(x_train, maxlen)
    return x_train

def pred(word_list, model):
    root = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'model_config.json')
    # 将持久化的模型配置文件加载到内存
    model_config = json.load(open(root, 'r', encoding='utf-8'))
    # 根据名字选择对应的配置列表
    config_list = model_config[model]
    save_path = config_list[5]
    # 对数据进行特征处理
    x_train = fea_process(word_list, config_list)
    result = model_load(save_path, x_train)
    # 将该线程中获取的结果放到模型预测结果列表中
    model_prediction.append([model, result[0][0]])

    return model_prediction
'''    print(x_train.astype(dtype=float).shape)
    print(x_train.astype(dtype=float).tolist())
    # 封装成tf-serving需要的数据体
    data = {'instances': x_train.astype(dtype=float).tolist()}
    # 向刚刚封装的微服务发送请求
    res = requests.post(url=config_list[5], json=data)
    print(eval(res.text))'''


def request_model_serve(word_list, model_list):
    """该函数开启多线程请求封装好的模型微服务"""
    def _start_thread(pred, x, y):
        """开启预测线程, 以线程需要执行的函数和函数的输入为参数"""
        t = threading.Thread(target=pred, args=(x, y))
        t.start()
        return t

    # 遍历model_list, 调用开启线程函数_start_thread，会获得一个所有开启后的线程列表
    t_list = list(map(lambda model: _start_thread(pred, word_list, model), model_list))
    # 线程将逐一join操作等待所有线程完成
    t_list = list(map(lambda t: t.join(), t_list))
    # 最后过滤掉所有概率预测小于0.5的类别，返回结果
    result = list(filter(lambda x: x[1] >= 0.8, model_prediction))
    model_prediction.clear()
    return result

if __name__ == '__main__':
    # 分词列表
    word_list = ["霸王别姬", "是一部", "非常", "值得", "看的", "电影"]
    # 预请求的模型名称
    model = "影视"
    # 预请求的模型列表
    model_list = ["影视", "美妆"]
    print(request_model_serve(word_list, model_list=model_list))
