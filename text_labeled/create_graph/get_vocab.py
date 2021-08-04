import os
import jieba
import jieba.posseg as pseg

# 使用jieba中的词性标注功能

# jieba中预定义的名词性类型,分别表示: 人名，名词，地名，机构团体名，其他专有名词
n_e = ["nr", "n", "ns", "nt", "nz"]
# 写入csv的路径
csv_path = "./labels"
# 用户自定义词典路径
userdict_path = "../userdict.txt"
def get_vocabulary(article_path, csv_name):
    """函数将读取path路径下的所有文章文本,并转化为词汇写入词汇csv文件"""
    if not os.path.exists(article_path): return
    if not os.path.exists(csv_path): os.mkdir(csv_path)
    def _get_n_list(text):
        """用于获取名词列表"""
        # 使用jieba的词性标注方法切分文本,获得具有词性属性flag和词汇属性word的对象,
        # 从而判断flag是否在我们定义的名词性列表中,来返回对应的词汇
        r = []
        for g in pseg.lcut(text):
            if g.flag in n_e:
                r.append(g.word)
        return r

    with open(os.path.join(csv_path, csv_name), "a") as u:
        for article in os.listdir(article_path):
            with open(os.path.join(article_path, article), mode="r", encoding='utf-8') as f:
                text = f.read()
            # 只获取长度大于等于2的名词
            n_list = list(filter(lambda x:len(x)>=2, set(_get_n_list(text))))
            list(map(lambda x: u.write(x + "\n"), n_list))

    with open(os.path.join(csv_path, csv_name), "r") as o:
        word = o.read()
        with open(userdict_path, "a") as f:
            f.write(word)
    return

if __name__ == '__main__':
    get_vocabulary(article_path='./fashion', csv_name='时尚.csv')