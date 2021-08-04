import os

# 限制句子的最小字符数和句子的最大字符数
MIN_LENGTH = 5
MAX_LENGTH = 500

def get_p_text_list(single_article_path):
    '''

    :param single_article_path:单篇文章的文本列表
    :return:
    '''
    with open(single_article_path, mode='r', encoding='utf-8') as f:
        text = f.read()
        # 去掉换行符，并以句号划分
        cl = text.replace('\n', '.').split('。')
        # 过滤掉长度范围之外的句子
        cl = list(filter(lambda x:MIN_LENGTH<len(x)<MAX_LENGTH, cl))

    return cl

def get_p_sample(a_path, p_path):
    """该函数用于获得正样本的csv, 以文章路径和正样本csv写入路径为参数"""
    if not os.path.exists(a_path): return
    if not os.path.exists(p_path): os.mkdir(p_path)
    # 以追加的方式打开预写入正样本的csv
    fp = open(os.path.join(p_path, "p_sample.csv"), mode='a', encoding='utf-8')
    # 遍历文章目录下的每一篇文章
    for u in os.listdir(a_path):
        cl = get_p_text_list(os.path.join(a_path, u))
        for clc in cl:
            fp.write("1" + "\t" + clc + "\n")
    fp.close()

def get_sample(p_path, n_path_csv_list: list):
    """该函数用于获取样本集包括正负样本, 以正样本csv文件路径和负样本csv文件路径列表为参数"""
    fp = open(os.path.join(p_path, "sample.csv"), "w", encoding='utf-8')
    with open(os.path.join(p_path, "p_sample.csv"), "r",encoding='utf-8') as f:
        text = f.read()
    # 先将正样本写入样本csv之中
    fp.write(text)
    # 遍历负样本的csv列表
    for n_p_c in n_path_csv_list:
        with open(n_p_c, "r", encoding='utf-8') as f:
            # 将其中的标签1改写为0
            text = f.read().replace("1", "0")
        # 然后写入样本的csv之中
        fp.write(text)
    fp.close()

if __name__ == '__main__':
    a_path = '../create_graph/beauty'
    p_path = './fashion'
    #get_p_sample(a_path, p_path)
    n_path_csv_list = ["./beauty/p_sample.csv","./movie/p_sample.csv", "./star/p_sample.csv"]
    get_sample(p_path, n_path_csv_list)
