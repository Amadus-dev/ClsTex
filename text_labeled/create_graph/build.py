import random
import sys
from neo4j import GraphDatabase
from settings import LABEL_STRUCTURE,NEO4J_CONFIG
import os
import fileinput
csv_path = './labels'
def create_label_node_and_rel():
    """该函数用于创建标签树的节点和边"""
    _driver = GraphDatabase.driver(**NEO4J_CONFIG)
    with _driver.session() as session:
        # 删除所有Label节点以及相关联的边
        cypher = "MATCH(a:Label) DETACH DELETE a"
        session.run(cypher)
        def _create_node_rel(l: dict):
            """根据标签树结构中的每一个字典去创建节点和关系"""
            if not l: return
            # 遍历字典中的k,v即父子标签
            for k, v in l.items():
                # MERGE一个父标签节点
                cypher = "MERGE(a:Label{title:%r})" % (k)
                session.run(cypher)

                def __c(word):
                    """用于创建子标签节点以及与父标签之间的关系"""
                    cypher = "CREATE(a:Label{title:%r}) SET a.name=%r " \
                             "WITH a MATCH(b:Label{title:%r}) MERGE(b)-[r:Contain]-(a)" % (word, word, k)
                    session.run(cypher)
                # 遍历子标签列表
                list(map(__c, v))
        # 遍历标签树列表
        list(map(_create_node_rel, LABEL_STRUCTURE))


def create_vocabulary_node_and_rel():
    """该函数用于创建词汇节点和关系"""
    _driver = GraphDatabase.driver(**NEO4J_CONFIG)
    with _driver.session() as session:
        # 删除所有词汇节点及其相关的边
        cypher = "MATCH(a:Vocabulary) DETACH DELETE a"
        session.run(cypher)

        def _create_v_and_r(csv):
            """读取单个csv文件,并写入数据库创建节点并与对应的标签建立关系"""
            path = os.path.join(csv_path, csv)
            # 使用fileinput的FileInput方法从持久化文件中读取数据,
            # 并进行strip()操作去掉两侧空白符, 再通过set来去重.
            word_list = list(
                set(map(lambda x: x.strip(), fileinput.FileInput(path))))

            def __create_node(word):
                """创建csv中单个词汇的节点和关系"""
                # 定义词汇的初始化权重,即词汇属于某个标签的初始概率，
                # 因为词汇本身来自该类型文章，因此初始概率定义在0.5-1之间的随机数
                weight = round(random.uniform(0.5, 1), 3)
                # 使用cypher语句创建词汇节点,然后匹配这个csv文件名字至后四位即类别名，
                # 在两者之间MERGE一条有权重的边
                cypher = "CREATE(a:Vocabulary{name:%r}) WITH a " \
                         "MATCH(b:Label{title:%r}) MERGE(a)-[r:Related{weight:%f}]-(b)" % (word, csv[:-4], weight)
                session.run(cypher)
            # 遍历词汇列表
            list(map(__create_node, word_list))
        # 遍历标签列表
        label_list = os.listdir(csv_path)
        list(map(_create_v_and_r, label_list))