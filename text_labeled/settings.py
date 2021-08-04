## Server Config File

# global config
NEO4J_CONFIG = dict({
    'uri': 'bolt://0.0.0.0:7687',
    'auth': ('neo4j', '123456'),
    'encrpted': False
})

# 必须使用扁平化的存储结构:
LABEL_STRUCTURE = [
    {
        "泛娱乐":[
            "明星",
            "时尚",
            "游戏",
            "影视",
            "音乐",
            "美妆"
        ]
    },
    {
        "游戏":[
            "LOL",
            "王者农药",
            "吃鸡"
        ],
        "影视":[
            "喜剧",
            "综艺",
            "科幻",
            "恐怖"
        ],
        "音乐":[
            "摇滚乐",
            "民谣",
            "Rap",
            "流行乐"
        ]
    }
]
