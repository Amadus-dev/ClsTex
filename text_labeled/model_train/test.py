import jieba
import json
model_config_path = './model_config.json'
model_config = json.load(open(model_config_path, 'r', encoding='utf-8'))
model_list = []
for model in model_config.keys():
    model_list.append(model)
result = [['影视', 0.920291483], ['美妆', 0.889144], ['明星', 0.1132456], ['时尚', 0.0154655]]
strings = ''
for item in result:
    strings += item[0]+':'+str(item[1])+'\n'

print(strings)
print(model_list)