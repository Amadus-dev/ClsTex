import json
import os

from django.conf import settings
from flask import Flask, request
from flask_cors import *
import jieba
import multithread_predict
app = Flask(__name__)
settings.configure()
root = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'model_config.json')
print(root)
model_config_path = './model_config.json'
model_config = json.load(open(root, 'r', encoding='utf-8'))
model_list = []
for model in model_config.keys():
    model_list.append(model)


@app.route('/v1/main_server/', methods=["POST"])
@cross_origin()
def main_server():
    text = request.form['text']
    print(text)
    word_list = jieba.lcut(text)
    print(word_list)
    result = multithread_predict.request_model_serve(word_list=word_list, model_list=model_list)
    print(result)
    result_str = ''
    #result = [['影视', 0.920291483], ['美妆', 0.889144], ['明星', 0.1132456], ['时尚', 0.0154655]]
    for item in result:
        result_str += item[0] + ':' + str(item[1]) + '\n'
    return result_str

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5556)