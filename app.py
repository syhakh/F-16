#! -*- coding:utf-8 -*-

import json
import numpy as np
import pandas as pd
from random import choice
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import re, os
import codecs


from keras.layers import *
from keras.models import Model
from keras.models import load_model
import keras.backend as K
from keras.optimizers import Adam
import tensorflow as tf
from keras import backend as K

sess = K.get_session() # 建立好的session 会话



maxlen = 100
learning_rate = 5e-5
min_learning_rate = 1e-5
config_path = './BERT/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './BERT/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './BERT/chinese_L-12_H-768_A-12/vocab.txt'


token_dict = {}

with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R

tokenizer = OurTokenizer(token_dict)


neg = pd.read_excel('data/neg.xls', header=None)
pos = pd.read_excel('data/pos.xls', header=None)

data = []

for d in neg[0]:
    data.append((d, 0))  # 添加上标签


for d in pos[0]:
    data.append((d, 1))


# 按照9:1的比例划分训练集和验证集
random_order = list(range(len(data)))
np.random.shuffle(random_order)
train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]
train_data = train_data[:500]
valid_data = valid_data[:100]

# 标签判别输出
def predict_label(s): # s代表输入的文本
    model.load_weights('model.h5')
    x1, x2 = tokenizer.encode(s)
    label = model.predict([  np.array([x1]), np.array([x2]) ])
    if label[0][0]< 0.5:
         return '这是一个坏评价'
    else:
         return '这是一个好评价'


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y]) # ？？？
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)  # ？？？
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []





bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

for l in bert_model.layers:
    l.trainable = True

x1_in = Input(shape=(None,))
x2_in = Input(shape=(None,))

x = bert_model([x1_in, x2_in]) # (?,?,768)
x = Lambda(lambda x: x[:, 0])(x) # (?,768)
p = Dense(1, activation='sigmoid')(x) # (?,1)

model = Model([x1_in, x2_in], p)
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(1e-5), # 用足够小的学习率
    metrics=['accuracy']
)
model.summary()



train_D = data_generator(train_data)
valid_D = data_generator(valid_data)



model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=len(train_D),
    epochs=1,
    validation_data=valid_D.__iter__(),
    validation_steps=len(valid_D)
)
model.save('model.h5')
graph = tf.get_default_graph()



from flask import Flask, render_template, request, redirect, url_for
import numpy as np

app = Flask(__name__)

'''
加载模型和预测模块
'''
import tensorflow as tf
from keras.models import load_model
import keras.backend.tensorflow_backend as K
# 程序开始时声明


@app.route('/success/<name>')  # 定义路由规则 metheds ,允许访问的请求方式 类型为可迭代对象,允许八种http请求方式
def success(name):
    with graph.as_default():
        model.load_weights('model.h5')
        x1, x2 = tokenizer.encode(name)
        label = model.predict([np.array([x1]), np.array([x2])])
        if label[0][0] < 0.5:
            return '这是一个坏评价'
        else:
            return '这是一个好评价'






@app.route('/login',methods = ['POST', 'GET'])
def login():
   if request.method == 'POST':
      user = request.form['nm']  # 文本框的输入
      return redirect(url_for('success', name=user))
   else:
      user = request.args.get('nm')
      return redirect(url_for('success', name=user))




if __name__ == '__main__':
    app.run()





