# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 08:43:14 2021

@author: qg
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
#print(tf.__version__)
#导入数据集
imdb=keras.datasets.imdb
(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000)
#参数 num_words=10000 会保留训练数据中出现频次在前 10000 位的字词。为确保数据规模处于可管理的水平，罕见字词将被舍弃。
#查看数据格式
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
#影评文本已转换为整数，其中每个整数都表示字典中的一个特定字词
#print(train_data[0])
#影评的长度可能会有所不同。以下代码显示了第一条和第二条影评中的字词数
#print(len(train_data[0]), len(train_data[1]))
'''
将数据集中的整数转换回单词
我们将创建一个辅助函数word_index来查询包含整数到字符串映射的字典对象：
'''
# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
#print(decode_review(train_data[0])) # 函数显示第一条影评的文本

'''
我们可以填充数组，使它们都具有相同的长度，然后创建一个形状为 max_length * num_reviews 
的整数张量。我们可以使用一个能够处理这种形状的嵌入层作为网络中的第一层。
'''
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
print(len(train_data[0]), len(train_data[1])) #样本长度
print(train_data[0])

'''
构建模型
神经网络通过堆叠层创建而成，这需要做出两个架构方面的主要决策：
1、要在模型中使用多少个层？
2、要针对每个层使用多少个隐藏单元？
在本示例中，输入数据由字词-索引数组（word-index）构成。要预测的标签是 0 或 1（好or坏）。
接下来，我们为此问题构建一个模型：
第一层是 Embedding 层。该层会在整数编码的词汇表中查找每个字词-索引的嵌入向量。
模型在接受训练时会学习这些向量。这些向量会向输出数组添加一个维度。
生成的维度为：(batch, sequence, embedding)。
接下来，一个 GlobalAveragePooling1D 层通过对序列维度求平均值，
针对每个样本返回一个长度固定的输出向量。
这样，模型便能够以尽可能简单的方式处理各种长度的输入。该
长度固定的输出向量会传入一个全连接 (Dense) 层（包含 16 个隐藏单元）。
最后一层与单个输出节点密集连接。应用 sigmoid 激活函数后，结果是介于 0 到 1 之间的浮点值，表示概率或置信水平。
隐藏单元
上述模型在输入和输出之间有两个中间层（也称为“隐藏”层）。输出（单元、节点或神经元）
的数量是相应层的表示法空间的维度。换句话说，该数值表示学习内部表示法时网络所允许的自由度。
如果模型具有更多隐藏单元（更高维度的表示空间）和/或更多层，则说明网络可以学习更复杂的表示法。
不过，这会使网络耗费更多计算资源，并且可能导致学习不必要的模式（
可以优化在训练数据上的表现，但不会优化在测试数据上的表现）。这称为过拟合，我们稍后会加以探讨。

'''
# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()
'''

模型在训练时需要一个损失函数和一个优化器。
由于这是一个二元分类问题且模型会输出一个概率
（应用 S 型激活函数的单个单元层），因此我们将使用 binary_crossentropy 损失函数。
该函数并不是唯一的损失函数，例如，您可以选择 mean_squared_error。
但一般来说，binary_crossentropy 更适合处理概率问题，它可测量概率分布之间的“差距”，
在本例中则为实际分布和预测之间的“差距”。稍后，在探索回归问题（比如预测房价）时，
我们将了解如何使用另一个称为均方误差的损失函数。现在，配置模型以使用优化器和损失函数：
'''
model.compile(optimizer = tf.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['acc'])


'''
创建验证集
在训练时，我们需要检查模型处理从未见过的数据的准确率。
我们从原始训练数据中分离出 10000 个样本，创建一个验证集。
（为什么现在不使用测试集？我们的目标是仅使用train数据开发和调整模型，然后仅使用一次测试数据评估准确率。）
'''

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]


'''
训练模型
用有 512 个样本的小批次训练模型 40 个周期。这将对 x_train 和 y_train 张量中的所有样本进行 40 次迭代。
在训练期间，监控模型在验证集的 10000 个样本上的损失和准确率：
model.fit() fit详解
x：输入数据。如果模型只有一个输入，那么x的类型是numpyarray，如果模型有多个输入，那么x的类型应当为list，list的元素是对应于各个输入的numpy array
y：标签，numpy array
batch_size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。
epochs：整数，训练终止时的epoch值，训练将在达到该epoch值时停止，当没有设置initial_epoch时，它就是训练的总轮数，否则训练的总轮数为epochs - inital_epoch
verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
validation_data：形式为（X，y）的tuple，是指定的验证集。此参数将覆盖validation_spilt。
'''
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=80,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)
'''
评估模型
我们来看看模型的表现如何。模型会返回两个值：test_data损失（表示误差的数字，越低越好）和test_labels准确率。
'''
results = model.evaluate(test_data, test_labels)
print(results)

'''
创建准确率和损失随时间变化的图
model.fit() 返回一个 History 对象，该对象包含一个字典，其中包括训练期间发生的所有情况：
一共有 4 个条目：每个条目对应训练和验证期间的一个受监控指标。
我们可以使用这些指标绘制训练损失与验证损失图表以进行对比，并绘制训练准确率与验证准确率图表：
'''

history_dict = history.history
history_dict.keys()
#dict_keys(['loss', 'val_loss', 'val_acc', 'acc'])

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

'''
在该图表中，圆点表示训练损失和准确率，实线表示验证损失和准确率。
可以注意到，训练损失随着周期数的增加而降低，训练准确率随着周期数的增加而提高。
在使用梯度下降法优化模型时，这属于正常现象 - 该方法应在每次迭代时尽可能降低目标值。
验证损失和准确率的变化情况并非如此，它们似乎在大约 20 个周期后达到峰值。
这是一种过拟合现象：模型在训练数据上的表现要优于在从未见过的数据上的表现。
在此之后，模型会过度优化和学习特定于训练数据的表示法，而无法泛化到测试数据。对于这种特殊情况，
我们可以在大约 20 个周期后停止训练，防止出现过拟合
'''
