from __future__ import print_function
import sklearn
import keras
import math
import random
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
import numpy as np
import pandas as pd

tbCallBack = TensorBoard(log_dir='./logs',  # log 目录
                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
#                 batch_size=32,     # 用多大量的数据计算直方图
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=True, # 是否可视化梯度直方图
                 write_images=True,# 是否可视化参数
                 embeddings_freq=0, 
                 embeddings_layer_names=None, 
                 embeddings_metadata=None)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=2)
def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()
log_path = './logs/adam_c'
callback = TensorBoard(log_path)
callback.set_model(Sequential)
train_names = ['loss', 'acc']
val_names = ['val_loss', 'val_acc']

batch_size = 1920
num_classes = 10
num_classes2 = 392
epochs = 100
loss_list = []
valloss_list = []

def step_decay(epoch):
    if epoch <= 64:
        lr_base = 0.001
        lr_max = 0.005
        stepsize = 8
    elif epoch > 64 and epoch <= 88:
        lr_base = 0.0001
        lr_max = 0.0005
        stepsize = 4
    elif epoch > 88:
        lr_base = 0.00001
        lr_max = 0.00005
        stepsize = 2
    cycle = math.floor(1 + epoch / (2 * stepsize))
    x = abs(epoch / stepsize - 2 * cycle + 1)
    lr = lr_base + (lr_max - lr_base) * max(0, (1 - x))
    print('lr: %f' % lr)
    return lr

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

train_spilt = np.split(x_train, 2,axis=2)
y_train = train_spilt[1].reshape(60000, 392)
x_train = train_spilt[0].reshape(60000, 28, 14, 1)

test_spilt = np.split(x_test, 2, axis=2)
y_test = test_spilt[1].reshape(10000, 392)
x_test = test_spilt[0].reshape(10000, 28, 14, 1)

input_shape = (28, 14, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
y_train /= 255
y_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes2, activation='sigmoid'))

model.compile(loss='mse',
              optimizer=keras.optimizers.adam(),
              metrics=['accuracy'])


lrate = LearningRateScheduler(step_decay)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.2,
          callbacks=[tbCallBack, early_stopping,lrate])

model.save('adam_c.h5')
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
