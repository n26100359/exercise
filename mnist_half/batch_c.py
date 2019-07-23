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
log_path = './logs/batch_c'
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

def calculate_lr(num_of_batch):
    if num_of_batch <= 1600:
        lr_base = 0.001
        lr_max = 0.005
        stepsize = 200
    elif num_of_batch > 1600 and num_of_batch <= 2200:
        lr_base = 0.0001
        lr_max = 0.0005
        stepsize = 100
    elif num_of_batch > 2200:
        lr_base = 0.00001
        lr_max = 0.00005
        stepsize = 50
    cycle = math.floor(1 + num_of_batch / (2 * stepsize))
    x = abs(num_of_batch / stepsize - 2 * cycle + 1)
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



for epoch in range(1, epochs + 1):
    #data shuffle
    index=np.arange(60000)
    np.random.shuffle(index)

    x_train = x_train[index]
    y_train = y_train[index]
    #split validation
    split_idx = math.floor(np.shape(x_train)[0] * 0.8)
    train_x = x_train[:split_idx]
    train_y = y_train[:split_idx]

    val_x = x_train[split_idx:]
    val_y = y_train[split_idx:]

    for batch in range(1, train_x.shape[0]//batch_size + 1):
        num_of_batch = (epoch-1) * (train_x.shape[0]/batch_size) + batch
        current_lr = calculate_lr(num_of_batch)
        K.set_value(model.optimizer.lr, current_lr)
        batch_train_x = train_x[(batch-1)*batch_size:batch*batch_size]
        batch_train_y = train_y[(batch-1)*batch_size:batch*batch_size]
        print(np.shape(batch_train_x))
        loss = model.train_on_batch(batch_train_x, batch_train_y)
        loss_list.append(loss)
        write_log(callback, train_names, loss, num_of_batch)
        print(f'batch: {num_of_batch}. loss: {loss}')
    validation_loss = model.evaluate(val_x, val_y)
    valloss_list.append(validation_loss)
    write_log(callback, val_names, validation_loss, epoch)
    print(f'epoch: {epoch}. val_loss: {validation_loss}')

model.save('batch_c.h5')
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
