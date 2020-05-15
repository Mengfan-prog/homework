#自定义卷积层与全连接层
#图像输入为100*100*3
#学习率衰减
#达到56%

import tensorflow as tf
from functools import partial

from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow_core.python.keras.callbacks import LearningRateScheduler
from math import pow, floor
from tensorflow.keras import layers, optimizers, datasets,Sequential
import random
import pathlib

epochs =50
learning_rate=0.0001
#定义学习率衰减函数
def scheduler(epoch):
    init_lrate = 0.001
    drop = 0.8
    epochs_drop = 2
    lrate = init_lrate * pow(drop, floor(1 + epoch) / epochs_drop)
    return lrate
change_Lr = LearningRateScheduler(scheduler)


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"



#训练数据
train_data_path = pathlib.Path('D:/document/Jupyter working directory/food-11/training')
train_image_paths = list(train_data_path.glob('*'))
train_image_paths = [str(path) for path in train_image_paths]  # 所有图片路径的列表
random.shuffle(train_image_paths)  # 打散
train_image_count = len(train_image_paths)
train_image_labels = [int(path.split('\\')[5].split('_')[0]) for path in train_image_paths]
#验证数据
valid_data_path = pathlib.Path('D:/document/Jupyter working directory/food-11/validation')
valid_image_paths = list(valid_data_path.glob('*'))
valid_image_paths = [str(path) for path in valid_image_paths]  # 所有图片路径的列表
# random.shuffle(all_image_paths)  # 打散
valid_image_count = len(valid_image_paths)
valid_image_labels = [int(path.split('\\')[5].split('_')[0]) for path in valid_image_paths]



train_ds = tf.data.Dataset.from_tensor_slices((train_image_paths,train_image_labels))

valid_ds = tf.data.Dataset.from_tensor_slices((valid_image_paths,valid_image_labels))




def load_and_preprocess_from_path_label(path,label):
    image = tf.io.read_file(path)  # 读取图片
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [100, 100])  # 原始图片大小为(512, 512, 3)，重设为(32, 32)
    image /= 255.0  # 归一化到[0,1]范围
    return image,label
#-------------
train_image_label_ds  = train_ds.map(load_and_preprocess_from_path_label,num_parallel_calls=6)
# train_ds = train_image_label_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=train_image_count))
BATCH_SIZE = 200
train_ds=train_image_label_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(buffer_size = tf.data.experimental.AUTOTUNE).repeat()

train_steps_per_epoch=int(train_image_count//BATCH_SIZE)
#-------------
valid_ds = valid_ds.map(load_and_preprocess_from_path_label)
valid_ds = valid_ds.batch(BATCH_SIZE).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)#不加re
valid_steps_per_epoch=int(valid_image_count//BATCH_SIZE)
x,y = next(iter(train_ds))
print('train sample:', x.shape, y.shape)



tf.random.set_seed(2)
DefaultConv2D = partial(keras.layers.Conv2D,kernel_size=[3,3], activation='relu', padding="SAME")
model = keras.models.Sequential([
    DefaultConv2D(filters=64, input_shape=[100, 100, 3]),
    DefaultConv2D(filters=64),
    keras.layers.MaxPooling2D(pool_size=[2,2],strides=2),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    keras.layers.MaxPooling2D(pool_size=2,strides=2),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    keras.layers.MaxPooling2D(pool_size=2,strides=2),
    DefaultConv2D(filters=512),
    DefaultConv2D(filters=512),
    keras.layers.MaxPooling2D(pool_size=2,strides=2),
    DefaultConv2D(filters=512),
    DefaultConv2D(filters=512),
    keras.layers.MaxPooling2D(pool_size=2,strides=2),
    keras.layers.Flatten(),
    keras.layers.Dense(units=256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=11, activation='softmax') ])

model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
             loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['acc'])


model.summary()


history = model.fit(train_ds,
                    epochs=epochs,
                    steps_per_epoch=train_steps_per_epoch,
                    validation_data=valid_ds,
                    validation_steps=valid_steps_per_epoch,
                    callbacks=[change_Lr])

model.save('model2.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('accuracy.png')
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('loss.png')
plt.show()

