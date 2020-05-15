#使用预训练的VGG16网络搭配自定义的全连接网络
#图像输入为256*256*3
#学习率衰减
#达到71%

import numpy as np
import os
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
# from tensorflow import keras
# from tensorflow.keras import layers, optimizers, datasets,Sequential
from keras.applications import VGG16
from tensorflow_core.python.keras.callbacks import LearningRateScheduler
from math import pow, floor
from keras.preprocessing.image import ImageDataGenerator
import pathlib

#-------------
def scheduler(epoch):
    init_lrate = 0.0001
    drop = 0.7
    epochs_drop = 4
    lrate = init_lrate * pow(drop, floor(1 + epoch) / epochs_drop)
    return lrate
change_Lr = LearningRateScheduler(scheduler)

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(256, 256, 3))
conv_base.summary()

base_dir = "D:/document/data/food-11"
train_dir = os.path.join(base_dir, 'training')
validation_dir = os.path.join(base_dir, 'validation')

datagen = ImageDataGenerator(rescale=1. / 255)
BATCH_SIZE = 500


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    # rotation_range=20,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # shear_range=0.1,
    # zoom_range=0.1,
    # horizontal_flip=True,
    # fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

def extract_features(directory,datagen, sample_count,num,BATCH_SIZE):
    features = np.zeros(shape=(sample_count*num, 8, 8, 512))
    labels = np.zeros(shape=(sample_count*num,11))
    generator = datagen.flow_from_directory(directory,
                                            target_size=(256, 256),
                                            batch_size=BATCH_SIZE,
                                            class_mode='categorical')
    i=0
    for item in range(num):
        for inputs_batch, labels_batch in generator:

            features_batch = conv_base.predict(inputs_batch)
            features[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = features_batch
            labels[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = labels_batch
            i += 1
            if i * BATCH_SIZE >= sample_count:
                break
    return features, labels
train_features, train_labels = extract_features(train_dir,train_datagen, 9866*1,1,4933)
validation_features, validation_labels = extract_features(validation_dir,test_datagen, 3430,1,343)

train_features = np.reshape(train_features, (9866*1, 8 * 8 * 512))
validation_features = np.reshape(validation_features, (3430, 8 * 8 * 512))



model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=8 * 8 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(11, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='categorical_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, train_labels,
                    epochs=50,
                    batch_size=BATCH_SIZE,
                    validation_data=(validation_features, validation_labels),
                    callbacks=[change_Lr])

model.save('model4.h5')

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
