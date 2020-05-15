#使用预训练的EfficientNet网络搭配自定义的全连接网络
#图像输入为150*150*3
#学习率衰减
#数据增强
#接近80%


import tensorflow as tf
import numpy as np
import os
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
# from tensorflow import keras
# from tensorflow.keras import layers, optimizers, datasets,Sequential
from keras.preprocessing.image import ImageDataGenerator
import efficientnet.keras as efn
from tensorflow.keras import regularizers
from tensorflow_core.python.keras.callbacks import LearningRateScheduler
from math import pow, floor

#-------------
def scheduler(epoch):
    init_lrate = 0.0002
    drop = 0.8
    epochs_drop = 2
    lrate = init_lrate * pow(drop, floor(1 + epoch) / epochs_drop)
    return lrate
change_Lr = LearningRateScheduler(scheduler)




efficient_net = efn.EfficientNetB7(weights='imagenet', include_top=False, input_shape=[150,150,3])
efficient_net.summary()

base_dir = "D:/document/data/food-11"
train_dir = os.path.join(base_dir, 'training')
validation_dir = os.path.join(base_dir, 'validation')


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)


def extract_features(directory,datagen, sample_count,num,BATCH_SIZE):
    features = np.zeros(shape=(sample_count*num, 5, 5, 2560))
    labels = np.zeros(shape=(sample_count*num,11))
    generator = datagen.flow_from_directory(directory,
                                            target_size=(150, 150),
                                            batch_size=BATCH_SIZE,
                                            class_mode='categorical')
    i=0
    for item in range(num):
        for inputs_batch, labels_batch in generator:

            features_batch = efficient_net.predict(inputs_batch)
            features[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = features_batch
            labels[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = labels_batch
            i += 1
            if i * BATCH_SIZE >= sample_count:
                break
    return features, labels
train_features, train_labels = extract_features(train_dir,train_datagen, 9866*5,1,4933)
validation_features, validation_labels = extract_features(validation_dir,test_datagen, 3430,1,343)
#
train_features = np.reshape(train_features, (9866*5, 5 * 5 * 2560))
validation_features = np.reshape(validation_features, (3430, 5 * 5 * 2560))

#kernel_regularizer = regularizers.l2(0.01)
model = models.Sequential()

model.add(layers.Dense(526, activation='relu',input_dim=5 * 5 * 2560))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(11, activation='sigmoid'))

# model.summary()

model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
              loss='categorical_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, train_labels,
                    epochs=25,
                    batch_size=1000,
                    validation_data=(validation_features, validation_labels),
                    callbacks=[change_Lr]
                    )


model.save('model6.h5')

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