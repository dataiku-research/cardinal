import os
import skimage
import sys
import keras
import numpy as np
from keras.applications import resnet_v2
from keras.layers import BatchNormalization, Dropout, Dense, GlobalAveragePooling2D
from keras import Model
from tensorflow.keras.preprocessing import image
from keras import optimizers, layers
from keras.models import Sequential
from skimage import transform
import copy
import tensorflow as tf


keras.backend.clear_session()


def create_split_resnet():

    keras.backend.clear_session()
    img_size = 224

    resnet = resnet_v2.ResNet50V2(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3), pooling='avg')
    
    resnet_model = Sequential()
    resnet_model.add(layers.Lambda(lambda image: tf.image.resize(image, (img_size, img_size))))
    resnet_model.add(resnet)

    for layer in resnet.layers:
        layer.trainable = False

    model = Sequential()
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.BatchNormalization())
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=2e-5),
                  metrics=['accuracy'])

    return resnet_model, model



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


from keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

X = np.vstack([x_train, x_test])

y_train = keras.utils.to_categorical(y_train, num_classes=10, dtype='float32')
y_test = keras.utils.to_categorical(y_test, num_classes=10, dtype='float32')

Y = np.vstack([y_train, y_test])

preprocess, predictor = create_split_resnet()

embeddings = preprocess.predict(X)
np.save('cifar_embeddings.npy', embeddings)
np.save('cifar_target.npy', Y)