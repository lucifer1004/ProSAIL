import numpy as np
import matplotlib.pyplot as plt
from keras.layers import concatenate, Input, Dense, Activation, Dropout, Flatten, UpSampling1D, UpSampling2D
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.pooling import MaxPooling1D, MaxPooling2D
from keras.models import Model
from keras.layers.advanced_activations import PReLU
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


def unet(pretrained_weights=None, input_size=(4096, 1,)):
    inputs = Input(input_size)
    conv1 = Conv1D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv1D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv1)

    conv2 = Conv1D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv1D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    conv3 = Conv1D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv1D(128, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling1D(pool_size=2)(conv3)

    conv4 = Conv1D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv1D(256, 3, activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling1D(pool_size=2)(drop4)

    conv5 = Conv1D(512, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv1D(512, 3, activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv1D(256, 2, activation='relu', padding='same')(UpSampling1D(size=2)(drop5))
    merge6 = concatenate([drop4, up6], axis=2)
    conv6 = Conv1D(256, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv1D(256, 3, activation='relu', padding='same')(conv6)

    up7 = Conv1D(128, 2, activation='relu', padding='same')(UpSampling1D(size=2)(conv6))
    merge7 = concatenate([conv3, up7], axis=2)
    conv7 = Conv1D(128, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv1D(128, 3, activation='relu', padding='same')(conv7)

    up8 = Conv1D(64, 2, activation='relu', padding='same')(UpSampling1D(size=2)(conv7))
    merge8 = concatenate([conv2, up8], axis=2)
    conv8 = Conv1D(64, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv1D(64, 3, activation='relu', padding='same')(conv8)

    up9 = Conv1D(32, 2, activation='relu', padding='same')(UpSampling1D(size=2)(conv8))
    merge9 = concatenate([conv1, up9], axis=2)
    conv9 = Conv1D(32, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv1D(32, 3, activation='relu', padding='same')(conv9)
    conv10 = Conv1D(1, 1)(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['accuracy'])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
