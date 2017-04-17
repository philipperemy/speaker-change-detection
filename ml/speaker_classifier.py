import numpy as np
from keras import regularizers
from keras.layers import Dense, Dropout
from keras.models import Sequential


def build_model(m):
    m.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'], )


def fit_model(m, kx_train, ky_train, kx_test, ky_test, epochs=1000):
    m.fit(kx_train,
          ky_train,
          batch_size=64,
          epochs=epochs,
          verbose=1,
          validation_data=(kx_test, ky_test))


def inference_model(m, input_list):
    probabilities = m.predict(input_list)
    k_star = np.argmax(np.sum(np.log(probabilities), axis=0))
    return k_star


# Dropout
# 4 speakers 500 per class - loss: 0.3613 - acc: 0.8646 - val_loss: 0.3992 - val_acc: 0.8493
def get_model_bak():
    m = Sequential()
    m.add(Dense(200, batch_input_shape=[None, 39 * 10], activation='sigmoid'))
    m.add(Dropout(0.5))
    m.add(Dense(4, activation='softmax'))
    return m


# kernel_regularizer=regularizers.l2(0.01)
# loss: 0.3954 - acc: 0.9077 - val_loss: 0.5580 - val_acc: 0.8394
def get_model_bak2():
    m = Sequential()
    m.add(Dense(200, batch_input_shape=[None, 39 * 10], activation='sigmoid',
                kernel_regularizer=regularizers.l2(0.01)))
    m.add(Dense(4, activation='softmax'))
    return m


# loss: 0.0460 - acc: 0.9921 - val_loss: 0.4071 - val_acc: 0.8713
def get_model():
    m = Sequential()
    m.add(Dense(200, batch_input_shape=[None, 39 * 10], activation='sigmoid'))
    m.add(Dense(4, activation='softmax'))
    return m
