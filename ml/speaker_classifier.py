import numpy as np
from keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential

from ml.mfcc_data_generation import normalize


def build_model(m):
    m.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'], )


def fit_model(m, kx_train, ky_train, kx_test, ky_test, max_epochs=1000):
    checkpoint = ModelCheckpoint(filepath='checkpoints/model_{epoch:02d}_{val_acc:.3f}.h5', save_best_only=True)
    # if the accuracy does not increase by 1.0% over 10 epochs, we stop the training.
    early_stopping = EarlyStopping(monitor='val_acc', min_delta=1.0, patience=10, verbose=0, mode='max')
    m.fit(kx_train,
          ky_train,
          batch_size=1,
          epochs=max_epochs,
          verbose=1,
          validation_data=(kx_test, ky_test),
          callbacks=[checkpoint, early_stopping])


def inference_model(m, input_list):
    log_probabilities = predict(m, input_list, log=True)
    k_star = np.argmax(np.sum(log_probabilities, axis=0))
    return k_star


def predict(m, input_list, log=False):
    mean = np.mean([np.mean(t) for t in input_list])
    std = np.mean([np.std(t) for t in input_list])
    input_list = normalize(input_list, mean, std)
    probabilities = m.predict(input_list)
    if log:
        probabilities = np.log(probabilities)
    return probabilities


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
