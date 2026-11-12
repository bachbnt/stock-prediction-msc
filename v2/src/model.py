import numpy as np
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from config import model_loss, model_optimizer, model_epochs, model_steps_per_epoch, model_batch_size


class LSTMModel:
    def __int__(self, x_train=[], y_train=[]):
        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        self.model.add(LSTM(units=50))
        self.model.add(Dense(1))

        self.model.compile(loss=model_loss, optimizer=model_optimizer)
        self.model.fit(x_train, y_train, epochs=model_epochs, steps_per_epoch=model_steps_per_epoch,
                  batch_size=model_batch_size)

    def get_model(self):
        return self.model
