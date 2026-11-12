import numpy as np
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import plot
from const import *
from stock_historical_data import StockHistoricalData
from stock_information import StockInformation

shd = StockHistoricalData()
shd.get_data()
shd.export_file()

si = StockInformation()
si.get_data()
si.export_file()

df = shd.df

pre_day = 30
scala_x = MinMaxScaler(feature_range=(0, 1))
scala_y = MinMaxScaler(feature_range=(0, 1))
cols_x = [oc_str, hl_str, f'{sma}{ma1}', f'{sma}{ma2}',
          f'{sma}{ma3}']
cols_y = [close_str]
scaled_data_x = scala_x.fit_transform(
    df[cols_x].values.reshape(-1, len(cols_x)))
scaled_data_y = scala_y.fit_transform(
    df[cols_y].values.reshape(-1, len(cols_y)))

x_total = []
y_total = []

for i in range(pre_day, len(df)):
    x_total.append(scaled_data_x[i - pre_day:i])
    y_total.append(scaled_data_y[i])

test_size = 365

x_train = np.array(x_total[:len(x_total) - test_size])
x_test = np.array(x_total[len(x_total) - test_size:])
y_train = np.array(y_total[:len(y_total) - test_size])
y_test = np.array(y_total[len(y_total) - test_size:])

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Build Model
model = Sequential()

model.add(LSTM(units=60, return_sequences=True,
               input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=60, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=60, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=60, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=60))
model.add(Dropout(0.2))
model.add(Dense(units=len(cols_y), activation='relu'))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=120,
          steps_per_epoch=40, use_multiprocessing=True)
# model.save(f'{company}.h5')
print('Done Training Model')

# Testing
predict_price = model.predict(x_test)
predict_price = scala_y.inverse_transform(predict_price)

# Plotting the Stat
real_price = df[len(df) - test_size:][close_str].values.reshape(-1, 1)
real_price = np.array(real_price)
real_price = real_price.reshape(real_price.shape[0], 1)

plot.draw(real_price=real_price, prediction_price=predict_price)

# Make Prediction
x_predict = df[len(df) - pre_day:][cols_x].values.reshape(-1, len(cols_x))
x_predict = scala_x.transform(x_predict)
x_predict = np.array(x_predict)
x_predict = x_predict.reshape(1, x_predict.shape[0], len(cols_x))
prediction = model.predict(x_predict)
prediction = scala_y.inverse_transform(prediction)
print(prediction)
