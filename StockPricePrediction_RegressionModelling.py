import numpy as np
import time
import keras
import math

from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from sklearn.metrics import mean_squared_error
from StockPricePrediction_Common import import_data
from StockPricePrediction_Common import plot_windows


STOCK_INDEX = '^GSPC'
# STOCK_INDEX = 'UCG.MI'
WINDOW = 6
PRED_LENGTH = int(WINDOW/2)


stock_df = import_data(STOCK_INDEX)
stock_close_series = stock_df.Close
# stock_close_series.plot()
# plt.show()
print("Data imported successfully!")

##### REGRESSION MODELING #####

# prepare training and testing data sets for LSTM based regression modeling
def reg_train_test(timeseries, sequence_length = 51, train_size = 0.9, roll_mean_window = 5, normalize = True, scale = False):
    # smoothen out series
    if roll_mean_window:
        timeseries = timeseries.rolling(roll_mean_window).mean().dropna()
    # create windows
    result = []
    for index in range(len(timeseries) - sequence_length):
        result.append(timeseries[index: index + sequence_length])
    #print(result)

    # normalize data as a variation of 0th index
    if normalize:
        normalised_data = []
        for window in result:
            normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
            normalised_data.append(normalised_window)
        result = normalised_data

    # identify train-test splits
    result = np.array(result)
    row = round(train_size * result.shape[0])
    # split train and test sets
    train = result[:int(row), :]
    test = result[int(row):, :]
    # scale data in 0-1 range
    scaler = None
    if scale:
        scaler = MinMaxScaler(feature_range=(0,1))
        train = scaler.fit_transform(train)
        test = scaler.transform(test)
    # split independent and dependent variables
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = test[:, :-1]
    y_test = test[:, -1]
    # Transforms for LSTM input
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

    return x_train, y_train, x_test, y_test, scaler


x_train,y_train,x_test,y_test,scaler = reg_train_test(stock_close_series, sequence_length=WINDOW+1, roll_mean_window=None, normalize=True, scale=False)
print("Data Split Complete!")
print("x_train shape={}".format(x_train.shape))
print("y_train shape={}".format(y_train.shape))
print("x_test shape={}".format(x_test.shape))
print("y_test shape={}".format(y_test.shape))

#Building LSTM network

def build_reg_LSTM(layer_units=[100,100],dropouts=[0.2,0.2],window_size=50):
    model = Sequential()

    #Hidden Layer 1
    model.add(LSTM(layer_units[0],input_shape=(window_size,1),return_sequences=True))
    model.add(Dropout(dropouts[0]))
    #Hidden Layer 2
    model.add(LSTM(layer_units[1]))
    model.add(Dropout(dropouts[1]))
    #Output Layer
    model.add(Dense(1))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compilation time: ", time.time() - start)
    print(model.summary())
    return model

lstm_model = None
try:
    lstm_model = build_reg_LSTM(layer_units=[50,100],window_size=WINDOW)
except:
    print("Model build failed! Trying Again!")
    lstm_model = build_reg_LSTM(layer_units=[50, 100], window_size=WINDOW)
    #print(lstm_model)

#Training the model. Using Early Stopping to avoid Overfitting
callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',patience=2,verbose=0)]
lstm_model.fit(x_train,y_train,epochs=20,batch_size=16,verbose=1,validation_split=0.05,callbacks=callbacks)
print("Model fit complete!")

#Predicting Windows
def predicting_windows(model, data, window_size=6, prediction_len=3):
    pred_list = []
    # loop for every sequence in the dataset
    for window in range(int(len(data) / prediction_len)):
        seq = data[window * prediction_len]
        predicted = []
        # loop till required prediction length is achieved
        for j in range(prediction_len):
            predicted.append(model.predict(seq[np.newaxis, :, :])[0, 0])
            seq = seq[1:]
            seq = np.insert(seq, [window_size - 1], predicted[-1], axis=0)
        pred_list.append(predicted)
    return pred_list

train_pred_seqs = predicting_windows(lstm_model, x_train, window_size=WINDOW, prediction_len=PRED_LENGTH)
train_offset = y_train.shape[0] - np.array(train_pred_seqs).flatten().shape[0]
train_rmse = math.sqrt(mean_squared_error(y_train[train_offset:], np.array(train_pred_seqs).flatten()))
print('Train Score: %.2f RMSE' % (train_rmse))
test_pred_seqs = predicting_windows(lstm_model,x_test,window_size=WINDOW,prediction_len=PRED_LENGTH)
test_offset = y_test.shape[0] - np.array(test_pred_seqs).flatten().shape[0]
test_rmse = math.sqrt(mean_squared_error(y_test[test_offset:], np.array(test_pred_seqs).flatten()))
print('Test Score: %.2f RMSE' % (test_rmse))

print("Prediction Complete!")
plot_windows(test_pred_seqs,y_test,prediction_len=PRED_LENGTH)