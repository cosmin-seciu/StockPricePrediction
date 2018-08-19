import math
import numpy as np
import matplotlib.pyplot as plt
import time

from StockPricePrediction_Common import import_data
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense
from sklearn.metrics import mean_squared_error
from keras.preprocessing.sequence import pad_sequences


TRAIN_PERCENT = 0.7
STOCK_INDEX = '^GSPC'
VERBOSE = True

# load data
stock_df = import_data(STOCK_INDEX)
stock_close_series = stock_df.Close
#stock_close_series.plot()
#plt.show()
print("Data imported successfully!")


##### SEQUENCE MODELING #####

# prepare training and testing data sets for LSTM based sequence modeling
def seq_train_test(time_series, scaling=True, train_size=0.9):
    scaler = None
    if scaling:
        scaler = MinMaxScaler(feature_range=(0, 1))
        time_series = np.array(time_series).reshape(-1, 1)
        scaled_stock_series = scaler.fit_transform(time_series)
    else:
        scaled_stock_series = time_series
    train_size = int(len(scaled_stock_series) * train_size)
    train = scaled_stock_series[0:train_size]
    test = scaled_stock_series[train_size:len(scaled_stock_series)]
    return train, test, scaler

#split train and test datasets
train, test, scaler = seq_train_test(stock_close_series,scaling=True,train_size=TRAIN_PERCENT)
train = np.reshape(train,(1,train.shape[0],1))
test = np.reshape(test,(1,test.shape[0],1))
x_train = train[:,:-1,:]
y_train = train[:,1:,:]
x_test = test[:,:-1,:]
y_test = test[:,1:,:]
print("Data Split Complete!")
print("x_train shape={}".format(x_train.shape))
print("y_train shape={}".format(y_train.shape))
print("x_test shape={}".format(x_test.shape))
print("y_test shape={}".format(y_test.shape))

#Building LSTM network

def build_seq_LSTM(hidden_units=4,input_shape=(1,1),verbose=False):
    model = Sequential()
    # samples*timesteps*features
    model.add(LSTM(input_shape=input_shape,units=hidden_units,return_sequences=True))
    # readout layer. TimeDistributedDense uses the same weights for all time steps
    model.add(TimeDistributed(Dense(1)))
    start=time.time()
    model.compile(loss="mse",optimizer="rmsprop")
    if verbose:
        print("Compilation time: ",time.time() - start)
        print(model.summary())
    return model

#Get the model

lstm_model = None
try:
    lstm_model = build_seq_LSTM(input_shape=(x_train.shape[1],1),verbose=VERBOSE)
except:
    print("Model build failed! Trying Again!")
    lstm_model = build_seq_LSTM(input_shape=(x_train.shape[1], 1), verbose=VERBOSE)

#Train the model
lstm_model.fit(x_train, y_train, epochs=150, batch_size=1, verbose=2)
print("Model fit complete!")
#Training performance
trainPredict = lstm_model.predict(x_train)
trainScore = math.sqrt(mean_squared_error(y_train[0],trainPredict[0]))
print('Train Score: %.2f RMSE' % (trainScore))
#Padding Input Sequence
testPredict = pad_sequences(x_test,maxlen=x_train.shape[1],padding='post',dtype='float64')
#Predicting Values
testPredict = lstm_model.predict(testPredict)
#Predicting performance
testScore = math.sqrt(mean_squared_error(y_test[0],testPredict[0][:x_test.shape[1]]))
print('Test Score: %.2f RMSE' % (testScore))

#Rescale prediction values to the original scale so we can compare them with the true data

trainPredict = np.array(trainPredict).reshape(-1,1)
trainPredict = scaler.inverse_transform(trainPredict)
testPredict = np.array(testPredict).reshape(-1,1)
testPredict = scaler.inverse_transform(testPredict)

#Plotting the true and predicted data

train_size = len(trainPredict)+1
plt.plot(stock_close_series.index,stock_close_series.values,c='black',alpha=0.3,label='True Data')
plt.plot(stock_close_series.index[1:train_size],trainPredict,label='Training Fit',c='g')
plt.plot(stock_close_series.index[train_size+1:],testPredict[:x_test.shape[1]],label='Testing Forecast')
plt.title('Forecast Plot')
plt.legend()
plt.show()
