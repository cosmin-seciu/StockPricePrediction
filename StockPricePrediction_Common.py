import numpy as np
import pandas as pnd
import pandas_datareader as pdr

import seaborn as sb
import matplotlib.pyplot as plt

sb.set_style('whitegrid')
sb.set_context('talk')

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}

plt.rcParams.update(params)

# get stock price information
def import_data(name,attempts = 3):
    if name:
        while attempts > 0:
            try:
                df = pdr.get_data_yahoo(name)
                ndf=df.reindex(index=pnd.date_range(df.index.min(),df.index.max(),freq='D')).fillna(method='ffill')
                attempts = 0
                return ndf
            except:
                print("Data import failed. {} attempts remaining".format(attempts))
                attempts = attempts - 1
    else:
        print("Please insert an index name!")
    return None


#Plot Windows
def plot_windows(predicted_data,true_data,prediction_len=3):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    #plot actual data
    ax.plot(true_data,label='True Data',c='black',alpha=0.3)
    #plot flattened data
    plt.plot(np.array(predicted_data).flatten(),label='Prediction_full',c='g',linestyle='--')
    #plot each window
    for i,data in enumerate(predicted_data):
        padding = [None for p in range(i*prediction_len)]
        plt.plot(padding+data,label='Prediction',c='black')
    plt.title("Forecast plot with Prediction Window={}".format(prediction_len))
    plt.show()