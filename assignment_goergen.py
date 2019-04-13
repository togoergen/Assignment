import numpy as np
import pandas as pd
import assignment_goergen_library as lib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# close all plots..
plt.close('all')

# choose number of timesteps (previous data) and forecast (# steps forward)
timesteps = 1
forecast = 4

# import the taxi trajectory data using pandas
raw_data = pd.read_csv('9368.txt', delimiter=',', parse_dates=[0], header=0,\
                       usecols=['timestamp', 'longitude', 'latitude'])

# fix duplicates and unnecessary data
raw_data = lib.fix_duplicates(raw_data)

# transform the taxi dataframe to speed
transformed_data = lib.transform_to_speed(raw_data)

# calculate average speed and covered distance
average_speed = transformed_data['average_speed_arrival'].sum()\
                /transformed_data.shape[0]
print('average speed: ', average_speed)
km_covered = (transformed_data.iloc[len(transformed_data)-1,0]\
                                    -transformed_data.iloc[0,0])\
              /np.timedelta64(1,'h')*average_speed
print('km covered: ', km_covered)

# set timestamp_arrival as index
transformed_data.set_index('time_arrival',inplace=True)

# manually remove obvious outliers
transformed_data.loc['2008-02-07 01:42:53':'2008-02-07 02:26:23'][:] = 0.0
transformed_data.loc['2008-02-07 12:25:08':'2008-02-07 12:46:58'][:] = 0.0
transformed_data.loc['2008-02-08 01:50:21':'2008-02-08 12:00:42'][:] = 0.0
transformed_data.loc['2008-02-07 13:58:49':'2008-02-07 20:06:38'][:] = 40.0

#plot data
fig, axx = plt.subplots(figsize=(10,5))
transformed_data.plot(ax=axx)

# scale the data as preparation for the LSTM-RNN
X_scaler = MinMaxScaler(feature_range=(-1, 1))
transformed_data[['average_speed_arrival']]\
                = X_scaler.fit_transform(transformed_data)

# re-frame data as supervised learning input for the LSTM-RNN
supervised_data = lib.re_frame(transformed_data, timesteps, forecast)

# split dataframe into 90% train and 10% test data
bound = round(len(transformed_data.index)*0.9)
supervised_train_data, supervised_test_data = supervised_data[:bound],\
                                              supervised_data[bound:]

# fit the model to training data
RNN = lib.fit_lstm(supervised_train_data, 1, 1500, 1, forecast, timesteps)
 
# walk-forward validation on the test data
predictions = list()
ax = list()
for i in range(len(supervised_test_data)-1):
    # seperate input X and output y as numpy array from the dataframe
    X, y = np.array(supervised_test_data.iloc[i+1, 0:-forecast]),\
           np.array(supervised_test_data.iloc[i, timesteps:forecast+timesteps])
    # make forecast
    ypred = lib.forecast_lstm(RNN, 1, X, timesteps) 
    # invert scaling
    ypred = X_scaler.inverse_transform(ypred.reshape(forecast, 1))
    y = X_scaler.inverse_transform(y.reshape(forecast, 1))
    # store forecast, solution and index
    predictions.append(ypred.reshape(forecast))
    ax.append(supervised_test_data.index[i])
    # print output
    print(supervised_test_data.index[i])
    print('Expected =', y.reshape(forecast))
    print('Predicted =', ypred.reshape(forecast))
    print('')
    
# plot predictions
plt.plot(ax, predictions)
plt.show()