import geopy.distance
import numpy as np
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# function that transforms the dataframe as follows:
# column_1: time at the arrival
# column_2: average speed at the arrival
def transform_to_speed(df):
    # generate new column 'time_arrival' by shifting 'timestamp'
    df['time_arrival']=df.timestamp.shift(-1)
    # save position at departure as tuple
    df['position_departure'] = list(zip(df.latitude, df.longitude))
    # generate new column 'position_arrival' by shifting 'position_departure'
    df['position_arrival']=df.position_departure.shift(-1)
    # deleting line which includes NaN entry
    df = df[:-1]
    # calculate distance between two (lat,lon) tuples using geopy package
    df['distance_covered'] = df.apply(lambda x: geopy.distance.geodesic\
                                                (x['position_arrival'],\
                                                 x['position_departure']).km,\
                                                 axis = 1)
    # calculate required time
    df['time_required'] = df.time_arrival - df.timestamp
    df['time_required'] = df['time_required']/np.timedelta64(1,'h')
    # calculate average speed at arrival
    df['average_speed_arrival'] = df['distance_covered']/df['time_required']
    # delete unused columns
    df = df.drop(columns=['longitude', 'latitude', 'position_departure',\
                          'position_arrival', 'timestamp',\
                          'distance_covered', 'time_required'])
    # fix outliers manually
    df = df[df['average_speed_arrival'] <= 150]
    return df

def fix_duplicates(df):
    # delete rows with longitude/latitude pairs that coincide with their
    # predeccessor and successor:
    # this will delete all data which belong to timestamps in a taxi's break,
    # except the start and end timestamps of the break
    df = df.loc[~((df.longitude == df.longitude.shift(1))\
                              & (df.longitude == df.longitude.shift(-1))\
                              & (df.latitude == df.latitude.shift(-1))\
                              & (df.latitude == df.latitude.shift(-1)))]
    # delete duplicates with respect to the 'timestamp' column
    df.drop_duplicates(subset = "timestamp", keep = 'first', inplace = True)
    return df

# re-frames the dataframe to a supervised learning format
def re_frame(data, timesteps, forecast, dropnan=True):
    # initialize lists
	cols, names = list(), list()
	# generate input sequence (t-n,...,t-1), depending on variable 'timesteps'
	for i in range(timesteps, 0, -1):
		cols.append(data.shift(i))
		names += [('t-%d' % i)]
	# generate forecast sequence (t,...,t+n), depending on 'timesteps'
	for i in range(0, forecast):
		cols.append(data.shift(-i))
		if i == 0:
			names += [('t')]
		else:
			names += [('t+%d' % i)]
	# summarize dataframe
	df = concat(cols, axis=1)
	df.columns = names
	# drop rows with NaN values
	if dropnan:
		df.dropna(inplace=True)
	return df

# fits (trains) the LSTM-RNN model
def fit_lstm(train, batch_size, nb_epoch, neurons, forecast, timesteps):
    # seperate input X and output y as numpy array from the dataframe
    X, y = train.iloc[:, 0:-forecast],\
           train.iloc[:, timesteps:forecast+timesteps]
    X, y = np.array(X), np.array(y)
    # reshape X (SAMPLES, TIMESTEPS, FEATURES) to fit the shape for the model
    X = X.reshape(X.shape[0], X.shape[1], 1)
    # initialize model, batch_input_shape=(batch_size, TIMESTEPS, FEATURES)
    model = Sequential()
    model.add(LSTM(neurons,\
                   batch_input_shape=(batch_size, X.shape[1], X.shape[2]),\
                   stateful=True))
    model.add(Dense(forecast))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # fit (train) the LSTM-RNN model
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size,\
                  verbose=0, shuffle=False)
        model.reset_states()
    return model

# execute a forecast step with respect to a single input object
def forecast_lstm(model, batch_size, X, timesteps):
    # reshape X (SAMPLES, TIMESTEPS, FEATURES) to fit the shape of the model
    X = X.reshape(1,timesteps)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    # forecast step
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,:]