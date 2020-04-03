import pandas as pd
import sys
import numpy as np
from multiprocessing import cpu_count, Pool
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Dense, Dropout, LSTM, Reshape, Flatten
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.linear_model import LinearRegression

dir_path = sys.argv[1]

headers = ['Date', 'Time', 'Global_active_power', 'Global_reactive_power',
           'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
           'Sub_metering_3']

dtypes = {'Date':'str', 'Time':'str', 'Global_active_power':'float',
          'Global_reactive_power': 'float', 'Voltage':'float',
          'Global_intensity':'float', 'Sub_metering_1':'float',
          'Sub_metering_2':'float', 'Sub_metering_3':'float'}

# print(df.head)


def parallel_map(data, func):
    n_cores = cpu_count()
    data_split = np.array_split(data, n_cores)
    pool = Pool(n_cores)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data

def parse(row):
    row['DateTime'] = pd.to_datetime(row['DateTime'],format='%d/%m/%Y %H:%M:%S')
    return row

def series_to_supervised(data, window_size, horizon, inputs, targets):
    """
    Frame a time series as a supervised learning dataset.
    
    Arguments:
        data: A pandas DataFrame containing the time series
        (the index must be a DateTimeIndex).
        window_size: Number of lagged observations as input.
        horizon: Number of steps to forecast ahead.
        inputs: A list of the columns of the dataframe to be lagged.
        targets: A list of the columns of the dataframe to be forecasted.
    
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    
    if targets == 'all':
        targets = data.columns
    
    if inputs == 'all':
        inputs = data.columns

    
    result = DataFrame(index=df.index)
    names = []
    
    # input sequence (t-w, ..., t-1)
    for i in range(window_size, 0, -1):
        result = pd.concat([result, data[inputs].shift(i)], axis=1)
        names += [(f'{data[inputs].columns[j]}(t-{i})') for j in range(len(inputs))]
    
    # the input not shifted (t)
    result = pd.concat([result, data.copy()], axis=1)
    names += [(f'{column}(t)') for column in data.columns]
    
    # forecast (t+h)
    for i in [horizon]:
        result = pd.concat([result, data[targets].shift(-i)], axis=1)
        names += [(f'{data[targets].columns[j]}(t+{i})') for j in range(len(targets))]
    
    # put it all together
    result.columns = names

    # drop rows with NaN values
    result.dropna(inplace=True)
    return result

def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None):
    np.random.seed(seed)
    m = len(df)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[:train_end]
    validate = df.iloc[train_end:validate_end]
    test = df.iloc[validate_end:]
    return train, validate, test

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


df = pd.read_csv(dir_path, sep=';',dtype=dtypes, na_values=['?'])
# Preprocessing


df['DateTime'] = df['Date'] + ' ' + df['Time']
df = parallel_map(df, parse)
# df.dtypes
print("DateTime Merged-",df.shape)

df.drop(['Date', 'Time'], axis=1, inplace=True)
df = df[[df.columns[-1]] + list(df.columns[:-1])]
df.set_index('DateTime', inplace=True)
# df.head()

df['hour'] = df.index.hour
df['day'] = df.index.day
df['month'] = df.index.month
df['day_of_week'] = df.index.dayofweek
df['Rest_active_power'] = df['Global_active_power'] * 1000 / 60 - df['Sub_metering_1'] - df['Sub_metering_2'] - df['Sub_metering_3']

inputs = ['Global_active_power', 'Global_reactive_power', 'Voltage',
          'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
          'Sub_metering_3', 'Rest_active_power']

targets = ['Global_active_power']

df_supervised = series_to_supervised(df, window_size=5, horizon=1, inputs=inputs, targets=targets)
# df_supervised.head()

train,validate, test = train_validate_test_split(df_supervised)
print(type(train))
print(train.shape)


X_train = train.values[:, :-1]
# print(X_train.shape)
y_train = train.values[:, -1]
# print(y_train.shape)

X_validate = validate.values[:, :-1]
y_validate = validate.values[:, -1]

X_test = test.values[:, :-1]
y_test = test.values[:, -1]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_validate = scaler.transform(X_validate)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

#Predictions on standard output
for i in predictions:
    print(i)