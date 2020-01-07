#### CNN

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Conv1D, Conv3D, Reshape,BatchNormalization, Activation, LSTM, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import load_model
import keras
import matplotlib.pyplot  as plt
import os
from keras.callbacks import ModelCheckpoint
import time


def look_back(X, a):
    X_lb = np.zeros((len(X)- 29*a , a, 12))
    for i in range(len(X) - 29 * a):
        for j in range(a):
            X_lb[i, j] = X[i+(j*29)]
    X_lb = X_lb.reshape(int(len(X)/29) - a, 29, a, 12)
    Y_lb = X[a*29:, 7]
    Y_lb = Y_lb.reshape(int(len(X)/29) - a, 29, 1)
    return X_lb, Y_lb

def division(data):
    train_size = int(len(data)*0.6)
    val_size = int(len(data)*0.8)
    data_train = data[0:train_size]
    data_val = data[train_size:val_size]
    data_test = data[val_size:len(data)]
    return data_train, data_val, data_test



ts = 5
df= pd.read_csv("녹조최종.csv", encoding='ms949')

x, y = look_back(df.values, 5)
y = y.reshape(1371-ts,29,1,1)
X_train, X_val, X_test = division(x)
Y_train, Y_val, Y_test = division(y)


# 조기종료 설정
early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 30, mode = 'min')
mc = ModelCheckpoint('weights/best_model.h5', monitor='val_loss', mode='min', save_best_only=True)


model = Sequential()
model.add(Conv2D(64, (1, 2), padding='same', input_shape=(29,ts,12)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32, (1, 2), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(16, (1, 2), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(8, (1, 2), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(1, (1, ts), padding='valid'))

model.compile(loss='mean_squared_error', optimizer='adam')
#hist = model.fit(X_train, Y_train, epochs=500, batch_size=8, validation_data=(X_val, Y_val), callbacks=[early_stopping, mc])


model.load_weights('가중치 저장소/12변수/2-3. 12변수 + time step 5 + CNN2D.h5')
y_predict = model.predict(X_test)
y_predict = y_predict.reshape(-1, 1).astype('float32')
real_y = Y_test.reshape(-1, 1).astype('float32')

try:
    rmse = sqrt(mean_squared_error(y_predict, real_y))
except ValueError:
    pass



#### LSTM

time_step=5
df= pd.read_csv("녹조최종.csv", encoding='ms949')

rmse_lst = []
lstm_rs = []
for j in range(29):
    stst = time.time()
    new_np = np.array([]).reshape(0,12)
    for i in range(1371):
        new_np = np.vstack((new_np, df.values[j+(i*29)]))

    x = new_np
    y = new_np[:,7]

    dataX, dataY = [], []
    for i in range(0, len(y) - time_step):    #1~7일 데이터로 8일 Label을 예측 / 2~8일 데이터로 9일 Label을 예측 ......
        _x = x[i:i + time_step]
        _y = y[i + time_step] 
        dataX.append(_x)
        dataY.append(_y)

    train_size = int(len(dataY) * 0.6)
    val_size = int(len(dataY) * 0.8)

    trainX, valX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:val_size]), np.array(dataX[val_size:len(dataX)])
    trainY, valY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:val_size]), np.array(dataY[val_size:len(dataY)])

    trainX = np.reshape(trainX, (trainX.shape[0], time_step, len(df.values[0])))    
    testX = np.reshape(testX, (testX.shape[0], time_step, len(df.values[0])))

    early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 60, mode = 'min')
    mc = ModelCheckpoint('가중치 저장소/best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
    model = Sequential() 
    model.add(LSTM(256, input_shape = (trainX.shape[1], trainX.shape[2]), return_sequences = True))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(LSTM(256, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(LSTM(128, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(LSTM(64, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(LSTM(32, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    hist = model.fit(trainX, trainY, epochs=1000, batch_size=8, validation_data=(valX, valY), callbacks=[early_stopping, mc])

    model.load_weights('가중치 저장소/best_model.h5')
    y_predict = model.predict(testX)
    y_predict = y_predict.reshape(-1, 1).astype('float32')
    real_y = testY.reshape(-1, 1).astype('float32')
    
    import statsmodels.api as sm
    stn = pd.read_csv("최종위치.csv", encoding='ms949')
    stn = stn.iloc[:,0].values
    pred = model.predict(testX)
    real = testY
    pred = pred.reshape(-1)
    real = real.reshape(-1)

    pred = model.predict(testX)
    real = testY
    pred = pred.reshape(-1)
    real = real.reshape(-1)
    raw= {'Observed': list(real), 'Predicted': list(pred)}
    rr = pd.DataFrame(raw)
    rr.to_csv("lstmrsquared/%s_rsquared.csv" %stn[j])
    reg = sm.OLS.from_formula("Observed ~ Predicted",rr).fit()
    lstm_rs.append(reg.rsquared)
    
    # RMSE
    # nan이 나와서 value errror가 나면 rmse 999로 리턴
    try:
        rmse = sqrt(mean_squared_error(y_predict, real_y))
        rmse_lst.append(rmse)
    except ValueError:
        pass
    print("end")
print(time.time()-stst)