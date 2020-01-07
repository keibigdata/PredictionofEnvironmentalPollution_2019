import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
import os
import time

from keras.models import Sequential
from keras.layers import Conv2D, Conv3D, Reshape,BatchNormalization, Activation, LSTM, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import keras

def division(data):
    train_size = int(len(data)*0.6)
    val_size = int(len(data)*0.8)
    data_train = data[0:train_size]
    data_val = data[train_size:val_size]
    data_test = data[val_size:len(data)]
    return data_train, data_val, data_test

def look_back(X, Y, a):
    X_lb = np.zeros((len(X) - a, a, 40, 28, 10))
    for l in range(len(X) - a):
        for r in range(a):
            X_lb[l, r] = X[l + r, 0]
    Y = Y[a:]
    return X_lb, Y
            
def my_model():
    model = Sequential()
    model.add(Conv3D(64, (12, 1, 1), padding='valid', input_shape=(12, 40, 28, 10)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Reshape((40,28,64)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(8, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(1, (3, 3), padding='same'))
    return model


path_dir = '/미세먼지/IDW/'
file_list = os.listdir(path_dir)

concat = []
dataset = []
flg = 0
i = 0
for f in file_list:
    st = time.time()
    df = pd.read_csv(path_dir + f, engine='python')
    del df['Unnamed: 0']

    df_flat = df.values.flatten()
    df_flat_rs = df_flat.reshape(len(df_flat), 1)

    if (i == 0):
        concat = df_flat_rs
    else:
        concat = np.concatenate((concat, df_flat_rs), axis=1)
    i = i+1
    print(time.time()-st)

if(flg==0):
    dataset = concat
else:
    dataset = np.concatenate((dataset, concat), axis=0)
flg = flg+1

# 합쳐진 데이터를 reshape (총 시간, timestep (현재는 설정 안했으므로 1), x축, y축, 채널(변수개수))
data = np.reshape(dataset, (8760, 1, 40, 28, 10))


var = ['CO', 'NO2', 'O3', 'PM10', 'PM25', 'SO2']

# 변수
for i in range(0,6):
    early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 20, mode = 'min')
    mc = ModelCheckpoint('가중치 저장소/best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
    if (i ==4) or (i==3):
        continue
    print("----- %s 변수를 독립변수로 -----" %var[i])
    X_data = data
    Y_data = data[:,:,:,:,i]
    Y_data = Y_data.reshape(8760,1,40,28,1)
    
    # time step
    X_data, Y_data = look_back(X_data, Y_data, 12)
    X_train, X_val, X_test = division(X_data)
    Y_train, Y_val, Y_test = division(Y_data)
    Y_train = Y_train.reshape(Y_train.shape[0], 40, 28, 1).astype('float32')
    Y_val = Y_val.reshape(Y_val.shape[0],  40, 28, 1).astype('float32')
    Y_test = Y_test.reshape(Y_test.shape[0], 40, 28, 1).astype('float32')
    model = my_model()
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, Y_train, epochs=2000, batch_size=512, validation_data=(X_val, Y_val), callbacks=[early_stopping, mc])
    
    # 예측값 생성
    model.load_weights('가중치 저장소/best_model.h5')
    y_predict = model.predict(X_test)

    # 예측값 0행렬 생성
    y_predict = model.predict(X_test)
    y_predict = y_predict.reshape(-1, 1).astype('float32')
    Y_test = Y_test.reshape(-1, 1).astype('float32')

    try:
        rmse = sqrt(mean_squared_error(y_predict, Y_test))
        
    except ValueError:
        rmse = 9999
    model.save('가중치 저장소/%s_TS%s_LAG%s_%s.h5' %(var[i], timestep, lag, rmse))
    del model, X_data, Y_data, X_train, X_val, X_test