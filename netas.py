#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pondoo
"""
from math import sqrt
import numpy as np
import datetime
import tensorflow as tf
from numpy.core._multiarray_umath import concatenate
import  pandas as pd
from sklearn.metrics import mean_absolute_error, max_error, median_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.callbacks import EarlyStopping
data = pd.read_csv("netas.csv")
date=data.iloc[:,0]
simdi=data.iloc[:,1]
acilis=data.iloc[:,2]
yuksek=data.iloc[:,3]
dusuk=data.iloc[:,4]
hacim=data.iloc[:,5]
for i in range(0,len(hacim)):
    hacim[i] = hacim[i].replace(',','.')
for i in range(0,len(hacim)):
    simdi[i] = simdi[i].replace(',','.')
for i in range(0,len(hacim)):
    acilis[i] = acilis[i].replace(',','.')
for i in range(0,len(hacim)):
    yuksek[i] = yuksek[i].replace(',','.')
for i in range(0,len(hacim)):
    dusuk[i] = dusuk[i].replace(',','.')

for i in range(0,len(hacim)):
    for a in range(0,len(hacim[i])):
        gecici=hacim[i]
        if(gecici[a]=='M'):
            gecici=gecici.replace('M','')
            gecici=float(gecici)
            hacim[i]=gecici*100000
        elif(gecici[a]=='K'):
            gecici=gecici.replace('K','')
            gecici=float(gecici)
            hacim[i]=gecici
    
dates=[]
for i in date:
    dates.append(i.split('.')[0])

new_data= np.array([simdi,dates,acilis,yuksek,dusuk,hacim])

new_data = new_data.transpose()

new_data = new_data.astype('float32')

X=new_data[:,1:6]
Y=new_data[:,0]
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X=min_max_scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.75, random_state = 1)

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
print(X_train.shape, X_train.shape, X_test.shape, X_test.shape)


from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
model = Sequential()
model.add(LSTM(64, return_sequences=True,input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(32))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=2, shuffle=False)

result=model.predict(X_test)

rmse = sqrt(mean_squared_error(y_test, result))
print('Test RMSE: %.3f' % rmse)

rkare = r2_score(y_test, result)
print('Test RKare: %.3f' % rkare)

mae = mean_absolute_error(y_test, result)
print('Test MAE: %.3f' % mae)

mae_2 = median_absolute_error(y_test, result)
print('Test MEDIAN ABSOLUTE ERROR: %.3f' % mae_2)

mpd = max_error(y_test, result)
print('Test MAX ERROR: %.3f' % mpd)
gercekDegerler = np.core.array(y_test)
tahminiDegerler = np.core.array(result)
import matplotlib.pyplot as plt
figs = plt.figure(figsize=(23,10))
plt.plot(gercekDegerler[0:300],  'bo',label = 'Gerçek Değerler',c='r', linewidth = 2)
plt.plot(tahminiDegerler[0:300],'b', label = 'Tahmin Değerleri', linewidth = 2)
plt.xlabel('Tahmin Sayısı', fontsize= 15)
plt.ylabel('Fiyatlar', fontsize= 15)
plt.legend(fontsize= 15)
plt.title("Gerçek Değer ve Tahmin Değerleri Analizi(İlk 300 değer İçin)", fontsize= 15)
