import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from datetime import datetime, date
import pandas as pd
import matplotlib.pyplot as plt


import reader

def convert_to_matrix(data, step):
    X, Y = [], []
    for i in range(len(data) - step):
        d = i + step  
        X.append(data[i:d,])
        Y.append(data[d,])
    return np.array(X), np.array(Y)

usd = reader.read_usd()
usd.date = usd.date.apply(lambda x: datetime.strptime(x, '%d.%m.%Y').date())
usd.sort_values(by='date', inplace=True)

seq = usd.price.values
seq = seq.reshape(seq.shape[0], 1)
ts = int(0.6 * seq.size)
train, test = seq[:ts, :], seq[ts:, :]

step = 2

train = np.append(train, np.repeat(train[-1,], step))
test = np.append(test, np.repeat(test[-1,], step))

X_train, y_train = convert_to_matrix(train, step)
X_test, y_test = convert_to_matrix(test, step)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

model = keras.Sequential()
model.add(SimpleRNN(units=32, input_shape=(1, step), activation="relu"))
model.add(Dense(8, activation="relu")) 
model.add(Dense(1))
model.compile(loss='mse', optimizer='rmsprop')
model.summary()

history = model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=2)
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)
predicted = np.concatenate((trainPredict,testPredict), axis=0)

trainScore = model.evaluate(X_train, y_train, verbose=0)
testScore = model.evaluate(X_test, y_test, verbose=0)
print(trainScore, testScore)

index = usd.price.index.values
plt.plot(index, usd.price)
plt.plot(index, predicted)
plt.axvline(usd.price.index[ts], c="r")
plt.show()

