import tensorflow as tf # for model arc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import pandas as pd # for split dataset
from sklearn.model_selection import train_test_split
import numpy as np # for predict or test the model
import pickle
import os

dataset = pd.read_csv('project/datasetKgCO2.csv', delimiter=',', header=0)

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

# Split dataset into train and validation
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)

# Model architecture
model = Sequential([
    Dense(64, activation='relu', input_shape=(None, 1)),
    Dense(32, activation='relu'),
    LSTM(4, activation='relu'),
    Dense(32, activation='relu'),
    Dense(4, activation='linear')])

# Optimizer and loss for the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['acc'])

# Training the model
model.fit(tf.expand_dims(x_train, axis=-1), y_train, epochs=100, validation_data=(x_val, y_val), steps_per_epoch=10)

pickle.dump(model, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
