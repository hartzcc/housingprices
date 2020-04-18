import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential

# Callback function
DESIRED_ACCURACY = 0.99


def compile_fit(x, y):
    model = Sequential()
    model.add(Dense(8, input_dim=37, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer=tf.keras.optimizers.RMSprop(0.001), metrics=['mse'])
    model.fit(x, y, epochs=10, batch_size=10)
    return model
