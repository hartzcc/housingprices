from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.wrappers.scikit_learn import KerasClassifier


def create_model(input_dims=227):
    model = Sequential()
    model.add(Dense(8, input_dim=input_dims, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer=RMSprop(0.001), metrics=['mse'])
    my_model = KerasClassifier(build_fn=model, epochs=50, batch_size=10, verbose=0)
    return my_model


