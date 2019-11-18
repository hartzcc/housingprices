import tensorflow as tf
import keras
import pandas as pd
from sklearn import preprocessing
import numpy as np

# Keras model for learning Kaggle Housing prices

def hp_model():

    DESIRED_ACCURACY = 0.99

    # Callback Training Function. Limits tensorflow training to 99% to speed up iterations

    class MyCallBack(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs is None:
                logs = {}
            if logs.get('acc') > DESIRED_ACCURACY:
                print(f"chr(10) Accuracy = {DESIRED_ACCURACY}. Exiting")
                self.model.stop_training = True

    callback = MyCallBack()

    # Read in Data.
    data_dir = '/home/me/Data_Source/hp/data/'
    train_data = data_dir + 'train.csv'
    test_data = data_dir + 'test.csv'

    # Remove non-numerical data (Don't know how to handle categorical data yet)
    train_df = pd.read_csv(train_data)
    train_df = train_df.select_dtypes(include=['int64'])

    test_df = pd.read_csv(test_data)
    test_df = train_df.select_dtypes(include=['int64'])

    # Normalize Data
    train_df = train_df.astype(float)
    min_max_scalar = preprocessing.MinMaxScaler()
    x_scaled = min_max_scalar.fit_transform(train_df)
    train_df_normalized = pd.DataFrame(x_scaled)

    test_df = test_df.astype(float)
    x_scaled = min_max_scalar.fit_transform(test_df)
    test_df_normalized = pd.DataFrame(x_scaled)

    print(train_df_normalized)








    return




if __name__ == "__main__":
    # execute only if run as a script
    hp_model()