import tensorflow as tf
import keras
import pandas as pd
import sklearn
import numpy as np

# Keras model for learning Kaggle Housing prices




def hp_model():

    DESIRED_ACCURACY = 0.99

    # Callback Training Function. Limits tensorflow training to 99% to speed up iterations

    class MyCallBack(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('acc') > DESIRED_ACCURACY:
                print(f"chr(10) Accuracy = {DESIRED_ACCURACY}. Exiting")
                self.model.stop_training = True

    callback = MyCallBack()

    # Read in Data.
    data_dir = 'home/me/Data_Source/hp/data'
    train_data = data_dir + 'train.csv'
    test_data = data_dir + 'test.csv'

    train_df = pd.read_csv()



    print('hello')






    return




if __name__ == "__main__":
    # execute only if run as a script
    hp_model()