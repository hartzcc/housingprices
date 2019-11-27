import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn import preprocessing

# Callback function
DESIRED_ACCURACY = 0.99

# Read in Data.
data_dir = '/home/me/Data_Source/hp/data/'
train_data = data_dir + 'train.csv'
test_data = data_dir + 'test.csv'

# Remove non-numerical data (Don't know how to handle categorical data yet)
train_df = pd.read_csv(train_data)
train_df = train_df.select_dtypes(include=['int64'])
train_df = train_df.dropna()

test_df = pd.read_csv(test_data)
test_df = train_df.select_dtypes(include=['int64'])
test_df = test_df.dropna()

# Normalize Data
train_df = train_df.astype(float)
min_max_scalar = preprocessing.MinMaxScaler()
x_scaled = min_max_scalar.fit_transform(train_df)
train_df_normalized = pd.DataFrame(x_scaled)

test_df = test_df.astype(float)
x_scaled = min_max_scalar.fit_transform(test_df)
test_df_normalized = pd.DataFrame(x_scaled)

# Break out solutions.  Column 34 for train_df_normalized is the price we are trying predict.  It is normalized so

y_train_df = train_df_normalized[34]
y_test_df = test_df.pop('SalePrice')

# Compile NN
model = Sequential()
model.add(Dense(8, input_dim=35, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer=tf.keras.optimizers.RMSprop(0.001), metrics=['mse'])

x = train_df_normalized.to_numpy()
y = y_train_df.to_numpy()

x_test = test_df_normalized.to_numpy()

model.fit(x, y, epochs=10, batch_size=10)

predictions = model.predict(x_test)

predictions = pd.DataFrame(predictions, columns=['SalePrice'])
predictions.index.name='Id'
predictions = (720100.00 - 34900.00) * predictions + 34900

# summarize the first 5 cases
# to get the original value back you need Max - Min * value + Min or (($720,100 - $34,900) * value) + $34,900.
predictions.to_csv('/home/me/Data_Source/hp/data/predictions.csv', index=True)

print(predictions.head())
print(y_test_df.head())
