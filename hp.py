import pandas as pd
import refine_data as rd
import hp_model as hpm
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer


def clean_data(df=None):
    my_imputer = SimpleImputer()
    min_max_scalar = preprocessing.MinMaxScaler()

    df = df.select_dtypes(include=['number'])
    col_names = df.columns
    df = pd.DataFrame(my_imputer.fit_transform(df))
    df = df.astype(float)

    x_scaled = min_max_scalar.fit_transform(df)
    df = pd.DataFrame(x_scaled)
    df.columns = col_names
    return df



data_dir = '/Users/dev/PycharmProjects/housingprices/data_source/hp/'
test_df = pd.read_csv(data_dir + 'test.csv')
data = pd.read_csv(data_dir + 'train.csv')

data = rd.clean_data(data)
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)


# Compile NN
# x = train_df_normalized.to_numpy()

model = hpm.compile_fit(X_train, y_train)

# Make Predictions
predictions = model.predict(test_df_normalized.to_numpy())
# predictions = model.predict(train_df_normalized.to_numpy())

# Create Output File
predictions = pd.DataFrame(predictions, columns=['SalePrice'])
predictions.index.name = 'Id'
predictions.index += 1461
predictions = (720100.00 - 34900.00) * predictions + 34900
print("Test Predictions:", predictions.head())

# to get the original value back you need Max - Min * value + Min or (($720,100 - $34,900) * value) + $34,900.
predictions.to_csv('/Users/dev/PycharmProjects/housingprices/predictions.csv', index=True)

