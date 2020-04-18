import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer


def clean_data(df=None):
    my_imputer = SimpleImputer()
    df = df.select_dtypes(include=['number'])
    col_names = df.columns
    df = pd.DataFrame(my_imputer.fit_transform(df))
    df.columns = col_names
    return df


def normalize_data(df=None):
    df = df.astype(float)
    min_max_scalar = preprocessing.MinMaxScaler()
    x_scaled = min_max_scalar.fit_transform(df)
    df = pd.DataFrame(x_scaled)
    return df
