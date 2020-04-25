#%%
import pandas as pd
import hp_model as hpm
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np



#%%
# Get Data Sets
data_dir = '/Users/dev/PycharmProjects/housingprices/data_source/hp/'
X_full = pd.read_csv(data_dir + 'train.csv', index_col='Id')
X_test_full = pd.read_csv(data_dir + 'test.csv', index_col='Id')

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop('SalePrice', axis=1, inplace=True)
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.8,
                                                                test_size=0.2, random_state=0)

#%%
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 10 and
                    X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if
                X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

#%%.
# Data Pipeline

# Numerical and Categorical Data
numerical_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('one_hot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

#%%

clf = Pipeline(steps=[('Preprocessor', preprocessor),
                      ('model', hpm.create_model())
                      ])

X_numpy = X_train.to_numpy()
y_numpy = y_train.to_numpy()

clf.fit(X_numpy, y_numpy)


#%%
# Make Predictions
predictions = clf.predict(X_valid)

# Create Output File
predictions = pd.DataFrame(predictions, columns=['SalePrice'])
predictions.index.name = 'Id'
predictions.index += 1461
predictions = (720100.00 - 34900.00) * predictions + 34900
print("Test Predictions:", predictions.head())

# to get the original value back you need Max - Min * value + Min or (($720,100 - $34,900) * value) + $34,900.
predictions.to_csv('/Users/dev/PycharmProjects/housingprices/predictions.csv', index=True)

