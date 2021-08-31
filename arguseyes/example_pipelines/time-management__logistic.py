#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

# random seed for reproducibility
seed = 123456789
np.random.seed(seed)

raw_data = pd.read_csv("datasets/time-management.csv")

# Question 6: You often feel that your life is aimless, with no definite purpose
target_column = '6'
raw_data = raw_data[raw_data[target_column] != 'Neither']
raw_data = raw_data[raw_data[target_column].notna()]
raw_data[target_column] = raw_data[target_column].replace('Strong Agree', 'Agree')
raw_data[target_column] = raw_data[target_column].replace('Strong Disagree', 'Disagree')

raw_data['label'] = (raw_data[target_column] == 'Agree')

impute_and_one_hot = Pipeline([
    ('impute', SimpleImputer(strategy= 'most_frequent')),
    ('encode', OneHotEncoder(handle_unknown='ignore'))
])

feature_encoding = ColumnTransformer([
    ("impute_and_one_hot", impute_and_one_hot, ['Course', 'Academic', 'Attendance', 'English', 'Age',  
                                                '8', '12', '14', '15'])
])

pipeline = Pipeline([
    ('features', feature_encoding),
    ('sgdclassifier', SGDClassifier(
        random_state=1,
        loss='log',

        # Params chosen via grid search cross-validation
        alpha=0.1,
        eta0=0.01,
        penalty='elasticnet',
    ))
])

train_data, test_data = train_test_split(raw_data, test_size=.3, random_state=seed)
X_train, y_train_raw = train_data, train_data[target_column]
X_test, y_test_raw = test_data, test_data[target_column]

y_train = np.squeeze(label_binarize(y_train_raw, classes=['Agree', 'Disagree']))
y_test = np.squeeze(label_binarize(y_test_raw, classes=['Agree', 'Disagree']))

model = pipeline.fit(X_train, y_train)

# score_train = model.score(X_train, y_train)
score_test = model.score(X_test, y_test)
# print("Train score:", score_train)
print("Test score:", score_test)
