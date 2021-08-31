#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# random seed for reproducibility
seed = 123456789
np.random.seed(seed)

raw_data = pd.read_csv("International students Time management data.csv")

#Question 6: You often feel that your life is aimless, with no definite purpose
label = '6'
raw_data = raw_data[raw_data[label] != 'Neither']
raw_data = raw_data[raw_data[label].notna()]
raw_data[label] = raw_data[label].replace('Strong Agree', 'Agree')
raw_data[label] = raw_data[label].replace('Strong Disagree', 'Disagree')

raw_data['label'] = (raw_data[label] == 'Agree')

impute_and_one_hot = Pipeline([
    ('impute', SimpleImputer(strategy= 'most_frequent')),
    ('encode', OneHotEncoder(handle_unknown='ignore'))
])

feature_encoding = ColumnTransformer([
    ("impute_and_one_hot", impute_and_one_hot, ['Course', 'Program', 'Attendance',
                                                '8','12', '14', '15']),
])

pipeline = Pipeline([
    ('features', feature_encoding),
    ('tree', DecisionTreeClassifier(
            random_state=1,

            # Params chosen via grid search cross-validation
            max_depth=5,
            max_features='auto',
            max_leaf_nodes=6,
            min_samples_split=5,
        )
    )
])

train_data, test_data = train_test_split(raw_data, test_size=.3, random_state=seed)
X_train, y_train_raw = train_data, train_data[label]
X_test, y_test_raw = test_data, test_data[label]

y_train = np.squeeze(label_binarize(y_train_raw, classes=['Agree', 'Disagree']))
y_test = np.squeeze(label_binarize(y_test_raw, classes=['Agree', 'Disagree']))

pipeline.fit(X_train, y_train)

# score_train = pipeline.score(X_train, y_train)
score_test = pipeline.score(X_test, y_test)
# print("Train score:", score_train)
print("Test score:", score_test)
