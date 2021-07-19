import pandas as pd
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer


def load_train_and_test_data(adult_train_location, adult_test_location, excluded_employment_types):

    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
               'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
               'native-country', 'income-per-year']

    adult_train = pd.read_csv(adult_train_location, names=columns, sep=', ', engine='python', na_values="?")
    adult_test = pd.read_csv(adult_test_location, names=columns, sep=', ', engine='python', na_values="?", skiprows=1)

    for employment_type in excluded_employment_types:
        adult_train = adult_train[adult_train['workclass'] != employment_type]
        adult_test = adult_test[adult_test['workclass'] != employment_type]

    return adult_train, adult_test


def create_feature_encoding_pipeline():

    def safe_log(x):
        return np.log(x, out=np.zeros_like(x), where=(x != 0))

    impute_and_one_hot_encode = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OneHotEncoder(sparse=False, handle_unknown='ignore'))
    ])

    impute_and_scale = Pipeline([
        ('impute', SimpleImputer(strategy='mean')),
        ('log_transform', FunctionTransformer(lambda x: safe_log(x))),
        ('scale', StandardScaler())
    ])

    featurisation = ColumnTransformer(transformers=[
        ("impute_and_one_hot_encode", impute_and_one_hot_encode, ['workclass', 'education', 'occupation']),
        ('impute_and_scale', impute_and_scale, ['age', 'capital-gain', 'capital-loss', 'hours-per-week']),
    ], remainder='drop')

    return featurisation


def extract_labels(adult_train, adult_test):
    adult_train_labels = label_binarize(adult_train['income-per-year'], classes=['<=50K', '>50K'])
    # The test data has a dot in the class names for some reason...
    adult_test_labels = label_binarize(adult_test['income-per-year'], classes=['<=50K.', '>50K.'])

    return adult_train_labels, adult_test_labels


def create_training_pipeline():
    featurisation = create_feature_encoding_pipeline()
    return Pipeline([
        ('features', featurisation),
        ('learner', SGDClassifier(loss='log'))
    ])


train_location = 'datasets/income/adult.data'
test_location = 'datasets/income/adult.test'

government_employed = ['Federal-gov', 'State-gov']

train, test = load_train_and_test_data(train_location, test_location, excluded_employment_types=government_employed)


train_labels, test_labels = extract_labels(train, test)
pipeline = create_training_pipeline()

model = pipeline.fit(train, train_labels)

score = model.score(test, test_labels)

print("Model accuracy on held-out data", score)
