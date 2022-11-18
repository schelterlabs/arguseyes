# https://www.openml.org/search?type=flow&id=8774
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer


def denormalize(records, workclasses, education, occupation, sex, race):
    data = records.merge(workclasses, on='workclass_id')
    data = data.merge(education, on='education_id', how='left')
    data = data.merge(occupation, on='occupation_id', how='left')
    data = data.merge(sex, on='sex_id', how='left')
    data = data.merge(race, on='race_id', how='left')
    return data


def load_train_and_test_data(data_location, employment_types):

    train = pd.read_csv(f'{data_location}/income_train.csv')
    test = pd.read_csv(f'{data_location}/income_test.csv')

    workclasses = pd.read_csv(f'{data_location}/workclass.csv')
    education = pd.read_csv(f'{data_location}/education.csv')
    occupation = pd.read_csv(f'{data_location}/occupation.csv')
    sex = pd.read_csv(f'{data_location}/sex.csv')
    race = pd.read_csv(f'{data_location}/race.csv')

    workclasses = workclasses[workclasses.workclass.isin(employment_types)]

    train = denormalize(train, workclasses, education, occupation, sex, race)
    test = denormalize(test, workclasses, education, occupation, sex, race)

    return train, test


def extract_labels(train, test):
    train_labels = label_binarize(train['income-per-year'], classes=['<=50K', '>50K'])
    # The test data has a dot in the class names for some reason...
    test_labels = label_binarize(test['income-per-year'], classes=['<=50K.', '>50K.'])

    return train_labels, test_labels


# https://www.openml.org/search?type=flow&id=8774
def openmlflow(numerical_columns, categorical_columns):

    num_pipe = Pipeline([('imputer', SimpleImputer(add_indicator=True)),
                         ('standardscaler', StandardScaler())])
    cat_pipe = Pipeline([('simpleimputer', SimpleImputer(strategy='most_frequent')),
                         ('onehotencoder', OneHotEncoder())])
    return Pipeline([
        ('columntransformer', ColumnTransformer([
            ('num', num_pipe, numerical_columns),
            ('cat', cat_pipe, categorical_columns),
        ])),
        ('decisiontreeclassifier', DecisionTreeClassifier(random_state=0))])


data_location = 'datasets/income/'

included_employment = ['Federal-gov', 'State-gov', 'Local-gov']

# Fix for fairness issues, increase data to contain more employment types
# included_employment = ['Federal-gov', 'State-gov', 'Local-gov', 'Private']

train, test = load_train_and_test_data(data_location, employment_types=included_employment)


train_labels, test_labels = extract_labels(train, test)

categorical_columns = ['workclass', 'education', 'occupation']
numerical_columns = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']

openml_pipeline = openmlflow(numerical_columns, categorical_columns)

model = openml_pipeline.fit(train, train_labels)

score = model.score(test, test_labels)

print("Accuracy", score)
