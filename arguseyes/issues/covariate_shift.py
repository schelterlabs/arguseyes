import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score


def detect(classification_pipeline) -> bool:

    X_train = classification_pipeline.X_train
    X_test = classification_pipeline.X_test

    X_train_test_labeled = np.append(X_train, X_test, axis=0)
    y_train_test_labeled = np.append(np.zeros((len(X_train), 1)), np.ones((len(X_test), 1)), axis=0)

    X_shift_train, X_shift_test, y_shift_train, y_shift_test = \
        train_test_split(X_train_test_labeled, y_train_test_labeled, test_size=0.2)

    grid = {
        'n_estimators': [10, 100],
        'min_samples_leaf': [1, 5],
    }

    clf = GridSearchCV(RandomForestClassifier(), grid)
    clf = clf.fit(X_shift_train, y_shift_train)

    y_shift_predicted = clf.predict_proba(X_shift_test)
    auc = roc_auc_score(y_shift_test, y_shift_predicted[:, 1])

    covariate_shift = auc > 0.7

    if covariate_shift:
        print("Covariate shift between train and test?", covariate_shift, 'AuC', auc)

    return covariate_shift
