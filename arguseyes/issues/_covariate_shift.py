import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

from arguseyes.issues import IssueDetector, Issue
from arguseyes.templates import Output


def evaluate_domain_classifier(X_one, X_two):
    X_train_test_labeled = np.append(X_one, X_two, axis=0)
    y_train_test_labeled = np.append(np.zeros((len(X_one), 1)), np.ones((len(X_two), 1)), axis=0)

    X_shift_train, X_shift_test, y_shift_train, y_shift_test = \
        train_test_split(X_train_test_labeled, y_train_test_labeled, test_size=0.2)

    # TODO enable again
    # grid = {
    #    'n_estimators': [10, 100],
    #    'min_samples_leaf': [1, 5],
    # }
    # clf = GridSearchCV(RandomForestClassifier(), grid)
    clf = RandomForestClassifier()
    clf = clf.fit(X_shift_train, y_shift_train)

    y_shift_predicted = clf.predict_proba(X_shift_test)
    auc = roc_auc_score(y_shift_test, y_shift_predicted[:, 1])

    return auc


class CovariateShift(IssueDetector):

    def _detect(self, pipeline) -> Issue:
        X_train = pipeline.outputs[Output.X_TRAIN]
        X_test = pipeline.outputs[Output.X_TEST]

        auc = evaluate_domain_classifier(X_train, X_test)

        auc_threshold = 0.7
        covariate_shift = auc > auc_threshold

        return Issue('covariate_shift', covariate_shift, {'test_size': 0.2, 'auc': auc, 'auc_threshold': auc_threshold})


    def error_msg(self, issue) -> str:
        return f'Found {issue.id}, random forest classifier could distinguish between train\n' + \
               f'and test set with a AuC of {issue.details["auc"]} > {issue.details["auc_threshold"]}!'

