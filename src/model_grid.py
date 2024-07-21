from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

hyper_parameters = [
        {
            'clf__estimator': [LogisticRegression()],
            'clf__estimator__penalty': ('l2', 'elasticnet', 'l1'),
            'clf__estimator__C': (0.01, 1, 10)
        },
        # {
        #     'clf__estimator': [RandomForestClassifier()],
        #     'clf__estimator__n_estimators': (25, 50, 100, 150),
        #     'clf__estimator__max_features': ('sqrt', 'log2', None),
        #     'clf__estimator__max_depth': (3, 6, 9),
        #     'clf__estimator__max_leaf_nodes': (3, 6, 9),
        # },
        # {
        #     'clf__estimator': [SVC()],
        #     'clf__estimator__C': (0.1, 1, 10, 100, 1000),
        #     'clf__estimator__gamma': (1, 0.1, 0.01, 0.001, 0.0001),
        #     'clf__estimator__kernel': ('linear', 'poly', 'sigmoid','rbf')
        # }
    ]