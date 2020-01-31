import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit

problem_title = 'Forest type classification'
_target_column_name = 'Cover_Type'
_ignore_column_names = []
_prediction_label_names = [1, 2, 3, 4, 5, 6, 7]
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)
# An object implementing the workflow
workflow = rw.workflows.Estimator()

score_types = [
    rw.score_types.Accuracy(name='acc'),
    rw.score_types.BalancedAccuracy(name='bac', precision=3),
    rw.score_types.NegativeLogLikelihood(name='nll'),
]


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=57)
    return cv.split(X, y)


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name), compression='gzip')
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name] + _ignore_column_names, axis=1)

    test = os.getenv('RAMP_TEST_MODE', 0)
    if test:
        return X_df[::30], y_array[::30]
    else:
        return X_df, y_array

    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv.gz'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv.gz'
    return _read_data(path, f_name)
