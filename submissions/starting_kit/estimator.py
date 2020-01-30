
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def get_estimator():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1., solver='liblinear')
    )
