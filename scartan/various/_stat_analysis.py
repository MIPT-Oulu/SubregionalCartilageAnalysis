import numpy as np
from scipy import stats
import sklearn


def cohen_d(d1, d2):
    """
    https://machinelearningmastery.com/effect-size-measures-in-python/
    """
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(d1), np.mean(d2)
    # calculate the effect size
    return (u1 - u2) / s


def cohen_d_var(d, n1, n2):
    """
    https://trendingsideways.com/the-cohens-d-formula
    """
    m1 = (n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2 - 2))
    m2 = (n1 + n2) / (n1 + n2 - 2)
    v = m1 * m2
    return v


def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2


def linreg(x, y):
    valid_idcs = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid_idcs], y[valid_idcs]

    tmp = stats.linregress(x, y)
    return {"slope": tmp[0],
            "intercept": tmp[1],
            "rvalue": tmp[2],
            "r2": tmp[2] ** 2,
            "pvalue": tmp[3],
            "stderr": tmp[4]}


def odds_ratio(x, y):
    """

    Args:
        x: (num_samples, num_features) ndarray
        y: (num_samples, ) ndarray of bool

    Returns:
        odds_ratio: float
        pvalue: float
    """
    clf = sklearn.linear_model.LogisticRegression(penalty='none',
                                                  class_weight='balanced',
                                                  solver='newton-cg')
    clf.fit(x, y)
    y_pred = clf.predict(x)

    cm = sklearn.metrics.confusion_matrix(y, y_pred)
    odds_ratio, pvalue = stats.fisher_exact(cm)
    return odds_ratio, pvalue
