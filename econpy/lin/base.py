import numpy as np

def regress(*Xs, preprocess_fn, estimator_fn, errors_fn):
    Xs_new = preprocess_fn(*Xs)
    return estimator_fn(*Xs_new, errors_fn)