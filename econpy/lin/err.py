import numpy as np


def const(res, inv_cov, X, ddof=0):
    sigma2 = (res**2).sum(1) / (res.shape[1] - ddof)
    return np.tensordot(sigma2, inv_cov, axes=0)


def hc0(res, inv_cov, X, ddof=0):
    return inv_cov @ np.inner((res**2)[:, None] * X[None, :], X) @ inv_cov


def hc0_red_mem(res, inv_cov, X, ddof=0):
    return (inv_cov\
            @ np.einsum('ji,ki,li', res**2, X, X, optimize=True)\
            @ inv_cov)


def hc1(res, inv_cov, X, ddof=0):
    alpha = res.shape[1] / (res.shape[1] - ddof)
    return ((alpha * inv_cov)\
            @ np.inner((res**2)[:, None] * X[None, :], X)\
            @ inv_cov)


def hc2(res, inv_cov, X, ddof=0):
    alpha = 1 / (1 - ((inv_cov @ X) * X).sum(0))
    return (inv_cov\
            @ np.inner((alpha * res**2)[:, None] * X[None, :], X)\
            @ inv_cov)


def hc3(res, inv_cov, X, ddof=0):
    alpha = 1 / (1 - ((inv_cov @ X) * X).sum(0))
    gamma = np.inner((alpha * res), X) / res.shape[1]**0.5
    err_mat = (np.inner((alpha * res**2)[:, None] * X[None, :], X)\
               - (gamma[:, None, :] * gamma[:, :, None]))
    return ((res.shape[1]-1) / res.shape[1] * inv_cov) @ err_mat @ inv_cov