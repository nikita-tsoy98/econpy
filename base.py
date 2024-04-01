import numpy as np

def get_const_err(res, inv_cov, X, ddof=0):
    sigma2 = (res**2).sum(1) / (res.shape[1] - ddof)
    return np.tensordot(sigma2, inv_cov, axes=0)

def get_hc0_err(res, inv_cov, X, ddof=0):
    return inv_cov @ np.inner((res**2)[:, None] * X[None, :], X) @ inv_cov

def get_hc0_err_reduced_mem(res, inv_cov, X, ddof=0):
    return inv_cov @ np.einsum('ji,ki,li', res**2, X, X) @ inv_cov

def get_hc1_err(res, inv_cov, X, ddof=0):
    alpha = res.shape[1] / (res.shape[1] - ddof)
    return get_hc0_err(res, alpha**0.5 * inv_cov, X)

def get_hc2_err(res, inv_cov, X, ddof=0):
    alpha = 1 / (1 - ((inv_cov @ X) * X).sum(0))
    return (inv_cov\
            @ np.inner((alpha * res**2)[:, None] * X[None, :], X)\
            @ inv_cov)

def get_hc3_err(res, inv_cov, X, ddof=0):
    alpha = 1 / (1 - ((inv_cov @ X) * X).sum(0))
    gamma = np.inner((alpha * res), X) / res.shape[1]**0.5
    err_mat = (np.inner((alpha * res**2)[:, None] * X[None, :], X)\
               - (gamma[:, None, :] * gamma[:, :, None]))
    return ((res.shape[1]-1) / res.shape[1] * inv_cov) @ err_mat @ inv_cov

def estimate_ols(Y, X, get_err_fn=get_hc1_err, ddof=0):
    inv_cov = np.linalg.inv(np.inner(X, X))
    prm = np.inner(Y, X) @ inv_cov
    res = Y - prm @ X
    prm_err = get_err_fn(res, inv_cov, X, prm.size + ddof)
    return prm, prm_err

def estimate_wls(Y, X, w, get_err_fn=get_hc1_err, ddof=0):
    sqrt_w = w**0.5
    return estimate_ols(sqrt_w * Y, sqrt_w * X, get_err_fn, ddof)

def estimate_tsls(Y, X, Z, get_err_fn=get_hc1_err, ddof=0):
    # first stage
    inv_cov_first = np.linalg.inv(np.inner(Z, Z))
    cov_XZ = np.inner(X, Z)
    prm_first = cov_XZ @ inv_cov_first
    X_hat = prm_first @ Z
    prm_err_first = get_err_fn(X-X_hat, inv_cov_first, Z, prm_first.size+ddof)

    # second stage
    inv_cov = np.linalg.inv(np.inner(prm_first, cov_XZ))
    prm = np.inner(Y, X_hat) @ inv_cov
    prm_err = get_err_fn(Y-prm@X, inv_cov, X_hat, prm.size+prm_first.size+ddof)
    return prm, prm_err, prm_first, prm_err_first

def estimate_wtsls(Y, X, Z, w, get_err_fn=get_hc1_err, ddof=0):
    sqrt_w = w**0.5
    return estimate_tsls(sqrt_w * Y, sqrt_w * X, sqrt_w * Z, get_err_fn, ddof)