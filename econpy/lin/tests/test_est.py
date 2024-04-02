import numpy as np
from econpy.lin import est


def gen_wls_data(d, n, rng):
    X = rng.standard_normal((d, n))
    beta = rng.uniform(0, 1, (1, d))
    w = rng.uniform(0, 1, n)
    return X, beta, w


def gen_wls_case(X, beta, rng):
    Y_err = rng.standard_normal((1, X.shape[1]))
    return beta @ X + Y_err


def gen_wtsls_data(d, r, n, rng):
    Z = rng.standard_normal((r, n))
    gamma = rng.uniform(0, 1, (d, r))
    beta = rng.uniform(0, 1, (1, d))
    beta_bias = rng.uniform(0, 1, (1, d))
    w = rng.uniform(0, 1, n)
    return Z, gamma, beta, beta_bias, w


def gen_wtsls_case(Z, gamma, beta, beta_bias, rng):
    X_err = rng.standard_normal((beta.shape[1], Z.shape[1]))
    X = gamma @ Z + X_err
    Y_err = rng.standard_normal((1, Z.shape[1]))
    Y = (beta @ gamma) @ Z + beta_bias @ X_err + Y_err
    return Y, X


def test():
    seed = 42
    rng = np.random.default_rng(seed)

    # tests of wls and ols

    n = 10001
    r = 10
    d = 6

    tries = 3

    for _ in range(tries):
        X, beta, w = gen_wls_data(d, n, rng)
        Y = gen_wls_case(X, beta, rng)
        print("OLS estimate", est.ols(Y, X)[0])
        print("WLS estimate", est.wls(Y, X, w)[0])
        print("Ground truth", beta, end="\n\n")

    for _ in range(tries):
        Z, gamma, beta, beta_bias, w = gen_wtsls_data(d, r, n, rng)
        Y, X = gen_wtsls_case(Z, gamma, beta, beta_bias, rng)
        print(" TSLS estimate", est.tsls(Y, X, Z)[0])
        print("WTSLS estimate", est.wtsls(Y, X, Z, w)[0])
        print("Ground truth  ", beta, end="\n\n")
