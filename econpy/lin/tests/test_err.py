import numpy as np
from econpy.lin import est
from econpy.lin.tests import test_est


def test(get_err_fn):
    seed = 42
    rng = np.random.default_rng(seed)

    # tests of wls and ols

    n = 301
    r = 10
    d = 6

    tries = 3
    reps = 1000

    for _ in range(tries):
        err_ols, err_wls = 0, 0
        X, beta, w = test_est.gen_wls_data(d, n, rng)
        for _ in range(reps):
            Y = test_est.gen_wls_case(X, beta, rng)
            prm, prm_err = est.ols(Y, X, get_err_fn)
            dev = np.abs(prm - beta)
            err_ols += dev[0][0] < prm_err[0].diagonal()[0]**0.5
            prm, prm_err = est.wls(Y, X, w, get_err_fn)
            dev = np.abs(prm - beta)
            err_wls += dev[0][0] < prm_err[0].diagonal()[0]**0.5
        print(
            f"Error rate OLS: {err_ols/reps:.2%}",
            f"Error rate WLS: {err_wls/reps:.2%}")
    print()

    for _ in range(tries):
        err_tsls, err_wtsls = 0, 0
        Z, gamma, beta, beta_bias, w = test_est.gen_wtsls_data(
            d, r, n, rng)
        for _ in range(reps):
            Y, X = test_est.gen_wtsls_case(
                Z, gamma, beta, beta_bias, rng)
            prm, prm_err, _, _ = est.tsls(Y, X, Z, get_err_fn)
            dev = np.abs(prm - beta)
            err_tsls += dev[0][0] < prm_err[0].diagonal()[0]**0.5
            prm, prm_err, _, _ = est.wtsls(
                Y, X, Z, w, get_err_fn)
            dev = np.abs(prm - beta)
            err_wtsls += dev[0][0] < prm_err[0].diagonal()[0]**0.5
        print(
            f"Error rate TSLS: {err_tsls/reps:.2%}",
            f"Error rate WTSLS: {err_wtsls/reps:.2%}")
