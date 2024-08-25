import torch
from econpy.lin import base, est
from econpy.lin.tests import test_est


def test(err_fn):
    base.set_deterministic_and_all_seed(42)

    # tests of wls and ols

    n = 301
    r = 10
    d = 6

    tries = 3
    reps = 1000

    print(
        f"Ideal error rate: {0.6827:.2%} (higher => too conservative)",
        end="\n\n")

    for _ in range(tries):
        err_ols, err_wls = 0, 0
        x, beta, w = test_est.gen_ols_data(d, n)
        for _ in range(reps):
            y = test_est.gen_ols_case(x, beta)
            prm, _, _, prm_err = est.ols_with_err(
                y, x, err_fn=err_fn)
            dev = (prm - beta).abs()
            err_ols += dev[0][0] < prm_err[0].diagonal()[0]**0.5
            prm, _, _, prm_err = est.ols_with_err(
                y, x, w, err_fn=err_fn)
            dev = (prm - beta).abs()
            err_wls += dev[0][0] < prm_err[0].diagonal()[0]**0.5
        print(f"Error rate OLS: {err_ols/reps:.2%}")
        print(f"Error rate WLS: {err_wls/reps:.2%}")
    print()

    for _ in range(tries):
        err_tsls, err_wtsls = 0, 0
        z, gamma, beta, beta_bias, w = test_est.gen_tsls_data(d, r, n)
        for _ in range(reps):
            y, x = test_est.gen_tsls_case(z, gamma, beta, beta_bias)
            prm, _, _, _, _, _, prm_err, _\
                = est.tsls_with_err(y, x, z, err_fn=err_fn)
            dev = (prm - beta).abs()
            err_tsls += dev[0][0] < prm_err[0].diagonal()[0]**0.5
            prm, _, _, _, _, _, prm_err, _\
                = est.tsls_with_err(y, x, z, w, err_fn=err_fn)
            dev = (prm - beta).abs()
            err_wtsls += dev[0][0] < prm_err[0].diagonal()[0]**0.5
        print(f"Error rate  TSLS: {err_tsls/reps:.2%}")
        print(f"Error rate WTSLS: {err_wtsls/reps:.2%}")
