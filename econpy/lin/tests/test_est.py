import torch
from econpy.lin import est, base


def gen_ols_data(d, n):
    x = torch.empty((d, n), device=base.CUDA).normal_()
    w = torch.rand((n,), device=base.CUDA)
    prm = torch.rand((1, d), device=base.CUDA)
    return x, prm, w


def gen_ols_case(x, prm):
    y_err = torch.empty((1, x.shape[1]), device=base.CUDA).normal_()
    return prm @ x + y_err


def gen_tsls_data(d, r, n):
    z = torch.empty((r, n), device=base.CUDA).normal_()
    w = torch.rand((n,), device=base.CUDA)
    gamma = torch.rand((d, r), device=base.CUDA)
    beta = torch.rand((1, d), device=base.CUDA)
    beta_bias = torch.rand((1, d), device=base.CUDA)
    return z, gamma, beta, beta_bias, w


def gen_tsls_case(z, gamma, beta, beta_bias):
    x_err = torch.empty(
        (beta.shape[1], z.shape[1]), device=base.CUDA).normal_()
    x = gamma @ z + x_err
    y_err = torch.empty((1, z.shape[1]), device=base.CUDA).normal_()
    y = (beta @ gamma) @ z + beta_bias @ x_err + y_err
    return y, x


def test():
    base.set_deterministic_and_all_seed(42)

    # tests of wls and ols

    n = 10001
    r = 10
    d = 6

    tries = 3

    for _ in range(tries):
        x, prm, w = gen_ols_data(d, n)
        y = gen_ols_case(x, prm)
        print("OLS estimate", est.ols(y, x)[0])
        print("WLS estimate", est.ols_with_err(y, x, w)[0])
        print("Ground truth", prm, end="\n\n")

    for _ in range(tries):
        z, gamma, beta, beta_bias, w = gen_tsls_data(d, r, n)
        y, x = gen_tsls_case(z, gamma, beta, beta_bias)
        print(" TSLS estimate", est.tsls(y, x, z)[0])
        print("WTSLS estimate", est.tsls_with_err(y, x, z, w)[0])
        print("Ground truth  ", beta, end="\n\n")