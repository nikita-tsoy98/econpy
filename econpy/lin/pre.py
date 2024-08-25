from collections.abc import Callable
from torch import Tensor

import torch
from econpy.lin import est


def absorb(y: Tensor, x: Tensor) -> Tensor:
    prm_yx = est.ols_only_prm(y, x)
    return y.sub_(torch.mm(prm_yx, x))


def absorb_fe(y: Tensor, fe_ind: Tensor, fe_count: Tensor) -> None:
    """
    Absorb one FE from the covariates in-place

    Parameters
    ----------
    y: Tensor(d, n)
        Vector of covariates of shape ``(d, n)``, where ``d`` is the
        number of covariates and ``n`` is the number of observations
    fe_ind: Tensor(d, n)
        Vector of FE
    fe_count: Tensor(k)
        Vector of the counts of different groups,
        where ``k`` is the number of groups

    Returns
    -------
    """
    fe_sum = torch.zeros(
        (y.shape[0], len(fe_count)), device=y.device, dtype=y.dtype)
    fe_sum.scatter_add_(1, fe_ind, y)
    fe_sum.div_(fe_count)
    y.sub_(fe_sum[:, fe_ind[0]])


def absorb_fe_trend(
        y: Tensor, fe_ind: Tensor, fe_cov: Tensor, fe_w: Tensor) -> None:
    """
    Absorb one local trend from the covariates in-place

    Parameters
    ----------
    y: Tensor(d, n)
        Vector of covariates of shape ``(d, n)``, where ``d`` is the
        number of covariates and ``n`` is the number of observations
    fe_ind: Tensor(d, n)
        Vector of FE
    fe_cov: Tensor(k)
        Vector of the covariates of different groups,
        where ``k`` is the number of groups
    fe_w: Tensor(n)
        Vector of the weights of local trends

    Returns
    -------
    """
    fe_sum = torch.zeros(
        (y.shape[0], len(fe_cov)), device=y.device, dtype=y.dtype)
    fe_sum.scatter_add_(1, fe_ind, y.mul(fe_w))
    fe_sum.div_(fe_cov)
    y.sub_(fe_sum[:, fe_ind[0]].mul_(fe_w))


def absorb_fes(
        y: Tensor, fes: Tensor, fe_trends: Tensor, fe_trend_ws: Tensor,
        eps: float = 2**(-11), delta: float = 1e-8, max_iter: int = 200
        ) -> Tensor:
    """
    Absorb fixed effects and local trends from covariates

    Parameters
    ----------
    y: Tensor(d, n)
        Vector of covariates of shape ``(d, n)``, where ``d`` is the
        number of covariates and ``n`` is the number of observations
    fes: Tensor(t, n)
        Vector of FEs, where ``t`` is the number of FEs
    fe_trends: Tensor(s, n)
        Vector of local trends, where ``s`` is the number of local trends
    fe_trend_w: Tensor(s, n)
        Vector of the weights of local trends
    eps: float
        Stopping criteria for absorbtion
    delta: float
        Parameter for stopping criteria regularization
    max_iter: int
        Maximum number of iterations

    Returns
    -------
    y: Tensor(d, n)
        Vector of covariates after FEs absorbtion
    """
    if fes.numel():
        fe_inds, fe_counts = zip(*(
            torch.unique(fe, False, True, True)[1:] for fe in fes))
    else:
        fe_inds = torch.empty(
            (0, y.shape[1]), device=y.device, dtype=torch.long)
        fe_counts = torch.empty(0, device=y.device, dtype=torch.long)
    if fe_trends.numel():
        fe_trend_groups, fe_trend_inds = zip(*(
            torch.unique(fe, False, True) for fe in fe_trends))
    else:
        fe_trend_inds = torch.empty(
            (0, y.shape[1]), device=y.device, dtype=torch.long)
    fe_trend_covs = tuple(
        torch.zeros(len(fe_group), device=y.device, dtype=y.dtype)\
            for fe_group in fe_trend_groups)
    for fe_ind, fe_w, fe_cov in zip(fe_trend_inds, fe_trend_ws, fe_trend_covs):
        fe_cov.scatter_add_(0, fe_ind, fe_w.square())
    fe_inds = tuple(fe_ind.expand((y.shape[0], -1)) for fe_ind in fe_inds)
    fe_trend_inds = tuple(
        fe_ind.expand((y.shape[0], -1)) for fe_ind in fe_trend_inds)

    with torch.no_grad():
        scale_y = y.abs().median(1).values.clamp_min_(delta)
    curr_eps = 2 * eps
    num_iter = 0
    while curr_eps > eps:
        num_iter += 1
        with torch.no_grad():
            z = y.clone()
        for fe_ind, fe_count in zip(fe_inds, fe_counts):
            absorb_fe(y, fe_ind, fe_count)
        for fe_ind, fe_cov, fe_w in zip(
            fe_trend_inds, fe_trend_covs, fe_trend_ws):
            absorb_fe_trend(y, fe_ind, fe_cov, fe_w)
        with torch.no_grad():
            z.sub_(y)
            curr_eps = z.abs().max(1).values.div_(scale_y).max()
        if num_iter > max_iter:
            print(
                "absorb_fes error: can not absorb fixed effects\
                    with the required precision. Try normalizing covariates.")
            break
    return y