from torch import Tensor

import torch


def const(res_y: Tensor, inv_xx: Tensor, x: Tensor, ddof: int = 0) -> Tensor:
    """
    Homoskedastic standard errors of parameter estimate
    (Note that this estimator should not be used with weighting)

    Parameters
    ----------
    res_y: Tensor(m, n)
        Vector of dependent parameters residual of shape ``(m, n)``,
        where ``m`` is the number of dependent variables
        and ``n`` is the number of observations
    inv_xx: Tensor(d, d)
        Inverse of covariance matrix of regressors, where ``d`` is the
        number of regressors
    x: Tensor(d, n)
        Vector of regressors of shape ``(d, n)``, where ``d`` is the
        number of regressors and ``n`` is the number of observations
    ddof: int
        Degress of freedom that were captured by preprocessing

    Returns
    -------
    err_yx: Tensor (m, d, d)
        Estimate of parameter vector errors

    Examples
    --------
    """
    sigma = torch.einsum("li,li->l", res_y, res_y) / (res_y.shape[1] - ddof)
    return torch.tensordot(sigma, inv_xx, 0)


def hc0(res_y: Tensor, inv_xx: Tensor, x: Tensor, ddof: int = 0) -> Tensor:
    """
    Heteroskedasticity-consistent standard errors of parameter estimate (HC0)

    Parameters
    ----------
    res_y: Tensor(m, n)
        Vector of dependent parameters residual of shape ``(m, n)``,
        where ``m`` is the number of dependent variables
        and ``n`` is the number of observations
    inv_xx: Tensor(d, d)
        Inverse of covariance matrix of regressors, where ``d`` is the
        number of regressors
    x: Tensor(d, n)
        Vector of regressors of shape ``(d, n)``, where ``d`` is the
        number of regressors and ``n`` is the number of observations
    ddof: int
        Degress of freedom that were captured by preprocessing

    Returns
    -------
    err_yx: Tensor (m, d, d)
        Estimate of parameter vector errors

    Examples
    --------
    """
    sigma = torch.einsum("li,ji,ki->ljk", res_y.square_(), x, x)
    return inv_xx @ sigma @ inv_xx


def hc1(res_y: Tensor, inv_xx: Tensor, x: Tensor, ddof: int = 0) -> Tensor:
    """
    Heteroskedasticity-consistent standard errors of parameter estimate (HC1)

    Parameters
    ----------
    res_y: Tensor(m, n)
        Vector of dependent parameters residual of shape ``(m, n)``,
        where ``m`` is the number of dependent variables
        and ``n`` is the number of observations
    inv_xx: Tensor(d, d)
        Inverse of covariance matrix of regressors, where ``d`` is the
        number of regressors
    x: Tensor(d, n)
        Vector of regressors of shape ``(d, n)``, where ``d`` is the
        number of regressors and ``n`` is the number of observations
    ddof: int
        Degress of freedom that were captured by preprocessing

    Returns
    -------
    err_yx: Tensor (m, d, d)
        Estimate of parameter vector errors

    Examples
    --------
    """
    alpha = res_y.shape[1] / (res_y.shape[1] - ddof)
    sigma = torch.einsum("li,ji,ki->ljk", res_y.square_(), x, x)
    return (alpha * inv_xx) @ sigma @ inv_xx


def hc2(res_y: Tensor, inv_xx: Tensor, x: Tensor, ddof: int = 0) -> Tensor:
    """
    Heteroskedasticity-consistent standard errors of parameter estimate (HC2)

    Parameters
    ----------
    res_y: Tensor(m, n)
        Vector of dependent parameters residual of shape ``(m, n)``,
        where ``m`` is the number of dependent variables
        and ``n`` is the number of observations
    inv_xx: Tensor(d, d)
        Inverse of covariance matrix of regressors, where ``d`` is the
        number of regressors
    x: Tensor(d, n)
        Vector of regressors of shape ``(d, n)``, where ``d`` is the
        number of regressors and ``n`` is the number of observations
    ddof: int
        Degress of freedom that were captured by preprocessing

    Returns
    -------
    err_yx: Tensor (m, d, d)
        Estimate of parameter vector errors

    Examples
    --------
    """
    alpha = 1 / (1 - torch.einsum("ji,ji->i", torch.mm(inv_xx, x), x))
    sigma = torch.einsum("li,ji,ki->ljk", res_y.square_().mul_(alpha), x, x)
    return inv_xx @ sigma @ inv_xx


def hc3(res_y: Tensor, inv_xx: Tensor, x: Tensor, ddof: int = 0) -> Tensor:
    """
    Heteroskedasticity-consistent standard errors of parameter estimate (HC3)

    Parameters
    ----------
    res_y: Tensor(m, n)
        Vector of dependent parameters residual of shape ``(m, n)``,
        where ``m`` is the number of dependent variables
        and ``n`` is the number of observations
    inv_xx: Tensor(d, d)
        Inverse of covariance matrix of regressors, where ``d`` is the
        number of regressors
    x: Tensor(d, n)
        Vector of regressors of shape ``(d, n)``, where ``d`` is the
        number of regressors and ``n`` is the number of observations
    ddof: int
        Degress of freedom that were captured by preprocessing

    Returns
    -------
    err_yx: Tensor (m, d, d)
        Estimate of parameter vector errors

    Examples
    --------
    """
    alpha = 1 / (1 - torch.einsum("ji,ji->i", torch.mm(inv_xx, x), x))
    n = res_y.shape[1]
    gamma = torch.einsum("li,ji->lj", res_y.mul(alpha), x) * n**(-0.5)
    sigma = torch.einsum("li,ji,ki->ljk", res_y.square_().mul_(alpha), x, x)\
        - (gamma[:, None, :] * gamma[:, :, None])
    return ((n-1) / n * inv_xx) @ sigma @ inv_xx