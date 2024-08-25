from collections.abc import Callable
from typing import Optional
from torch import Tensor

import torch
from econpy.lin import err


def no_reg(cov_xx: Tensor) -> Tensor:
    """
    Dummy regularization of covariance matrix
    
    Parameters
    ----------
    cov_xx: Tensor
        Covarinace matrix for regularization
    
    Returns
    -------
    cov_xx: Tensor
        Covariance matrix after regularization
    """
    return cov_xx


def ridge_reg(cov_xx: Tensor, a: float = 2**(-7)) -> Tensor:
    """
    Ridge-like regularization of covariance matrix

    Parameters
    ----------
    cov_xx: Tensor
        Covarinace matrix for regularization
    a: float
        Strength of regularization
    
    Returns
    -------
    cov_xx: Tensor
        Covariance matrix after regularization
    """
    diag_x = cov_xx.diagonal()
    with torch.no_grad():
        diag_x += a * torch.linalg.matrix_norm(cov_xx, 2)
        cov_xx.div_(1 + a)
    return cov_xx


def ols(
        y: Tensor, x: Tensor,
        reg_fn: Callable[[Tensor], Tensor] = no_reg
        ) -> tuple[Tensor, Tensor, Tensor]:
    """
    Get OLS estimate of the model ``y = prm_yx @ x + res_y``

    Parameters
    ----------
    y: Tensor(m, n)
        Vector of dependent variables of shape ``(m, n)``, where ``m`` is the
        number of dependent variables and ``n`` is the number of observations
    x: Tensor(d, n)
        Vector of regressors of shape ``(d, n)``, where ``d`` is the
        number of regressors and ``n`` is the number of observations
    reg_fn: Callable[[Tensor], Tensor]
        Function for covarinace matrix regularization

    Returns
    -------
    prm_yx: Tensor(m, d)
        Estimate of parameter vector
    y_hat: Tensor(m, n)
        Estimate of dependent variables
    inv_xx: Tensor(d, d)
        Inverse of covariance matrix of regressors

    Examples
    --------
    """
    inv_xx = torch.linalg.inv(reg_fn(torch.mm(x, x.T)))
    prm_yx = torch.mm(torch.mm(y, x.T), inv_xx)
    y_hat = torch.mm(prm_yx, x)
    return prm_yx, y_hat, inv_xx


def tsls(
        y: Tensor, x: Tensor, z: Tensor,
        reg_fn: Callable[[Tensor, Callable], Tensor] = no_reg
        ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Get TSLS estimate of the model
    ``y = prm_yx @ x + res_y, x = prm_xz @ z + res_x``

    Parameters
    ----------
    y: Tensor(m, n)
        Vector of dependent variables of shape ``(m, n)``, where ``m`` is the
        number of dependent variables and ``n`` is the number of observations
    x: Tensor(d, n)
        Vector of regressors of shape ``(d, n)``, where ``d`` is the
        number of regressors and ``n`` is the number of observations
    z: Tensor(r, n)
        Vector of instruments of shape ``(r, n)``, where ``r`` is the
        number of instruments and ``n`` is the number of observations
    reg_fn: Callable[[Tensor], Tensor]
        Function for covarinace matrix regularization

    Returns
    -------
    prm_yx: Tensor(m, d)
        Estimate of parameter vector
    prm_xz: Tensor(d, r)
        Estimate of parameter vector for the first stage
    y_hat: Tensor(m, n)
        Estimate of dependent variables
    x_hat: Tensor(d, n)
        Estimate of endogenous variables
    inv_xx: Tensor(d, d)
        Inverse of covariance matrix of regressors
    inv_zz: Tensor(r, r)
        Inverse of covariance matrix of instruments

    Examples
    --------
    """
    # first stage
    inv_zz = torch.linalg.inv(reg_fn(torch.mm(z, z.T)))
    cov_xz = torch.mm(x, z.T)
    prm_xz = torch.mm(cov_xz, inv_zz)
    x_hat = torch.mm(prm_xz, z)

    # second stage
    inv_xx = torch.linalg.inv(reg_fn(torch.mm(prm_xz, cov_xz.T)))
    prm_yx = torch.mm(torch.mm(y, x_hat.T), inv_xx)
    y_hat = torch.mm(prm_yx, x)
    return prm_yx, prm_xz, y_hat, x_hat, inv_xx, inv_zz


def ols_only_prm(
        y: Tensor, x: Tensor,
        reg_fn: Callable[[Tensor], Tensor] = no_reg) -> Tensor:
    """
    Get OLS estimate of the model ``y = prm_yx @ x + res_y``

    Parameters
    ----------
    y: Tensor(m, n)
        Vector of dependent variables of shape ``(m, n)``, where ``m`` is the
        number of dependent variables and ``n`` is the number of observations
    x: Tensor(d, n)
        Vector of regressors of shape ``(d, n)``, where ``d`` is the
        number of regressors and ``n`` is the number of observations
    reg_fn: Callable[[Tensor], Tensor]
        Function for covarinace matrix regularization

    Returns
    -------
    prm_yx: Tensor(m, d)
        Estimate of parameter vector

    Examples
    --------
    """
    return torch.linalg.solve(
        reg_fn(torch.mm(x, x.T)), torch.mm(y, x.T), left=False)


def tsls_only_prm(
        y: Tensor, x: Tensor, z: Tensor,
        reg_fn: Callable[[Tensor, Callable], Tensor] = no_reg
        ) -> tuple[Tensor, Tensor]:
    """
    Get TSLS estimate of the model
    ``y = prm_yx @ x + res_y, x = prm_xz @ z + res_x``

    Parameters
    ----------
    y: Tensor(m, n)
        Vector of dependent variables of shape ``(m, n)``, where ``m`` is the
        number of dependent variables and ``n`` is the number of observations
    x: Tensor(d, n)
        Vector of regressors of shape ``(d, n)``, where ``d`` is the
        number of regressors and ``n`` is the number of observations
    z: Tensor(r, n)
        Vector of instruments of shape ``(r, n)``, where ``r`` is the
        number of instruments and ``n`` is the number of observations
    reg_fn: Callable[[Tensor], Tensor]
        Function for covarinace matrix regularization

    Returns
    -------
    prm_yx: Tensor(m, d)
        Estimate of parameter vector
    prm_xz: Tensor(d, r)
        Estimate of parameter vector for the first stage

    Examples
    --------
    """
    # first stage
    cov_xz = torch.mm(x, z.T)
    prm_xz = torch.linalg.solve(reg_fn(torch.mm(z, z.T)), cov_xz, left=False)
    x_hat = torch.mm(prm_xz, z)

    # second stage
    prm_yx = torch.linalg.solve(
        reg_fn(torch.mm(prm_xz, cov_xz.T)), torch.mm(y, x_hat.T), left=False)
    return prm_yx, prm_xz


def ols_with_err(
        y: Tensor, x: Tensor, w: Optional[Tensor] = None, ddof: int = 0,
        reg_fn: Callable[[Tensor], Tensor] = no_reg,
        err_fn: Callable[[Tensor, Tensor, Tensor, int], Tensor] = err.hc0
        ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Get OLS estimate of the model ``y = prm_yx @ x + res_y``

    Parameters
    ----------
    y: Tensor(m, n)
        Vector of dependent variables of shape ``(m, n)``, where ``m`` is the
        number of dependent variables and ``n`` is the number of observations
    x: Tensor(d, n)
        Vector of regressors of shape ``(d, n)``, where ``d`` is the
        number of regressors and ``n`` is the number of observations
    w: Optional[Tensor(n)]
        Optional weighting vector
    reg_fn: Callable[[Tensor], Tensor]
        Function for covarinace matrix regularization
    err_fn: Callable[[Tensor, Tensor, Tensor, int], Tensor]
        Function for parameters error estimation

    Returns
    -------
    prm_yx: Tensor(m, d)
        Estimate of parameter vector
    y_hat: Tensor(m, n)
        Estimate of dependent variables
    inv_xx: Tensor(d, d)
        Inverse of covariance matrix of regressors
    err_yx: Tensor(m, d, d)
        Estimate of parameter vector errors

    Examples
    --------
    """
    if w is not None:
        w = w.sqrt()
        y = y.mul(w)
        x = x.mul(w)
    prm_yx, y_hat, inv_xx = ols(y, x, reg_fn)
    err_yx = err_fn(y.sub(y_hat), inv_xx, x, ddof)
    if w is not None:
        y_hat.div_(w)
    return prm_yx, y_hat, inv_xx, err_yx


def tsls_with_err(
        y: Tensor, x: Tensor, z: Tensor, w: Optional[Tensor] = None,
        ddof: int = 0,
        reg_fn: Callable[[Tensor, Callable], Tensor] = no_reg,
        err_fn: Callable[[Tensor, Tensor, Tensor, int], Tensor] = err.hc0
        ) -> tuple[
            Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Get TSLS estimate of the model
    ``y = prm_yx @ x + res_y, x = prm_xz @ z + res_x``

    Parameters
    ----------
    y: Tensor(m, n)
        Vector of dependent variables of shape ``(m, n)``, where ``m`` is the
        number of dependent variables and ``n`` is the number of observations
    x: Tensor(d, n)
        Vector of regressors of shape ``(d, n)``, where ``d`` is the
        number of regressors and ``n`` is the number of observations
    z: Tensor(r, n)
        Vector of instruments of shape ``(r, n)``, where ``r`` is the
        number of instruments and ``n`` is the number of observations
    w: Optional[Tensor(n)]
        Optional weighting vector
    reg_fn: Callable[[Tensor], Tensor]
        Function for covarinace matrix regularization
    err_fn: Callable[[Tensor, Tensor, Tensor, int], Tensor]
        Function for parameters error estimation

    Returns
    -------
    prm_yx: Tensor(m, d)
        Estimate of parameter vector
    prm_xz: Tensor(d, r)
        Estimate of parameter vector for the first stage
    y_hat: Tensor(m, n)
        Estimate of dependent variables
    x_hat: Tensor(d, n)
        Estimate of endogenous variables
    inv_xx: Tensor(d, d)
        Inverse of covariance matrix of regressors
    inv_zz: Tensor(r, r)
        Inverse of covariance matrix of instruments
    err_yx: Tensor(m, d, d)
        Estimate of parameter vector errors
    err_xz: Tensor(d, r, r)
        Estimate of parameter vector errors for the first stage

    Examples
    --------
    """
    if w is not None:
        w = w.sqrt()
        y = y.mul(w)
        x = x.mul(w)
        z = z.mul(w)
    prm_yx, prm_xz, y_hat, x_hat, inv_xx, inv_zz = tsls(y, x, z, reg_fn)
    err_yx = err_fn(y.sub(y_hat), inv_xx, x_hat, ddof + x.shape[0])
    err_xz = err_fn(x.sub(x_hat), inv_zz, z, ddof + z.shape[0])
    if w is not None:
        y_hat.div_(w)
        x_hat.div_(w)
    return prm_yx, prm_xz, y_hat, x_hat, inv_xx, inv_zz, err_yx, err_xz