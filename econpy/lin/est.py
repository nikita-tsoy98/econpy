import numpy as np
from econpy.lin import err


def ols(Y, X, get_err_fn=err.hc1, ddof=0):
    """
    Get OLS estimate of the model ``Y = prm @ X + Y_err``

    Parameters
    ----------
    Y : ndarray(m, n)
        Vector of dependent variables of shape ``(m, n)``, where ``m`` is the
        number of dependent variables and ``n`` is the number of observations
    X : ndarray(d, n)
        Vector of regressors of shape ``(d, n)``, where ``d`` is the
        number of regressors and ``n`` is the number of observations
    get_err_fn : function ((m, n), (d, d), (d, n), int) -> (m, d, d)
        Error function
    ddof : int
        Additional degrees of freedom that will be used to calculate errors

    Returns
    -------
    prm : ndarray(m, d)
        Estimate of parameter vector
    prm_err : ndarray(m, d, d)
        Estimate of parameter vector variance

    Examples
    --------
    >>> import econpy as ep
    >>> import numpy as np
    >>> n, d = 1001, 2
    >>> rng = np.random.default_rng(179)
    >>> X = rng.standard_normal((d, n))
    >>> b = np.array([[1., 2.]])
    >>> Y_err = rng.standard_normal((1, n))
    >>> Y = b @ X + Y_err
    >>> ep.lin.est.ols(Y, X, ep.lin.err.hc1)
    (array([[1.04029972, 2.00481945]]),
    array([[[9.01651519e-04, 7.89637346e-05],
            [7.89637346e-05, 8.96064163e-04]]]))

    """
    inv_cov = np.linalg.inv(np.inner(X, X))
    prm = np.inner(Y, X) @ inv_cov
    res = Y - prm @ X
    prm_err = get_err_fn(res, inv_cov, X, prm.size + ddof)
    return prm, prm_err


def wls(Y, X, w, get_err_fn=err.hc1, ddof=0):
    """
    Get WLS estimate of the model ``Y = prm @ X + Y_err``

    Parameters
    ----------
    Y : ndarray(m, n)
        Vector of dependent variables of shape ``(m, n)``, where ``m`` is the
        number of dependent variables and ``n`` is the number of observations
    X : ndarray(d, n)
        Vector of regressors of shape ``(d, n)``, where ``d`` is the
        number of regressors and ``n`` is the number of observations
    w : ndarray(n)
        Vector of weights of shape ``(n,)``, where ``n`` is the number of
        observations
    get_err_fn : function ((m, n), (d, d), (d, n), int) -> (m, d, d)
        Error function
    ddof : int
        Additional degrees of freedom that will be used to calculate errors

    Returns
    -------
    prm : ndarray(m, d)
        Estimate of parameter vector
    prm_err : ndarray(m, d, d)
        Estimate of parameter vector variance

    Examples
    --------
    >>> import econpy as ep
    >>> import numpy as np
    >>> n, d = 1001, 2
    >>> rng = np.random.default_rng(179)
    >>> X = rng.standard_normal((d, n))
    >>> b = np.array([[1., 2.]])
    >>> w = rng.uniform(0, 1, n)
    >>> Y_err = rng.standard_normal((1, n))
    >>> Y = b @ X + Y_err
    >>> ep.lin.est.wls(Y, X, w, ep.lin.err.hc1)
    (array([[1.01209159, 1.99392029]]),
    array([[[1.19092651e-03, 9.15351248e-05],
            [9.15351248e-05, 1.19738269e-03]]]))

    """
    sqrt_w = np.sqrt(w)
    return ols(sqrt_w * Y, sqrt_w * X, get_err_fn, ddof)


def tsls(Y, X, Z, get_err_fn=err.hc1, ddof=0):
    """
    Get TSLS estimate of the model ``Y = prm @ X + Y_err, X = prm_f @ Z + X_err``

    Parameters
    ----------
    Y : ndarray(m, n)
        Vector of dependent variables of shape ``(m, n)``, where ``m`` is the
        number of dependent variables and ``n`` is the number of observations
    X : ndarray(d, n)
        Vector of regressors of shape ``(d, n)``, where ``d`` is the
        number of regressors and ``n`` is the number of observations
    Z : ndarray(r, n)
        Vector of instruments of shape ``(r, n)``, where ``r`` is the
        number of instruments and ``n`` is the number of observations
    get_err_fn : function ((m, n), (d, d), (d, n), int) -> (m, d, d)
        Error function
    ddof : int
        Additional degrees of freedom that will be used to calculate errors

    Returns
    -------
    prm : ndarray(m, d)
        Estimate of parameter vector
    prm_err : ndarray(m, d, d)
        Estimate of parameter vector variance
    prm_f : ndarray(d, r)
        Estimate of parameter vector for the first stage
    prm_f_err : ndarray(d, r, r)
        Estimate of parameter vector variance for the first stage

    Examples
    --------
    >>> import econpy as ep
    >>> import numpy as np
    >>> n, r, d = 1001, 3, 2
    >>> rng = np.random.default_rng(179)
    >>> Z = rng.standard_normal((r, n))
    >>> g = np.array([[1., 2., 0.],
                      [0., 1., 1.]])
    >>> b = np.array([[1., 2.]])
    >>> X_err = rng.standard_normal((d, n))
    >>> X = g @ Z + X_err
    >>> Y_err = rng.standard_normal((1, n))
    >>> Y = (b @ g) @ Z + Y_err
    >>> ep.lin.est.tsls(Y, X, Z, ep.lin.err.hc1)
    (array([[1.02661761, 1.99233687]]),
    array([[[ 0.00160102, -0.00174609],
            [-0.00174609,  0.00496165]]]),
    array([[9.84476024e-01, 2.01435404e+00, 1.79104118e-02],
            [4.69398516e-04, 9.72084169e-01, 9.65296423e-01]]),
    array([[[ 9.41789054e-04,  5.53232016e-05,  6.52132465e-05],
            [ 5.53232016e-05,  8.67519282e-04, -9.57919621e-06],
            [ 6.52132465e-05, -9.57919621e-06,  9.83312340e-04]],
    
            [[ 8.88927449e-04, -4.97567021e-05,  7.62662634e-05],
            [-4.97567021e-05,  9.44267008e-04,  7.22023295e-05],
            [ 7.62662634e-05,  7.22023295e-05,  9.96610029e-04]]]))

    """
    # first stage
    inv_cov_f = np.linalg.inv(np.inner(Z, Z))
    cov_XZ = np.inner(X, Z)
    prm_f = cov_XZ @ inv_cov_f
    X_hat = prm_f @ Z
    prm_f_err = get_err_fn(X-X_hat, inv_cov_f, Z, prm_f.size+ddof)

    # second stage
    inv_cov = np.linalg.inv(np.inner(prm_f, cov_XZ))
    prm = np.inner(Y, X_hat) @ inv_cov
    prm_err = get_err_fn(Y-prm@X, inv_cov, X_hat, prm.size+prm_f.size+ddof)
    return prm, prm_err, prm_f, prm_f_err


def wtsls(Y, X, Z, w, get_err_fn=err.hc1, ddof=0):
    """
    Get weighted TSLS estimate of the model ``Y = prm @ X + Y_err,
    X = prm_f @ Z + X_err``

    Parameters
    ----------
    Y : ndarray(m, n)
        Vector of dependent variables of shape ``(m, n)``, where ``m`` is the
        number of dependent variables and ``n`` is the number of observations
    X : ndarray(d, n)
        Vector of regressors of shape ``(d, n)``, where ``d`` is the
        number of regressors and ``n`` is the number of observations
    Z : ndarray(r, n)
        Vector of instruments of shape ``(r, n)``, where ``r`` is the
        number of instruments and ``n`` is the number of observations
    w : ndarray(n)
        Vector of weights of shape ``(n,)``, where ``n`` is the number of
        observations
    get_err_fn : function ((m, n), (d, d), (d, n), int) -> (m, d, d)
        Error function
    ddof : int
        Additional degrees of freedom that will be used to calculate errors

    Returns
    -------
    prm : ndarray(m, d)
        Estimate of parameter vector
    prm_err : ndarray(m, d, d)
        Estimate of parameter vector variance
    prm_f : ndarray(d, r)
        Estimate of parameter vector for the first stage
    prm_f_err : ndarray(d, r, r)
        Estimate of parameter vector variance for the first stage

    Examples
    --------
    >>> import econpy as ep
    >>> import numpy as np
    >>> n, r, d = 1001, 3, 2
    >>> rng = np.random.default_rng(179)
    >>> Z = rng.standard_normal((r, n))
    >>> g = np.array([[1., 2., 0.],
                      [0., 1., 1.]])
    >>> b = np.array([[1., 2.]])
    >>> w = rng.uniform(0, 1, n)
    >>> X_err = rng.standard_normal((d, n))
    >>> X = g @ Z + X_err
    >>> Y_err = rng.standard_normal((1, n))
    >>> Y = (b @ g) @ Z + Y_err
    >>> ep.lin.est.wtsls(Y, X, Z, w, ep.lin.err.hc1)
    (array([[1.05791051, 1.94524836]]),
    array([[[ 0.00207487, -0.00232696],
            [-0.00232696,  0.00672986]]]),
    array([[0.95792117, 2.01579459, 0.0452929 ],
            [0.00909182, 0.97323327, 0.97836013]]),
    array([[[ 1.28319702e-03,  1.22815799e-04,  1.79026520e-04],
            [ 1.22815799e-04,  1.22042438e-03,  1.91599383e-05],
            [ 1.79026520e-04,  1.91599383e-05,  1.34582689e-03]],
    
            [[ 1.11670761e-03, -3.98396203e-05,  1.14882954e-04],
            [-3.98396203e-05,  1.23630416e-03,  1.70753579e-04],
            [ 1.14882954e-04,  1.70753579e-04,  1.38360794e-03]]]))

    """
    sqrt_w = np.sqrt(w)
    return tsls(sqrt_w * Y, sqrt_w * X, sqrt_w * Z, get_err_fn, ddof)