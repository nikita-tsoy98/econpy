import numpy as np

def absorb(Ys, X):
    inv_cov = np.linalg.inv(np.inner(X, X))
    return tuple(Y - (np.inner(Y, X) @ inv_cov) @ X for Y in Ys)

def absorb_fes(Ys, int_fes, real_fes, real_fe_weights, eps=1e-4):
    def absorb_one_int_fe(Ys, fe_sum, fe_inv_ind, fe_counts):
        for Y in Ys:
            for i in range(Y.shape[0]):
                fe_sum[:] = 0
                np.add.at(fe_sum, fe_inv_ind, Y[i])
                fe_sum /= fe_counts
                Y[i] -= fe_sum[fe_inv_ind]

    def absorb_one_real_fe(Ys, fe_sum, fe_inv_ind, fe_cov, fe_weight):
        for Y in Ys:
            for i in range(Y.shape[0]):
                fe_sum[:] = 0
                np.add.at(fe_sum, fe_inv_ind, fe_weight * Y[i])
                fe_sum /= fe_cov
                Y[i] -= fe_sum[fe_inv_ind] * fe_weight

    def absorb_iter(
            Ys, Zs, int_fe_sums, int_fe_inv_inds, int_fe_counts, real_fe_sums,
            real_fe_inv_inds, real_fe_covs, real_fe_weights):
        eps = 0
        for fe_sum, fe_inv_ind, fe_counts in zip(
            int_fe_sums, int_fe_inv_inds, int_fe_counts):
            absorb_one_int_fe(Ys, fe_sum, fe_inv_ind, fe_counts)
        for fe_sum, fe_inv_ind, fe_cov, fe_weight in zip(
            real_fe_sums, real_fe_inv_inds, real_fe_covs, real_fe_weights):
            absorb_one_real_fe(Ys, fe_sum, fe_inv_ind, fe_cov, fe_weight)
        for Z, Y in zip(Zs, Ys):
            eps = np.maximum(eps, np.sqrt(((Z-Y)**2 / Z.size).sum()))
            np.copyto(Z, Y)
        return eps

    int_fe_inv_inds, int_fe_counts = zip(*(
        np.unique(fe, return_inverse=True, return_counts=True)[1:]\
            for fe in int_fes))
    int_fe_sums = tuple(
        np.empty(len(fe_count)) for fe_count in int_fe_counts)
    real_fe_inv_inds = zip(*(
        np.unique(fe, return_inverse=True)[1] for fe in real_fes))
    real_fe_covs = tuple(
        np.empty(fe_inv_ind.max()) for fe_inv_ind in real_fe_inv_inds)
    real_fe_sums = tuple(
        np.empty(len(fe_cov)) for fe_cov in real_fe_covs)
    for fe_inv_ind, fe_weight, fe_cov in zip(
        real_fe_inv_inds, real_fe_weights, real_fe_covs):
        np.add.at(fe_cov, fe_inv_ind, fe_weight**2)

    Zs = tuple(np.empty_like(Y) for Y in Ys)
    for fe_sum, fe_inv_ind, fe_counts in zip(
        int_fe_sums, int_fe_inv_inds, int_fe_counts):
        absorb_one_int_fe(Ys, fe_sum, fe_inv_ind, fe_counts)
        for Z, Y in zip(Zs, Ys):
            np.copyto(Z, Y)
        absorb_one_int_fe(Ys, fe_sum, fe_inv_ind, fe_counts)
        for Z, Y in zip(Zs, Ys):
            eps = np.maximum(eps, np.sqrt(((Z-Y)**2 / Z.size).sum()))
            np.copyto(Z, Y)
    for fe_sum, fe_inv_ind, fe_cov, fe_weight in zip(
        real_fe_sums, real_fe_inv_inds, real_fe_covs, real_fe_weights):
        absorb_one_real_fe(Ys, fe_sum, fe_inv_ind, fe_cov, fe_weight)
        for Z, Y in zip(Zs, Ys):
            np.copyto(Z, Y)
        absorb_one_real_fe(Ys, fe_sum, fe_inv_ind, fe_cov, fe_weight)
        for Z, Y in zip(Zs, Ys):
            eps = np.maximum(eps, np.sqrt(((Z-Y)**2 / Z.size).sum()))
            np.copyto(Z, Y)
    iter_eps = 2 * eps
    iters = 2
    while iter_eps > eps:
        iter_eps = absorb_iter(
            Ys, Zs, int_fe_sums, int_fe_inv_inds, int_fe_counts, real_fe_sums,
            real_fe_inv_inds, real_fe_covs, real_fe_weights)
        iters += 1
    print(iters, eps)
    return Ys