import numpy as np
import pickle


def eigens_and_rank(ZZ):
    _, eigs, _ = np.linalg.svd(ZZ, full_matrices=False)
    r = np.linalg.matrix_rank(ZZ)

    return eigs, r


def pre_transrate(f, y, normalize=True):
    if normalize:
        l2 = np.atleast_1d(np.linalg.norm(f, 2, -1))
        Z = f / np.expand_dims(l2, -1)
    else:
        Z = f

    K = int(y.max() + 1)

    n, d = f.shape

    eig_c = []
    nc = []
    rank_c = []

    for i in range(K):
        y_ = (y == i).flatten()
        Zi = Z[y_]

        ZZi = Zi.transpose() @ Zi
        if i == 0:
            ZZ = ZZi
        else:
            ZZ = ZZ + ZZi

        nc.append(Zi.shape[0])

        eigs_i, ri = eigens_and_rank(ZZi / float(Zi.shape[0]))

        eig_c.extend(np.expand_dims(eigs_i, axis=0))
        rank_c.append(ri)

    ZZ = ZZ / float(n)
    eig_Z, rank_Z = eigens_and_rank(ZZ)

    return eig_Z, rank_Z, np.stack(eig_c), rank_c, nc


def transrate_eig(z_centralized, y, eig_file_path):
    eig_Z, rank_Z, eig_Zc, rank_Zc, n_Zc = pre_transrate(z_centralized, np.copy(y))

    data = {
        'eig_Z': eig_Z,
        'rank_Z': rank_Z,
        'eig_Zc': eig_Zc,
        'rank_Zc': rank_Zc,
        'n_Zc': n_Zc
    }

    file = open(eig_file_path + '/Z_centralized.pkl', 'wb')
    pickle.dump(data, file)
    file.close()


def pre_transrate_low_dim_proj(f, y):
    n, d = f.shape
    Z = f

    mean_f = np.mean(f, axis=0)
    mean_f = np.expand_dims(mean_f, 1)

    covf = f.T @ f / n - mean_f * mean_f.T

    K = int(y.max() + 1)

    eig_c = []
    nc = []
    rank_c = []
    cov_Zi = []

    for i in range(K):
        y_ = (y == i).flatten()
        Zi = Z[y_]
        nci = Zi.shape[0]
        nc.append(nci)

        mean_Zi = np.mean(Zi, axis=0)
        mean_Zi = np.expand_dims(mean_Zi, 1)
        g = mean_Zi - mean_f

        if i == 0:
            covg = g @ g.T * nci
        else:
            covg = g @ g.T * nci + covg

        ZZi = Zi.T @ Zi - (mean_Zi * mean_Zi.T) * nci

        cov_Zi.append(ZZi)

    proj_mat = np.dot(np.linalg.pinv(covf, rcond=1e-15), covg/n)


    for i in range(K):
        eigs_i, ri = eigens_and_rank(cov_Zi[i] @ proj_mat / nc[i])
        eig_c.extend(np.expand_dims(eigs_i, axis=0))
        rank_c.append(ri)
    eig_Z, rank_Z = eigens_and_rank(covf @ proj_mat)

    return eig_Z, rank_Z, np.stack(eig_c), rank_c, nc


def transrate_eig_proj(z, y, eig_file_path):
    eig_Z, rank_Z, eig_Zc, rank_Zc, n_Zc = pre_transrate_low_dim_proj(np.copy(z), np.copy(y), opt='mul', opt_new=False)
    data = {
        'eig_Z': eig_Z,
        'rank_Z': rank_Z,
        'eig_Zc': eig_Zc,
        'rank_Zc': rank_Zc,
        'n_Zc': n_Zc
    }

    file = open(eig_file_path + '/Z_new_proj_mul.pkl', 'wb')
    pickle.dump(data, file)
    file.close()


