import numpy as np
import pickle
from scipy.stats import pearsonr, weightedtau, kendalltau


def process_transrate_logs(path, eps=1e-4, reg=1., opt_active_eigs='all', opt_res='weight', normalize_eigz=False, normalize_opt='rank'):
    """
    :param path: path of the file
    :param eps: value of eps^2 in transrate
    :param reg: >= 1
    :param opt_active_eigs: option to define the indeces of active eigenvalues and the residuals
        'rank': matrix rank
        'all': use all eigenvalues
    :param opt_res: option to handle the residuals: first sum the residuals, then add it the all active eigenvalues
        'avg': each active eigenvalues add the same values
        'weight': each active eigenvalues add the weighted values
        others: drop the residuals
    :param normalize_eigz: option to normalize the sum of eigenvalues to 1
    :param normalize_opt: option to normalize the the value of transrate
    """

    file = open(path, 'rb')
    data = pickle.load(file)
    file.close()

    nc = data['n_Zc']
    n = np.sum(nc)

    eig_z = data['eig_Z']
    rank_z = data['rank_Z']
    eig_zc = data['eig_Zc']
    rank_zc = data['rank_Zc']

    if normalize_eigz:
        sum_eig_z = np.sum(eig_z)
        eig_z = eig_z / sum_eig_z
        eig_zc = eig_zc / sum_eig_z

    d = eig_z.shape[0]

    if opt_active_eigs == 'rank':
        idx = rank_z
    elif opt_active_eigs == 'all':
        idx = d

    if opt_res == 'avg':
        eig_z[:idx] = eig_z[:idx] + np.sum(eig_z[idx:]) / idx
    elif opt_res == 'weight':
        eig_z[:idx] = eig_z[:idx] + np.sum(eig_z[idx:]) * eig_z[:idx] / np.sum(eig_z[:idx])
    else:
        raise ValueError("Not supported opt res")

    rz = np.sum(np.log(reg + 1/eps * np.abs(eig_z[:idx])))


    nClass = len(rank_zc)

    rzc = np.zeros(nClass)

    for i in range(nClass):
        eig_zc_i = eig_zc[i]

        if opt_res == 'avg':
            eig_zc_i[:idx] = eig_zc_i[:idx] + np.sum(eig_zc_i[idx:]) / idx
        elif opt_res == 'weight':
            eig_zc_i[:idx] = eig_zc_i[:idx] + np.sum(eig_zc_i[idx:]) * eig_zc_i[:idx] / np.sum(eig_zc_i[:idx])

        rzc[i] = np.sum(np.log(reg + 1/eps * np.abs(eig_zc_i[:idx])))

    if normalize_opt == 'dim':
        normal_r = min(d, idx)
        rz = rz / normal_r
        rzc = rzc / normal_r
    elif normalize_opt == 'rank':
        normal_r = min(rank_z, idx)
        rz = rz / normal_r
        rzc = rzc / normal_r

    return rz/2., np.sum(rzc * np.array(nc)/n)/2.


def analyze_correlation(acc, score):
    corr, conf = pearsonr(score, acc)
    corr_ktau, conf_ktau = kendalltau(score, acc)
    corr_wtau, conf_wtau = weightedtau(score, acc)

    return np.round(corr, 4), np.round(corr_ktau, 4), np.round(corr_wtau, 4)





