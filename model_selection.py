import numpy as np

from extract_result_utils import process_transrate_logs, analyze_correlation


def extract_transrate(id_list, eps, use_Z_proj=True, normalize_eigz=False):
    trans_log_path = './logs_trans/'

    len_res = len(id_list)
    rz = np.zeros(len_res)
    rzc = np.zeros(len_res)

    for idx, config_id in enumerate(id_list):
        id_trans_path = trans_log_path + config_id

        if use_Z_proj:
            z_path = '/Z_new_proj_mul.pkl'
        else:
            z_path = '/Z_centralized.pkl'


        rz[idx], rzc[idx] = process_transrate_logs(id_trans_path + z_path, eps=eps, normalize_eigz=normalize_eigz)

    transrate = rz-rzc

    return transrate


def model_selection(case):
    print('Model Selection. Source: ImageNet')
    if case == 1:
        print('Target: CIFAR100')
        id_list = ['1001', '1601', '1701', '1702', '1703', '1704', '1705']

        transrate = extract_transrate(id_list, eps=1e-5, use_Z_proj=True, normalize_eigz=True)
        # transrate = extract_transrate(id_list, eps=1e-4, use_Z_proj=True, normalize_eigz=True)

        acc = np.array([0.8158, 0.8391, 0.8405, 0.6900, 0.7512, 0.7865, 0.8083])

        nce = np.array([-2.7121, -2.515, -2.5163, -3.0111, -2.791, -2.7456, -2.6407])
        leep = np.array([-2.881, -2.6225, -2.612, -3.2982, -2.9966, -2.8792, -2.7594])
        lfc = np.array([0.2061, 0.2293, 0.247, 0.2285, 0.267, 0.2632, 0.3531])
        h_score = np.array([24.047, 27.5055, 40.2022, 26.2639, 29.8263, 30.4649, 31.0708])
        logme = np.array([1.0126, 1.0356, 1.0989, 1.0093, 1.0344, 1.0387, 1.0444])

    elif case == 2:
        print('Target: Caltech-101')
        id_list = ['A1301', 'A1401', 'A1402', 'A1403', 'A1404', 'A1405', 'A1406']

        # transrate = extract_transrate(id_list, eps=1e-4, use_Z_proj=True, normalize_eigz=True)
        transrate = extract_transrate(id_list, eps=1e-3, use_Z_proj=True, normalize_eigz=True)

        acc = np.array([0.9386, 0.9486, 0.9584, 0.8423, 0.9106, 0.9267, 0.9443])
        nce = np.array([-1.2483, -1.0444, -1.0923, -2.1722, -1.6571, -1.3565, -1.3025])
        leep = np.array([-1.8678, -1.5322, -1.5751, -2.9693, -2.3094, -1.948, -1.9013])
        lfc = np.array([0.2941, 0.3585, 0.3142, 0.2543, 0.3183, 0.3296, 0.464])
        h_score = np.array([53.4583, 57.7621, 89.2963, 67.8296, 71.9401, 73.15, 72.9655])
        logme = np.array([1.0867, 1.1263, 1.195, 1.0506, 1.1012, 1.1127, 1.1238])

    elif case == 3:
        print('Target: Caltech-256')
        id_list = ['A1311', 'A1411', 'A1412', 'A1413', 'A1414', 'A1415', 'A1416']

        # transrate = extract_transrate(id_list, eps=1e-4, use_Z_proj=True, normalize_eigz=True)
        transrate = extract_transrate(id_list, eps=1e-3, use_Z_proj=True, normalize_eigz=True)

        acc = np.array([0.7844, 0.8021, 0.8239, 0.5657, 0.6983, 0.7516, 0.793])
        nce = np.array([-2.1283, -1.8915, -1.9081, -3.412, -2.7283, -2.363, -2.2325])
        leep = np.array([-2.6353, -2.2806, -2.3084, -4.0048, -3.2229, -2.7896, -2.6729])
        lfc = np.array([0.277, 0.3366, 0.2969, 0.2329, 0.2944, 0.3159, 0.4617])
        h_score = np.array([57.0536, 66.3607, 120.8305, 63.9218, 76.2908, 81.1525, 83.8847])
        logme = np.array([1.4387, 1.4602, 1.5289, 1.4183, 1.4434, 1.4536, 1.464])


    elif case == 4:
        print('Target: SUN397')
        id_list = ['A1341', 'A1441', 'A1442', 'A1443', 'A1444', 'A1445', 'A1446']

        # transrate = extract_transrate(id_list, eps=1e-3, use_Z_proj=True, normalize_eigz=True)
        transrate = extract_transrate(id_list, eps=5e-3, use_Z_proj=True, normalize_eigz=True)

        acc = np.array([0.5297, 0.5448, 0.5821, 0.4043, 0.4670, 0.5099, 0.5525])
        nce = np.array([-3.5545, -3.3159, -3.3684, -4.4518, -4.0388, -3.7164, -3.6056])
        leep = np.array([-4.1713, -3.8587, -3.9156, -4.9365, -4.5200, -4.2045, -4.1001])
        lfc = np.array([0.2216, 0.2768, 0.2379, 0.1874, 0.2291, 0.2583, 0.3617])
        h_score = np.array([38.8592, 43.5620, 95.4907, 56.4381, 62.8448, 65.7015, 66.6171])
        logme = np.array([1.5984, 1.6034, 1.6187, 1.5954, 1.6016, 1.6048, 1.6079])

    pr = np.zeros(6)
    kt = np.zeros(6)
    wt = np.zeros(6)

    pr[0], kt[0], wt[0] = analyze_correlation(acc, nce)
    pr[1], kt[1], wt[1] = analyze_correlation(acc, leep)
    pr[2], kt[2], wt[2] = analyze_correlation(acc, lfc)
    pr[3], kt[3], wt[3] = analyze_correlation(acc, h_score)
    pr[4], kt[4], wt[4] = analyze_correlation(acc, logme)
    pr[5], kt[5], wt[5] = analyze_correlation(acc, transrate)

    np.set_printoptions(edgeitems=30, linewidth=100000)

    print('Measure  :    NCE,   LEEP,    LFC,   h-Sc,  LogMe,    TrR')
    print('Pearson R: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(pr[0], pr[1], pr[2], pr[3], pr[4], pr[5]))
    print('K Tau    : {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(kt[0], kt[1], kt[2], kt[3], kt[4], kt[5]))
    print('W Tau    : {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(wt[0], wt[1], wt[2], wt[3], wt[4], wt[5]))


if __name__ == '__main__':
    ### options: 1=CIFAR100, 2=Cal101, 3=Cal256, 4=SUN397
    model_selection(1)
