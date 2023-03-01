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


def model_selection(case=0):
    print('Model Selection with more networks. Source: ImageNet')
    if case == 1:
        print('Target: CIFAR100')
        id_list = ['1001', '1601', '1701', '1702', '1703', '1704', '1705', 'A1801', 'A1802', 'A1803', 'A1804',
                   'A1805', 'A1806']

        acc = np.array([0.8158, 0.8391, 0.8405, 0.6900, 0.7512, 0.7865, 0.8083, 0.8269, 0.8418, 0.8435, 0.8375,
                        0.6700, 0.7811])
        nce = np.array([-2.7121, -2.5150, -2.5163, -3.0111, -2.791, -2.7456, -2.6407, -2.5036, -2.4182, -2.3052,
                        -2.6147, -2.7423, -2.6433])
        leep = np.array([-2.881, -2.6225, -2.6120, -3.2982, -2.9966, -2.8792, -2.7594, -2.6652, -2.4936, -2.4699,
                         -2.9954, -3.2081, -2.7322])
        lfc = np.array([0.2061, 0.2293, 0.2470, 0.2285, 0.2670, 0.2632, 0.3531, 0.2412, 0.3056, 0.3423, 0.2527,
                        0.3748, 0.2615])
        h_score = np.array([24.047, 27.5055, 40.2022, 26.2639, 29.8263, 30.4649, 31.0708, 32.1335, 37.5069, 39.7525,
                            37.8277, 30.2821, 32.0453])
        logme = np.array([1.0126, 1.0356, 1.0989, 1.0093, 1.0344, 1.0387, 1.0444, 1.0292, 1.0513, 1.0614, 1.0807,
                          1.0371, 1.051])

        transrate = extract_transrate(id_list, eps=1.5e-2, use_Z_proj=True, normalize_eigz=False)
        # transrate = extract_transrate(id_list, eps=1e-2, use_Z_proj=True, normalize_eigz=False)
        # transrate = extract_transrate(id_list, eps=1e-3, use_Z_proj=True, normalize_eigz=True)

    elif case == 2:
        print('Target: Caltech-101')
        id_list = ['A1301', 'A1401', 'A1402', 'A1403', 'A1404', 'A1405', 'A1406', 'A1701', 'A1702', 'A1703', 'A1704',
                   'A1705', 'A1706']

        transrate = extract_transrate(id_list, eps=2e-2, use_Z_proj=True, normalize_eigz=True)
        # transrate = extract_transrate(id_list, eps=1e-2, use_Z_proj=True, normalize_eigz=True)

        acc = np.array([0.9386, 0.9486, 0.9584, 0.8423, 0.9106, 0.9267, 0.9443, 0.9484, 0.9544, 0.9566, 0.9585,
                        0.8087, 0.9073])
        nce = np.array([-1.2483, -1.0444, -1.0923, -2.1722, -1.6571, -1.3565, -1.3025, -0.7275, -0.6425, -0.7006,
                        -0.6772, -1.6431, -1.233])
        leep = np.array([-1.8678, -1.5322, -1.5751, -2.9693, -2.3094, -1.948, -1.9013, -1.193, -0.9917, -1.0779,
                         -1.3238, -2.5778, -1.6993])
        lfc = np.array([0.2941, 0.3585, 0.3142, 0.2543, 0.3183, 0.3296, 0.464, 0.3965, 0.4347, 0.4267, 0.3041,
                        0.5116, 0.4297])
        h_score = np.array([53.4583, 57.7621, 89.2963, 67.8296, 71.9401, 73.15, 72.9655, 73.5696, 85.0681, 88.817,
                            92.7606, 71.7094, 73.846])
        logme = np.array([1.0867, 1.1263, 1.195, 1.0506, 1.1012, 1.1127, 1.1238, 1.1538, 1.1946, 1.207, 1.3712,
                          1.1084, 1.1363])

    elif case == 3:
        print('Target: Caltech-256')
        id_list = ['A1311', 'A1411', 'A1412', 'A1413', 'A1414', 'A1415', 'A1416', 'A1711', 'A1712', 'A1713', 'A1714',
                   'A1715', 'A1716']

        transrate = extract_transrate(id_list, eps=1e-2, use_Z_proj=True, normalize_eigz=True)

        acc = np.array([0.7844, 0.8021, 0.8239, 0.5657, 0.6983, 0.7516, 0.793, 0.8038, 0.8217, 0.8357, 0.8386,
                        0.7175, 0.7826])
        nce = np.array([-2.1283, -1.8915, -1.9081, -3.412, -2.7283, -2.363, -2.2325, -1.4863, -1.3785, -1.4291,
                         -1.4456, -2.6164, -2.1319])
        leep = np.array([-2.6353, -2.2806, -2.3084, -4.0048, -3.2229, -2.7896, -2.6729, -1.9195, -1.7155, -1.787,
                         -2.0959, -3.4937, -2.5099])
        lfc = np.array([0.277, 0.3366, 0.2969, 0.2329, 0.2944, 0.3159, 0.4617, 0.3778, 0.4503, 0.442,
                        0.3084, 0.5442, 0.4349])
        h_score = np.array([57.0536, 66.3607, 120.8305, 63.9218, 76.2908, 81.1525, 83.8847, 92.3724, 116.0468, 122.7543,
                            147.0105, 76.3352, 86.0607])
        logme = np.array([1.4387, 1.4602, 1.5289, 1.4183, 1.4434, 1.4536, 1.464, 1.4739, 1.504, 1.5085, 1.6458,
                          1.4453, 1.4683])


    elif case == 4:
        print('Target: SUN397')
        id_list = ['A1341', 'A1441', 'A1442', 'A1443', 'A1444', 'A1445', 'A1446', 'A1741', 'A1742', 'A1743', 'A1744',
                   'A1745', 'A1746']

        transrate = extract_transrate(id_list, eps=7e-3, use_Z_proj=True, normalize_eigz=True)
        # transrate = extract_transrate(id_list, eps=1e-2, use_Z_proj=True, normalize_eigz=True)

        acc = np.array([0.529713, 0.544766, 0.582086, 0.404343, 0.466952, 0.509945, 0.552474, 0.567123, 0.582136, 0.591144,
                    0.583295, 0.496705, 0.546146])
        nce = np.array([-3.5545, -3.3159, -3.3684, -4.4518, -4.0388, -3.7164, -3.6056, -3.0558, -2.9476, -3.0179,
                        -3.0938, -3.8872, -3.5207])
        leep = np.array([-4.1713, -3.8587, -3.9156, -4.9365, -4.52, -4.2045, -4.1001, -3.6232, -3.3694, -3.4582,
                         -3.7617, -4.7025, -3.9819])
        lfc = np.array([0.2216, 0.2768, 0.2379, 0.1874, 0.2291, 0.2583, 0.3617, 0.303, 0.3829, 0.3708, 0.2743,
                        0.4539, 0.3783])
        h_score = np.array([38.8592, 43.562, 95.4907, 56.4381, 62.8448, 65.7015, 66.6171, 67.2303, 90.7404, 97.6819,
                            110.293, 61.6177, 68.443])
        logme = np.array([1.5984, 1.6034, 1.6187, 1.5954, 1.6016, 1.6048, 1.6079, 1.6088, 1.6177, 1.6183, 1.642,
                          1.6025, 1.6093])



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
    model_selection(4)
