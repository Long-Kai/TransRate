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


def source_selection(case=0):
    print('Source Selection. Network: ResNet-18')
    if case == 1:
        print('Target: CIFAR100')

        id_list = ['0001', '0131', '0161', '0191', '0221', '0251', '0281', '0311', '0341', '0371', '0401']

        transrate = extract_transrate(id_list, eps=1e-4, use_Z_proj=False, normalize_eigz=True)

        acc = np.array([0.8158, 0.8115, 0.8114, 0.8076, 0.7998, 0.8008, 0.8027, 0.7885, 0.7546, 0.7543, 0.7659])
        nce = np.array([-2.7121, -4.0177, -4.2861, -4.2773, -4.3605, -4.2601, -4.3614, -4.3491, -4.3007,
                        -4.2031, -4.4164])
        leep = np.array([-2.881, -4.3003, -4.4539, -4.3354, -4.3966, -4.3697, -4.4897, -4.3909, -4.2787,
                         -4.3187, -4.4256])
        lfc = np.array([0.2061, 0.1818, 0.1314, 0.1733, 0.0748, 0.0963, 0.1307, 0.0621, 0.1147, 0.1149, 0.0604])
        h_score = np.array([24.047, 12.1642, 8.8752, 10.8438, 9.6819, 9.1715, 10.5363, 8.2319, 8.7525, 7.9191, 8.368])
        logme = np.array([1.0126, 0.9399, 0.9224, 0.9325, 0.9261, 0.9235, 0.9304, 0.9187, 0.9211, 0.9167, 0.9193])


    elif case == 2:
        print('Target: Caltech-101')
        id_list = ['A1301', 'A1302', 'A1303', 'A1304', 'A1305', 'A1306', 'A1307', 'A1308', 'A1309', 'A1310']

        transrate = extract_transrate(id_list, eps=1e-1, use_Z_proj=True, normalize_eigz=True)

        acc = np.array([0.9386, 0.9361, 0.9217, 0.9081, 0.9041, 0.9261, 0.8792, 0.8558, 0.8181, 0.8594])
        nce = np.array([-1.2483, -1.1075, -3.2525, -3.1169, -2.0472, -3.6512, -3.2237, -3.3416, -2.1451, -3.0443])
        leep = np.array([-1.8678, -1.4865, -3.5305, -3.5482, -2.9458, -4.0513, -3.9541, -3.8549, -3.4556, -3.7815])
        lfc = np.array([0.2941, 0.4089, 0.3386, 0.259, 0.3733, 0.274, 0.1927, 0.1864, 0.273, 0.22])
        h_score = np.array([53.4583, 59.8433, 49.3024, 42.6633, 46.9551, 50.7588, 36.3448, 34.647, 34.3416, 36.8218])
        logme = np.array([1.0867, 1.1607, 1.0473, 0.9964, 1.0351, 1.0612, 0.9644, 0.9597, 0.9506, 0.9713])


    elif case == 3:
        print('Target: Caltech-256')
        id_list = ['A1311', 'A1312', 'A1313', 'A1314', 'A1315', 'A1316', 'A1317', 'A1318', 'A1319', 'A1320']

        transrate = extract_transrate(id_list, eps=1e-4, use_Z_proj=True, normalize_eigz=True)

        acc = np.array([0.7844, 0.7629, 0.7539, 0.7356, 0.7348, 0.7568, 0.6864, 0.6334, 0.6031, 0.64])
        nce = np.array([-2.1283, -3.085, -4.3912, -4.3228, -3.0926, -4.6575, -4.4699, -4.5562, -3.5743, -4.2455])
        leep = np.array([-2.6353, -3.5692, -4.6061, -4.655, -3.8348, -5.0588, -4.9615, -4.9275, -4.6381, -4.8479])
        lfc = np.array([0.277, 0.3912, 0.3548, 0.2544, 0.389, 0.305, 0.1736, 0.1921, 0.2873, 0.2278])
        h_score = np.array([57.0536, 65.7746, 50.7607, 39.5129, 49.2177, 54.8041, 30.9459, 26.944, 27.0477, 29.8972])
        logme = np.array([1.4387, 1.4617, 1.424, 1.4016, 1.422, 1.4345, 1.3883, 1.3798, 1.3807, 1.3856])

    elif case == 4:
        print('Target: SUN397')
        id_list = ['A1341', 'A1342', 'A1343', 'A1344', 'A1345', 'A1346', 'A1347', 'A1348', 'A1349', 'A1350']

        transrate = extract_transrate(id_list, eps=1e-4, use_Z_proj=True, normalize_eigz=True)

        acc = np.array([0.5297, 0.5154, 0.498, 0.5061, 0.4877, 0.5081, 0.456, 0.4173, 0.4032, 0.423])
        nce = np.array([-3.5545, -4.6488, -3.9043, -5.0868, -4.9521, -5.2551, -4.9131, -4.9862, -4.198, -4.7165])
        leep = np.array([-4.1713, -5.0688, -4.3606, -5.2961, -5.3643, -5.6742, -5.3509, -5.4566, -5.2279, -5.359])
        lfc = np.array([0.2216, 0.2926, 0.3288, 0.3359, 0.2433, 0.2578, 0.1465, 0.1806, 0.2967, 0.2085])
        h_score = np.array([38.8592, 43.7152, 42.7502, 39.8856, 33.0018, 40.9591, 28.8426, 26.4518, 25.1805, 28.5982])
        logme = np.array([1.5984, 1.6038, 1.6026, 1.5993, 1.5916, 1.6007, 1.5872, 1.5854, 1.5844, 1.5872])


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
    source_selection(3)