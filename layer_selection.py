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


def layer_selection(case=1):
    print('Layer Selection. Target: CIFAR-100')
    if case == 1:
        print('Source: ImageNet. Network: ResNet-18')

        id_list = ['0001', '0002', '0003', '0004']

        transrate = extract_transrate(id_list, eps=1e-4, use_Z_proj=False, normalize_eigz=True)

        acc = np.array([0.8158, 0.8005, 0.7681, 0.7479])

        lfc = np.array([0.2061, 0.2778, 0.1803, 0.2245])
        h_score = np.array([24.047, 21.9933, 14.0985, 12.2865])
        logme = np.array([1.0126, 0.9995, 0.9564, 0.9455])

    elif case == 2:
        print('Source: Caltech-101. Network: ResNet-18')

        id_list = ['0131', '0132', '0133', '0134']

        transrate = extract_transrate(id_list, eps=1e-4, use_Z_proj=False, normalize_eigz=True)

        acc = np.array([0.8115, 0.7985, 0.7659, 0.7457])

        lfc = np.array([0.1818, 0.1262, 0.0703, 0.0701])
        h_score = np.array([12.1642, 13.0729, 10.1815, 9.7586])
        logme = np.array([0.9399, 0.945, 0.9342, 0.9316])


    elif case == 3:
        print('Source: Caltech-256. Network: ResNet-18')
        id_list = ['0161', '0162', '0163', '0164']

        transrate = extract_transrate(id_list, eps=1e-5, use_Z_proj=False, normalize_eigz=True)
        # transrate = extract_transrate(id_list, eps=1e-4, use_Z_proj=False, normalize_eigz=True)

        acc = np.array([0.8114, 0.8017, 0.7674, 0.7490])

        lfc = np.array([0.1314, 0.0702, 0.0439, 0.0448])
        h_score = np.array([8.8752, 10.6875, 9.1852, 9.4428])
        logme = np.array([0.9224, 0.9324, 0.9289, 0.9300])

    elif case == 4:
        print('Source: SUN397. Network: ResNet-18')

        id_list = ['0251', '0252', '0253', '0254']

        transrate = extract_transrate(id_list, eps=1e-5, use_Z_proj=False, normalize_eigz=True)
        # transrate = extract_transrate(id_list, eps=1e-4, use_Z_proj=False, normalize_eigz=True)

        acc = np.array([0.8008, 0.7903, 0.7643, 0.7453])

        lfc = np.array([0.0963, 0.0579, 0.0480, 0.0513])
        h_score = np.array([9.1715, 10.5121, 8.7682, 8.6996])
        logme = np.array([0.9235, 0.9304, 0.9262, 0.9257])

    elif case == 5:
        print('Source: SVHN. Network: ResNet-20')

        id_list = ['0471', '0472', '0473', '0474', '0475', '0476']

        transrate = extract_transrate(id_list, eps=1e-2, use_Z_proj=False, normalize_eigz=True)
        # transrate = extract_transrate(id_list, eps=1e-4, use_Z_proj=False, normalize_eigz=True)

        acc = np.array([0.4939, 0.4983, 0.4965, 0.511, 0.5091, 0.508])

        lfc = np.array([0.0319, 0.0279, 0.0721, 0.022, 0.0319, 0.0647])
        h_score = np.array([3.0115, 3.6555, 4.0529, 3.0631, 3.11, 2.952])
        logme = np.array([0.9013, 0.9043, 0.9065, 0.9026, 0.9027, 0.9019])


    elif case == 6:
        print('Source: CIFAR-10. Network: ResNet-20')

        id_list = ['0431', '0432', '0433', '0434', '0435', '0436']

        transrate = extract_transrate(id_list, eps=1e-4, use_Z_proj=False, normalize_eigz=True)

        acc = np.array([0.5412, 0.5476, 0.5447, 0.5391, 0.5414, 0.5428])

        lfc = np.array([0.1704, 0.1523, 0.1977, 0.0962, 0.1048, 0.142])
        h_score = np.array([5.5182, 5.8830, 6.0494, 4.1724, 3.9870, 3.8021])
        logme = np.array([0.9142, 0.9165, 0.9172, 0.9082, 0.9071, 0.9061])

    elif case == 7:
        print('Source: ImageNet. Network: ResNet-34')
        id_list = ['0601', '0602', '0603', '0604', '0605', '0606', '0607', '0608', '0609']

        transrate = extract_transrate(id_list, eps=1e-4, use_Z_proj=False, normalize_eigz=True)

        acc = np.array([0.8391, 0.8406, 0.8348, 0.8157, 0.801, 0.7835, 0.7666, 0.7604, 0.7580])

        lfc = np.array([0.2293, 0.3073, 0.3274, 0.1521, 0.1611, 0.1465, 0.1854, 0.1717, 0.1663])
        h_score = np.array([27.5055, 26.6508, 24.8302, 16.5587, 15.8265, 14.8158, 13.8373, 12.7348, 11.6754])
        logme = np.array([1.0356, 1.0306, 1.0173, 0.9714, 0.9670, 0.9607, 0.9549, 0.9483, 0.9420])


    pr = np.zeros(4)
    kt = np.zeros(4)
    wt = np.zeros(4)

    pr[0], kt[0], wt[0] = analyze_correlation(acc, lfc)
    pr[1], kt[1], wt[1] = analyze_correlation(acc, h_score)
    pr[2], kt[2], wt[2] = analyze_correlation(acc, logme)
    pr[3], kt[3], wt[3] = analyze_correlation(acc, transrate)

    np.set_printoptions(edgeitems=30, linewidth=100000)
    print('Measure  :    LFC,   h-Sc,  LogMe,    TrR')
    print('Pearson R: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(pr[0], pr[1], pr[2], pr[3]))
    print('K Tau    : {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(kt[0], kt[1], kt[2], kt[3]))
    print('W Tau    : {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(wt[0], wt[1], wt[2], wt[3]))


if __name__ == '__main__':
    ### options: 1=IM(ResNet18)->CIFAR100, 2=Cal101->CIFAR100, 3=Cal256->CIFAR100, 4=SUN->CIFAR100
    ###          4=SVHN(ResNet20)->CIFAR100, 5=CIFAR10(ResNet20), 6=IM(ResNet34)->CIFAR100
    layer_selection(7)

