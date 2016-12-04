#!/usr/bin/env  python

def load_data():
    res = []
    #log16
    log16 =[{'recommendee_recs_l1_norm': 12440.870895847107, 'recommendee_qii_l1_norm': 110.34031836671433, 'perturbations': [{'perturbed_user_id': 5351, 'recs_ls': 24803.92732815367, 'perturbed_qii_l1_norm': 109.6501679140283, 'qii_ls': 219.99048628074257, 'perturbed_recs_l1_norm': 12363.0564323066}, {'perturbed_user_id': 2208, 'recs_ls': 24974.352479683064, 'perturbed_qii_l1_norm': 111.16169919145035, 'qii_ls': 221.50201755816485, 'perturbed_recs_l1_norm': 12533.48158383599}, {'perturbed_user_id': 5265, 'recs_ls': 24914.581696356694, 'perturbed_qii_l1_norm': 110.63158137924304, 'qii_ls': 220.97189974595761, 'perturbed_recs_l1_norm': 12473.71080050967}, {'perturbed_user_id': 1798, 'recs_ls': 24899.19178686823, 'perturbed_qii_l1_norm': 110.49508550794904, 'qii_ls': 220.83540387466354, 'perturbed_recs_l1_norm': 12458.320891021172}, {'perturbed_user_id': 3966, 'recs_ls': 24706.616923165533, 'perturbed_qii_l1_norm': 108.78710445515195, 'qii_ls': 219.12742282186628, 'perturbed_recs_l1_norm': 12265.746027318439}], 'recommendee_user_id': 552}, {'recommendee_recs_l1_norm': 12115.660437208904, 'recommendee_qii_l1_norm': 90.54050146820967, 'perturbations': [{'perturbed_user_id': 2409, 'recs_ls': 24154.43693737859, 'perturbed_qii_l1_norm': 89.96594672144549, 'qii_ls': 180.50644818965517, 'perturbed_recs_l1_norm': 12038.776500169693}, {'perturbed_user_id': 1367, 'recs_ls': 24191.193007609974, 'perturbed_qii_l1_norm': 90.24062535312173, 'qii_ls': 180.7811268213314, 'perturbed_recs_l1_norm': 12075.532570401092}, {'perturbed_user_id': 5067, 'recs_ls': 24227.3966111016, 'perturbed_qii_l1_norm': 90.51117539305372, 'qii_ls': 181.05167686126322, 'perturbed_recs_l1_norm': 12111.736173892721}, {'perturbed_user_id': 5800, 'recs_ls': 24253.437558575984, 'perturbed_qii_l1_norm': 90.70577976111446, 'qii_ls': 181.24628122932413, 'perturbed_recs_l1_norm': 12137.777121366977}, {'perturbed_user_id': 2620, 'recs_ls': 24239.91200233149, 'perturbed_qii_l1_norm': 90.60470308837849, 'qii_ls': 181.1452045565881, 'perturbed_recs_l1_norm': 12124.251565122615}], 'recommendee_user_id': 2904}]
    res += log16

    return res


def main():
    data = load_data()
    print data

if __name__ == "__main__":
    main()
