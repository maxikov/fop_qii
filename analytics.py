#!/usr/bin/env  python

from prettytable import PrettyTable

def load_data():
    res = []
    log16 =[{'recommendee_recs_l1_norm': 12440.870895847107, 'recommendee_qii_l1_norm': 110.34031836671433, 'perturbations': [{'perturbed_user_id': 5351, 'recs_ls': 24803.92732815367, 'perturbed_qii_l1_norm': 109.6501679140283, 'qii_ls': 219.99048628074257, 'perturbed_recs_l1_norm': 12363.0564323066}, {'perturbed_user_id': 2208, 'recs_ls': 24974.352479683064, 'perturbed_qii_l1_norm': 111.16169919145035, 'qii_ls': 221.50201755816485, 'perturbed_recs_l1_norm': 12533.48158383599}, {'perturbed_user_id': 5265, 'recs_ls': 24914.581696356694, 'perturbed_qii_l1_norm': 110.63158137924304, 'qii_ls': 220.97189974595761, 'perturbed_recs_l1_norm': 12473.71080050967}, {'perturbed_user_id': 1798, 'recs_ls': 24899.19178686823, 'perturbed_qii_l1_norm': 110.49508550794904, 'qii_ls': 220.83540387466354, 'perturbed_recs_l1_norm': 12458.320891021172}, {'perturbed_user_id': 3966, 'recs_ls': 24706.616923165533, 'perturbed_qii_l1_norm': 108.78710445515195, 'qii_ls': 219.12742282186628, 'perturbed_recs_l1_norm': 12265.746027318439}], 'recommendee_user_id': 552}, {'recommendee_recs_l1_norm': 12115.660437208904, 'recommendee_qii_l1_norm': 90.54050146820967, 'perturbations': [{'perturbed_user_id': 2409, 'recs_ls': 24154.43693737859, 'perturbed_qii_l1_norm': 89.96594672144549, 'qii_ls': 180.50644818965517, 'perturbed_recs_l1_norm': 12038.776500169693}, {'perturbed_user_id': 1367, 'recs_ls': 24191.193007609974, 'perturbed_qii_l1_norm': 90.24062535312173, 'qii_ls': 180.7811268213314, 'perturbed_recs_l1_norm': 12075.532570401092}, {'perturbed_user_id': 5067, 'recs_ls': 24227.3966111016, 'perturbed_qii_l1_norm': 90.51117539305372, 'qii_ls': 181.05167686126322, 'perturbed_recs_l1_norm': 12111.736173892721}, {'perturbed_user_id': 5800, 'recs_ls': 24253.437558575984, 'perturbed_qii_l1_norm': 90.70577976111446, 'qii_ls': 181.24628122932413, 'perturbed_recs_l1_norm': 12137.777121366977}, {'perturbed_user_id': 2620, 'recs_ls': 24239.91200233149, 'perturbed_qii_l1_norm': 90.60470308837849, 'qii_ls': 181.1452045565881, 'perturbed_recs_l1_norm': 12124.251565122615}], 'recommendee_user_id': 2904}]
    res += log16

    uid_1 =[{'recommendee_recs_l1_norm': 11814.821318131742, 'recommendee_qii_l1_norm': 98.18411067699546, 'perturbations': [{'perturbed_user_id': 4435, 'recs_ls': 23649.774371983374, 'perturbed_qii_l1_norm': 98.35141041981913, 'qii_ls': 196.53552109681473, 'perturbed_recs_l1_norm': 11834.953053851626}, {'perturbed_user_id': 1105, 'recs_ls': 23611.393612118492, 'perturbed_qii_l1_norm': 98.03245673673165, 'qii_ls': 196.216567413727, 'perturbed_recs_l1_norm': 11796.572293986774}, {'perturbed_user_id': 462, 'recs_ls': 23727.888731533847, 'perturbed_qii_l1_norm': 99.00056022217765, 'qii_ls': 197.18467089917291, 'perturbed_recs_l1_norm': 11913.067413402072}, {'perturbed_user_id': 5668, 'recs_ls': 23588.791623464625, 'perturbed_qii_l1_norm': 97.8446285761743, 'qii_ls': 196.02873925316968, 'perturbed_recs_l1_norm': 11773.970305332925}, {'perturbed_user_id': 3583, 'recs_ls': 23773.936530587474, 'perturbed_qii_l1_norm': 99.38322891237476, 'qii_ls': 197.56733958937014, 'perturbed_recs_l1_norm': 11959.11521245576}], 'recommendee_user_id': 1}]
    res += uid_1

    return res


def main():
    data = load_data()
    table = PrettyTable(["QII ls",
                        "Rec ls",
                        "Rel QII ls",
                        "Rel Rec ls"
                        ])
    for item in data:
        for per in item["perturbations"]:
            table.add_row([
                per["qii_ls"],
                per["recs_ls"],
                per["qii_ls"]/item['recommendee_qii_l1_norm'],
                per["recs_ls"]/item['recommendee_recs_l1_norm'],
            ])

    print table

if __name__ == "__main__":
    main()
