#!/usr/bin/env  python

from prettytable import PrettyTable
import scipy.stats
import numpy
import matplotlib.pyplot as plt
import os.path
import glob

logs_path = "logs"

def process_file(fname):
    f = open(fname, "r")
    line = f.readlines()[-1]
    f.close()
    data_str = ": ".join(line.split(": ")[1:])
    data = eval(data_str)
    return data

def load_data():
    global logs_path
    files = glob.glob(os.path.join(logs_path, "*.txt"))
    res = []
    for fname in files:
        res += process_file(fname)
    return res


def main():
    data = load_data()
    table = PrettyTable(["Base user",
                        "Perturbed user",
                        "QII LS",
                        "recs LS",
                        "Normalized QII LS",
                        "Normalized recs LS",
                        ])
    qii_lss = []
    recs_lss = []
    qii_lss_norm = []
    recs_lss_norm = []
    for item in data:
        for per in item["perturbations"]:
            qii_lss.append(per["qii_ls"])
            recs_lss.append(per["recs_ls"])
            qii_lss_norm.append(per["qii_ls"]/(item['recommendee_qii_l0_norm']*4))
            recs_lss_norm.append(per["recs_ls"]/(item['recommendee_recs_l0_norm']*4))
            table.add_row([item["recommendee_user_id"],
                per["perturbed_user_id"],
                per["qii_ls"],
                per["recs_ls"],
                per["qii_ls"]/(item['recommendee_qii_l0_norm']*4),
                per["recs_ls"]/(item['recommendee_recs_l0_norm']*4),
            ])

    print table

    table = PrettyTable(["Metric",
                        "QII LS",
                        "Recs LS",
                        "Normalized QII LS",
                        "Normalized recs LS",
                        ])
    stats = [("Max", max),
            ("Mean", numpy.mean),
            ("Median", numpy.median),
            ("STD", numpy.std),
            ("Skewness", scipy.stats.skew),
            ("Max-mean sigmas", lambda x:
                (max(x) - numpy.mean(x))/numpy.std(x)),
            ]

    for name, f in stats:
        table.add_row([name,
            f(qii_lss),
            f(recs_lss),
            f(qii_lss_norm),
            f(recs_lss_norm),
        ])
    print table

    # n, bins, patches = plt.hist(qii_lss)
    # plt.show()

if __name__ == "__main__":
    main()
