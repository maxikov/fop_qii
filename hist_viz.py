#!/usr/bin/env python

import argparse
import math
import re

from matplotlib import pyplot as plt

def overall_recommender_hist(results):
    r = results["baseline_rec_eval"]
    print r

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fname", type=str, nargs=1, help="Log file name")
    args = parser.parse_args()

    f = open(args.fname[0], "r")
    src = f.readlines()
    f.close()

    src = [x for x in src if  "Overall results dict" in x]
    src = [x.split("Overall results dict: ")[1] for x in src]
    src = src[0]
    src = re.sub("DecisionTreeModel regressor of depth [0-9]+ with"+\
            " [0-9]+ nodes", "'model'", src)
    results = eval(src)
    print results

    overall_recommender_hist(results)
    return

    nplots = len(src)
    plt_rows = int(round(math.sqrt(nplots)))
    plt_cols = int(math.ceil(nplots/float(plt_rows)))

    f, axes = plt.subplots(plt_rows, plt_cols)
    axes_flat = reduce(lambda x, y: list(x)+list(y), axes, [])

    for d, ax in zip(src, axes_flat):
        f, data = d
        ys = data["abs_errors_histogram"][1]
        bin_borders = data["abs_errors_histogram"][0]
        bin_centers = []
        for i in xrange(1, len(bin_borders)):
            bin_centers.append((bin_borders[i-1]+\
                    bin_borders[i])/2.0)
        width = bin_centers[1] - bin_centers[0]
        ax.bar(bin_centers, ys, width)
        ax.set_title("Feature {}".format(f))

    plt.show()

if __name__ == "__main__":
    main()
