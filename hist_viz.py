#!/usr/bin/env python

import argparse
import math

from matplotlib import pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fname", type=str, nargs=1, help="Log file name")
    args = parser.parse_args()

    f = open(args.fname[0], "r")
    src = f.readlines()
    f.close()

    src = [x for x in src if  "Evaluation of recommender with replaced feature" in x]
    src = [x.split("Evaluation of recommender with replaced feature ")[1] for x
            in src]
    src = [x.split(" on test set: ") for x in src]
    src = [(int(x[0]), eval(x[1])) for x in src]
    src.sort(key=lambda x: x[0])

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
