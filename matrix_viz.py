#!/usr/bin/env python

import argparse
import numpy

from matplotlib import pyplot as plt

def extract_and_sort(src, training=False, top_results=0, coeff_threshold=None,
        is_qii=False):
    tr = "" if training else "_test"
    rf = src["features"]
    regs = [x for x in rf.items() if x[1]["type"] == "regression"]
    mrae = ("mrae", "MRAE")
    regs.sort(key = lambda x: x[1]["eval"+tr][mrae[0]])
    if top_results > 0:
        regs = regs[:top_results]
    new_regs = []
    for f, info in regs:
        info["title"] = "{} ({}: {:1.3f})".format(\
                info["name"], mrae[1], info["eval"+tr][mrae[0]])
        if coeff_threshold is not None:
            info["qii" if is_qii else "weights"] = [x if abs(x) < coeff_threshold else
                    coeff_threshold for x in info["qii" if is_qii else "weights"]]
        new_regs.append(info)
    clss = [x for x in rf.items() if x[1]["type"] == "classification"]
    if "not_linlog" in src:
        better = ("recall", " recall")
    else:
        better = ("better", "x better")
    clss.sort(key = lambda x: -x[1]["eval"+tr][better[0]])
    if top_results > 0:
        clss = clss[:top_results]
    new_clss = []
    for f, info in clss:
        info["title"] = "{} ({:2.3f}{})".format(\
                info["name"], info["eval"+tr][better[0]],
                better[1])
        if coeff_threshold is not None:
            info["qii" if is_qii else "weights"] = [x if abs(x) < coeff_threshold else
                    coeff_threshold for x in info["qii" if is_qii else "weights"]]
        new_clss.append(info)
    res = new_regs + new_clss
    return res


def display_matrix(reg_models_res, is_qii=False):
    matrix = [x["qii" if is_qii else "weights"] for x in reg_models_res]
    matrix = numpy.array(matrix)
    fig, ax = plt.subplots()
    cax = ax.imshow(matrix, cmap='viridis', interpolation='nearest')
    ax.set_yticks(range(len(reg_models_res)))
    ax.set_yticklabels(x["title"] for x in reg_models_res)
    ax.set_ylabel("Metadata features")
    ax.set_xticks(range(len(reg_models_res[0]["weights"])))
    ax.set_xticklabels(range(len(reg_models_res[0]["weights"])))
    ax.set_xlabel("Internal features")
    if is_qii:
        ax.set_title("Feature global influence")
    else:
        ax.set_title("Coefficients for regression")
    cbar = fig.colorbar(cax)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", action="store_true", help=\
            "Show data for training instead of test set")
    parser.add_argument("--top-results", action="store", type=int,
                        default=0, help="Only display n top performing "+\
                        "predictors in each category. If 0 (default) "+\
                        "all are displayed")
    parser.add_argument("--coeff-threshold", action="store", type=float,
                        help="If specified, regression coefficients whose "+\
                             "absolute value exceeds the specified number "+\
                             "are thresholded")
    parser.add_argument("--qii", action="store_true", help="Display QII "+\
                        "instead of coefficients")
    parser.add_argument("fname", type=str, nargs=1, help="Log file name")
    args = parser.parse_args()

    f = open(args.fname[0], "r")
    src = f.readlines()
    f.close()

    src = [x for x in src if  "Overall results dict" in x]
    src = [x.split("Overall results dict: ")[1] for x in src]
    src = src[0]
    results = eval(src)

    is_qii = args.qii or ("linlog" not in results)

    reg_models_res = extract_and_sort(results, args.training, args.top_results,
                                      args.coeff_threshold, is_qii)
    display_matrix(reg_models_res, is_qii)

    plt.show()

if __name__ == "__main__":
    main()
