#!/usr/bin/env python

import argparse
import math
import re

from matplotlib import pyplot as plt

def edges_to_centers(edges):
    centers = []
    for i in xrange(1, len(edges)):
        centers.append((edges[i-1]+edges[i])/2.0)
    return centers

def overall_recommender_hist(results, signed=False):
    opacity = 0.4
    r = results["baseline_rec_eval"]
    ys = r["obs_histogram"][1]
    bin_edges = r["obs_histogram"][0]
    width = bin_edges[1] - bin_edges[0]
    tr = plt.bar(bin_edges[:-1], ys, width, label="True rating", color="blue",
                    alpha=opacity)
    ys = r["preds_histogram"][1]
    bin_edges = r["preds_histogram"][0]
    width = bin_edges[1] - bin_edges[0]
    pr = plt.bar(bin_edges[:-1], ys, width, label="Predicted rating",
            color="green", alpha=opacity)
    ys = r[("" if signed else "abs_") + "errors_histogram"][1]
    bin_edges = r[("" if signed else "abs_") + "errors_histogram"][0]
    width = bin_edges[1] - bin_edges[0]
    ar = plt.bar(bin_edges[:-1], ys, width, label="Absolute error",
                    color="red", alpha=opacity)
    plt.legend([ar, tr, pr], ["Error" if signed else "Absolute error", "True rating",
               "Predicted rating"])
    plt.xlabel("Star rating")
    plt.ylabel("Number of observations")
    plt.title("Original recommender model compared to ground truth")
    plt.show()

def create_axes(nplots):
    plt_rows = int(round(math.sqrt(nplots)))
    plt_cols = int(math.ceil(nplots/float(plt_rows)))
    print nplots, plt_rows, plt_cols

    f, axes = plt.subplots(plt_rows, plt_cols)
    axes_flat = []
    for a in axes:
        try:
            axes_flat += list(a)
        except Exception:
            axes_flat += [a]
    return f, axes_flat

def feature_regressions(results, training=False, signed=False):
    opacity = 0.4
    feats = results["features"]
    fig, axes = create_axes(len(feats))
    for f in xrange(len(feats)):
        ax = axes[f]
        data = feats[f]["regression_evaluation" + ("" if training else "_test")]

        xs, ys = data["obs_histogram"]
        print "observations:", xs, ys
        width = xs[1] - xs[0]
        obs = ax.bar(xs[:-1], ys, width, color="blue", alpha=opacity)

        xs, ys = data["preds_histogram"]
        width = xs[1] - xs[0]
        preds = ax.bar(xs[:-1], ys, width, color="green", alpha=opacity)

        xs, ys = data[("" if signed else "abs_") + "errors_histogram"]
        width = xs[1] - xs[0]
        errs = ax.bar(xs[:-1], ys, width, color="red", alpha=opacity)

        ax.set_title("Feature {}".format(f))
        err = data["mean_" + ("" if signed else "abs_") + "err"]
        _abs = "" if signed else "abs "
        ax.set_title("Feature {} (mean {}err: {:1.3f})".format(f, _abs, err))
    fig.suptitle("Performance of regression on internal feature values")
    fig.legend([obs, preds, errs], ["True value", "Predicted value",
                  "Error" if signed else "Absolute error"], loc="lower center")

    plt.show()

def feature_recommenders(results, training=False, signed=False):
    opacity = 0.4
    feats = results["features"]
    fig, axes = create_axes(len(feats)+1)
    for f in xrange(len(feats)):
        ax = axes[f]

        xs, ys = feats[f]\
                      ["replaced_rec_eval" + ("" if training else "_test")]\
                      ["obs_histogram"]
        print "observations:", xs, ys
        width = xs[1] - xs[0]
        obs = ax.bar(xs[:-1], ys, width, color="blue", alpha=opacity)

        xs, ys = feats[f]\
                      ["randomized_rec_eval" + ("" if training else "_test")]\
                      [("" if signed else "abs_") + "errors_histogram"]
        width = xs[1] - xs[0]
        errs = ax.bar(xs[:-1], ys, width, color="red", alpha=opacity)

        xs, ys = feats[f]\
                      ["replaced_rec_eval" + ("" if training else "_test")]\
                      [("" if signed else "abs_") + "errors_histogram"]
        width = xs[1] - xs[0]
        preds = ax.bar(xs[:-1], ys, width, color="green", alpha=opacity)

        err = feats[f]\
                      ["replaced_rec_eval" + ("" if training else "_test")]\
                      ["mean_" + ("" if signed else "abs_") + "err"]
        _abs = "" if signed else "abs "
        ax.set_title("Feature {} (mean {}err: {:1.3f})".format(f, _abs, err))

    ax = axes[len(feats)]
    xs, ys = results["all_replaced_rec_eval" + ("" if training else "_test")]\
                      ["obs_histogram"]
    print "observations:", xs, ys
    width = xs[1] - xs[0]
    obs = ax.bar(xs[:-1], ys, width, color="blue", alpha=opacity)

    xs, ys = results["all_random_rec_eval" + ("" if training else "_test")]\
                      [("" if signed else "abs_") + "errors_histogram"]
    width = xs[1] - xs[0]
    errs = ax.bar(xs[:-1], ys, width, color="red", alpha=opacity)

    xs, ys = results["all_replaced_rec_eval" + ("" if training else "_test")]\
                      [("" if signed else "abs_") + "errors_histogram"]
    width = xs[1] - xs[0]
    preds = ax.bar(xs[:-1], ys, width, color="green", alpha=opacity)
    err = results["all_replaced_rec_eval"]\
                 ["mean_" + ("" if signed else "abs_") + "err"]
    _abs = "" if signed else "abs "
    ax.set_title("All features (mean {}err: {:1.3f})".format(_abs, err))

    fig.suptitle("Performance of recommenders with substituted features")
    fig.legend([obs, preds, errs], ["Original recommender output",
        ("" if signed else "Absolute ") + "error with regression",
        ("" if signed else "Absolute ") + "error randomized"], loc="lower center")

    plt.show()

def main():
    all_programs = ["overall", "feature_regressions", "feature_recommenders"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", type=str, default="overall", choices=\
            all_programs)
    parser.add_argument("--training", action="store_true", help=\
            "Show data for training instead of test set")
    parser.add_argument("--signed", action="store_true", help=\
            "Display signed rather than absolute error histograms")
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
    src = re.sub("<pyspark.mllib.classification.NaiveBayesModel "+\
            "object at 0x[0-9a-f]+>", "'model'", src)
    results = eval(src)

    if args.program == "overall":
        overall_recommender_hist(results, args.signed)
    elif args.program == "feature_regressions":
        feature_regressions(results, args.training, args.signed)
    elif args.program == "feature_recommenders":
        feature_recommenders(results, args.training, args.signed)

    plt.show()

if __name__ == "__main__":
    main()
