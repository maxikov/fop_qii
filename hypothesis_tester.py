#!/usr/bin/env python

import os.path
import glob
import argparse
from colorama import Back, Style
import numpy
import scipy.stats

def get_recommender_error(fname):
    with open(fname, "r") as f:
        for line in f:
            if "Original recommender relative to ground truth" in line:
                merr = line.split("mean absolute error: ")[1].split(", ")[0]
                merr = float(merr)
                break
    return merr

def get_shadow_faith(fname):
    with open(fname, "r") as f:
        for line in f:
            if "Shadow model is " in line:
                merr = line.split("Shadow model is ")[1].split(" ")[0]
                merr = float(merr)
                break
    return merr

def get_group_values(data_root, control_or_experimental, rec_or_shadow):
    path_suffix = os.path.join("hypothesis_testing",
                                control_or_experimental, "logs")
    fname_mask = "explanation_correctness_new_synth_{}_subj_{}.txt"\
                 .format(control_or_experimental, "{}")
    all_files = glob.glob(os.path.join(data_root, path_suffix,
                          fname_mask.format("*")))
    print "{} subjects found in {} group".format(len(all_files),
                                                 control_or_experimental)
    res = []
    for fname in all_files:
        if rec_or_shadow == "rec":
            cur = get_recommender_error(fname)
        else:
            cur = get_shadow_faith(fname)
        res.append(cur)
    return res


def get_average_correctness(fname):
    with open(fname, "r") as f:
        for line in f:
            if "Average correctness: " in line:
                corr = float(line.split(": ")[1])
                break
    return corr

def get_all_corrs(fname):
    res = []
    with open(fname, "r") as f:
        for line in f:
            if "Correctness score: " in line:
                corr = float(line.split(": ")[1])
                res.append(corr)
    return res

def get_group_correctnesses(data_root, control_or_experimental):
    path_suffix = os.path.join("hypothesis_testing",
                                control_or_experimental, "logs")
    fname_mask = "explanation_correctness_new_synth_{}_subj_{}.txt"\
                 .format(control_or_experimental, "{}")
    all_files = glob.glob(os.path.join(data_root, path_suffix,
                          fname_mask.format("*")))
    print "{} subjects found in {} group".format(len(all_files),
                                                 control_or_experimental)
    res = []
    for fname in all_files:
        try:
            corr = get_average_correctness(fname)
        except Exception as e:
            print Back.RED + "Loading average correctness from {} failed, trying to load individually"\
                             .format(fname) + Style.RESET_ALL
            corrs = get_all_corrs(fname)
            if len(corrs) > 0:
                print Back.RED + "{} values found, averaging"\
                                 .format(len(corrs)) + Style.RESET_ALL
                corr = numpy.mean(corrs)
            else:
                print Back.RED + "no values found"
        res.append(corr)
    return res

def effect_size(control, experimental):
    #Effect size for unequal variance
    #https://stats.stackexchange.com/questions/210352/do-cohens-d-and-hedges-g-apply-to-the-welch-t-test/247011#247011
    #Bonett (2007) https://www.ncbi.nlm.nih.gov/pubmed/18557680
    cmean = numpy.mean(control)
    emean = numpy.mean(experimental)
    cstd = numpy.std(control)
    estd = numpy.std(experimental)
    sd = (float(cstd**2 + estd**2)/2.0)**0.5
    res = abs(cmean - emean) / sd
    return res

def hypothesis_test(control, experimental):
    cmean = numpy.mean(control)
    emean = numpy.mean(experimental)
    if cmean == emean:
        sign = "="
    elif emean > cmean:
        sign = ">"
    else:
        sign = "<"
    #Welch t-test with unequal variance
    statistic, pvalue = scipy.stats.ttest_ind(control, experimental,
                                              equal_var=False)
    effsize = effect_size(control, experimental)
    print "Experimental mean ({}) {} control mean ({}), p={}, effect size {}".\
            format(emean, sign, cmean, pvalue, effsize)

def explanation_correctness_test(data_root):
    print "\n"
    print "Testing the explanation correctness hypothesis"
    print "Loading control group"
    control = get_group_correctnesses(data_root, "control")
    print "Loading experimental group"
    experimental = get_group_correctnesses(data_root, "experimental")
    hypothesis_test(control, experimental)
    print "\n"

def rec_err_test(data_root):
    print "\n"
    print "Testing the original recommender error hypothesis"
    print "Loading control group"
    control = get_group_values(data_root, "control",
            "rec")
    print "Loading experimental group"
    experimental = get_group_values(data_root, "experimental",
            "rec")
    hypothesis_test(control, experimental)
    print "\n"

def shadow_faith_test(data_root):
    print "\n"
    print "Testing the shadow model faithfulness hypothesis"
    print "Loading control group"
    control = get_group_values(data_root, "control",
            "shadow")
    print "Loading experimental group"
    experimental = get_group_values(data_root, "experimental",
            "shadow")
    hypothesis_test(control, experimental)
    print "\n"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", action="store", type=str, help=\
                        "Path from which to load all data points")
    args = parser.parse_args()

    explanation_correctness_test(args.data_root)
    rec_err_test(args.data_root)
    shadow_faith_test(args.data_root)

if __name__ == "__main__":
    main()
