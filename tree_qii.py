#standard library
import argparse

#project files
import rating_explanation

#pyspark libraryb
from pyspark import SparkConf, SparkContext

def get_used_features(lr_model):
    model_string = lr_model.toDebugString()
    lines = model_string.split("\n")
    res = set()
    for line in lines:
        if "(feature " not in line:
            continue
        right = line.split("(feature ")[1]
        if " not in " in right:
            left = right.split(" not in ")[0]
        elif "<" in line:
            left = right.split("<")[0]
        elif ">" in line:
            left = right.split(">")[0]
        else:
            left = right.split(" in ")[0]
        fid = int(left)
        res.add(fid)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist-dir", action="store", type=str, help=\
                        "Path from which to loaad models and features to analyze")
    parser.add_argument("--qii-iterations", action="store", type=int,
                        default=10, help="Number of QII iterations. "+\
                                         "10 by default.")
    parser.add_argument("--feature", action="store", type=int, help=\
                        "Feature ID of the model to analyze")
    args = parser.parse_args()
    conf = SparkConf().setMaster("local[*]")
    sc = SparkContext(conf=conf)

    lr_model = rating_explanation.load_regression_tree_model(sc, args.persist_dir, args.feature)
    results = rating_explanation.load_results_dict(args.persist_dir)
    print lr_model.toDebugString()

    print "Used features:", ", ".join(["{} ({})".format(x, results["feature_names"][x]) for x in get_used_features(lr_model)])


if __name__ == "__main__":
    main()
