#standard library
import argparse
import os.path
import pickle

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

def load_features_indicators(fname, sc, num_partitions):
            ifile = open(fname, "rb")
            objects = pickle.load(ifile)
            ifile.close()
            (training_movies, test_movies, features_test_c, features_training_c,
             features_original_test_c, features_original_training_c,
             indicators_training_c, indicators_test_c) = objects
            indicators_training = sc.parallelize(indicators_training_c)\
                    .repartition(num_partitions)\
                    .cache()
            indicators_training = sc.parallelize(indicators_training_c)\
                    .repartition(num_partitions)\
                    .cache()
            features_training = sc.parallelize(features_training_c)\
                    .repartition(num_partitions)\
                    .cache()
            features_original_training = sc.parallelize(features_original_training_c)\
                    .repartition(num_partitions)\
                    .cache()
            if indicators_test_c is not None:
                indicators_test = sc.parallelize(indicators_test_c)\
                    .repartition(num_partitions)\
                    .cache()
            else:
                features_test = None
            if features_test_c is not None:
                features_test = sc.parallelize(features_test_c)\
                    .repartition(num_partitions)\
                    .cache()
            else:
                features_test = None
            if features_original_test_c is not None:
                features_original_test = sc.parallelize(features_original_test_c)\
                    .repartition(num_partitions)\
                    .cache()
            else:
                features_original_test = None
            if indicators_test_c is not None:
                indicators_test = sc.parallelize(indicators_test_c)\
                    .repartition(num_partitions)\
                    .cache()
            else:
                indicators_test = None
            return (training_movies, test_movies, features_test, features_training,
             features_original_test, features_original_training,
             indicators_training, indicators_test)

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

    print "Loading the tree model"
    lr_model = rating_explanation.load_regression_tree_model(sc, args.persist_dir, args.feature)
    print "Loading results dict"
    results = rating_explanation.load_results_dict(args.persist_dir)
    print "Loading feature indicator sets"
    (training_movies, test_movies, features_test, features_training,
             features_original_test, features_original_training,
             indicators_training, indicators_test) = load_features_indicators(os.path.join(args.persist_dir, "features_training_test.pkl"), sc, 4)

    print lr_model.toDebugString()

    print "Used features:", ", ".join(["{} ({})".format(x, results["feature_names"][x]) for x in get_used_features(lr_model)])


if __name__ == "__main__":
    main()
