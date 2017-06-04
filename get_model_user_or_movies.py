#standard library
import argparse
import os.path
import functools
import random

#project files
import rating_explanation
import rating_qii
import tree_qii
import common_utils

#pyspark libraryb
from pyspark import SparkConf, SparkContext

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist-dir", action="store", type=str, help=\
                        "Path from which to load models and features to analyze")
    parser.add_argument("--movies", action="store_true", help=\
                        "Display movies")
    parser.add_argument("--users", action="store_true", help=\
                        "Display users")
    parser.add_argument("--n", action="store", type=int, help=\
                        "Number of users to movies to display")
    args = parser.parse_args()
    conf = SparkConf().setMaster("local[*]")
    sc = SparkContext(conf=conf)

    if args.movies:
        (training_movies, test_movies, features_test, features_training,
             features_original_test, features_original_training,
             indicators_training, indicators_test) =\
        tree_qii.load_features_indicators(os.path.join(args.persist_dir,
                                              "features_training_test.pkl"), sc, 7)
        res = indicators_training.union(indicators_test).keys().take(args.n)
        for r in res:
            print r
    elif args.users:
        model = rating_qii.load_als_model(sc, os.path.join(args.persist_dir,
                                                       "als_model.pkl"))
        res = model.userFeatures().keys().take(args.n)
        for r in res:
            print r

if __name__ == "__main__":
    main()
