#standard library
import argparse
import os.path
import pickle
import random
import collections
import sys
import functools
import time

#project files
import rating_explanation
import rating_qii
import tree_qii
import shadow_model_qii
import parsers_and_loaders
import internal_feature_predictor
import common_utils

#pyspark libraryb
from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import LabeledPoint

class StupidLogger(object):
    def debug(self, *args):
        print args

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist-dir", action="store", type=str, help=\
                        "Path from which to load models and features to analyze")
    parser.add_argument("--data-path", action="store", type=str)
    parser.add_argument("--nbins", action="store", type=int)
    parser.add_argument("--max-depth", action="store", type=int)
    args = parser.parse_args()
    conf = SparkConf().setMaster("local[*]")\
            .set("spark.driver.memory", "4g")\
            .set("spark.executor.memory", "10g")\
            .set("spark.driver.maxResultSize", "4g")\
            .set("spark.default.parallelism", "8")\
            .set("spark.python.worker.memory", "4g")\
            .set("spark.network.timeout", "3600000s")
    sc = SparkContext(conf=conf)
    logger = StupidLogger()

    print "Loading results dict"
    results = rating_explanation.load_results_dict(args.persist_dir)

    print "Loading indicators"
    (training_movies, test_movies, features_test, features_training,
             features_original_test, features_original_training,
             indicators_training, indicators_test) =\
        tree_qii.load_features_indicators(os.path.join(args.persist_dir,
                                              "features_training_test.pkl"), sc, 7)
    indicators = indicators_training.union(indicators_test).sortByKey().cache()
    products = indicators.keys().collect()
    print "Done"
    sys.stdout.flush()
    remove_first_line = True
    extension = ".csv"
    sep=","
    print "Loading ratings"
    start = time.time()
    ratings_rdd = sc.parallelize(
        parsers_and_loaders.loadCSV(
            os.path.join(args.data_path, "ratings" + extension),
            remove_first_line=remove_first_line
            )
        ).cache()
    ratings = ratings_rdd.map(lambda x: parsers_and_loaders.parseRating(x,\
         sep=sep)).values()
    print "Done,", ratings.count(), "ratings loaded"
    sys.stdout.flush()

    profiles, user_profiles = pickle.load(open(os.path.join(args.data_path,
                                                            "profiles.pkl"),
                                               "rb")
                                         )


    print "Filtering known movies"
    ratings = ratings.filter(
            functools.partial(lambda products, x: x[1] in products,
                set(products)))\
            .map(lambda (u, m, r): (m, (u, r)))\
            .sortByKey().cache()
    print ratings.count(), "ratings left"
    print "Building data set"
    last_feature = max(results["feature_names"].keys())
    uid_fid = last_feature + 1
    print "Example rating:", ratings.take(1)
    users = ratings.map(lambda (m, (u, r)): u).distinct()
    print "Example user:", users.take(1)
    all_users = set(users.collect())
    n_users = len(all_users)
    print n_users, "users found, making it a new feature", uid_fid
    results["feature_names"][uid_fid] = "user profile"
    results["categorical_features"][uid_fid] = len(profiles)
    data_set = ratings.join(indicators)\
            .map(functools.partial(lambda up, (m, ((u, r), inds)):
                (inds + [up[u]], m, r), user_profiles ) )
    ids = data_set.map(lambda (_, m, __): m)
    data_set = data_set.map(lambda (inds, m, r):
                LabeledPoint(r, inds))
    print "Data set created, training the tree"
    tree_model = internal_feature_predictor.train_regression_model(data_set,
        regression_model="regression_tree",
        categorical_features=results["categorical_features"],
        max_bins=args.nbins, max_depth=args.max_depth, logger=logger)
    print "Done"
    print tree_model.toDebugString()
    preds = tree_model.predict(data_set.map(lambda x: x.features))
    original_ratings = common_utils.safe_zip(data_set, ids)\
            .map(lambda (lp, mid): (lp.features[-1], mid, lp.label))
    predicted_ratings = common_utils.safe_zip(preds, data_set)
    predicted_ratings = common_utils.safe_zip(predicted_ratings, ids)\
            .map(lambda ((r, lp), mid): (lp.features[-1], mid, r))
    rec_eval = common_utils.evaluate_recommender(original_ratings,
        predicted_ratings, logger, args.nbins, "Tree recommender")
    print rec_eval

if __name__ == "__main__":
    main()
