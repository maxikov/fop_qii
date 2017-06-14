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
import CustomFeaturesRecommender

#pyspark libraryb
from pyspark import SparkConf, SparkContext

def rate(user_profile, inds):
    res = 3.0 + inds[user_profile["pos"]]\
            - inds[user_profile["neg"]]
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist-dir", action="store", type=str, help=\
                        "Path to saved state (for indicators)")
    parser.add_argument("--data-path", action="store", type=str, help=\
                        "Path to data set (for ratings and profiles)")
    args = parser.parse_args()
    conf = SparkConf().setMaster("local[*]")\
            .set("spark.driver.memory", "4g")\
            .set("spark.executor.memory", "10g")\
            .set("spark.driver.maxResultSize", "4g")\
            .set("spark.default.parallelism", "8")\
            .set("spark.python.worker.memory", "4g")\
            .set("spark.network.timeout", "3600000s")
    sc = SparkContext(conf=conf)

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

    uprs = {u: profiles[p] for u, p in user_profiles.items()}

    rank = len(profiles)
    users = sorted(user_profiles.keys())

    user_features = sc.parallelize(
            [(u, [1.0 if i == user_profiles[u] else 0.0 for i in xrange(rank)])
                for u in users])
    print "User features sample:"
    for x in user_features.take(5):
        print x

    product_features = indicators.map(
            lambda (mid, inds):\
                    (mid,
                        [rate(profiles[i], inds) for i in xrange(rank)]))
    print "product features sample:"
    for x in product_features.take(5):
        print x

    model = CustomFeaturesRecommender.CustomFeaturesRecommender(rank,
            user_features, product_features)
    CustomFeaturesRecommender.save(model, os.path.join(args.data_path,
        "upr_model.pkl"))


if __name__ == "__main__":
    main()
