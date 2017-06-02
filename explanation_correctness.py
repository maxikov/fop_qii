#standard library
import argparse
import os.path
import pickle
import random
import collections
import sys

#project files
import rating_explanation
import rating_qii
import tree_qii
import shadow_model_qii
import parsers_and_loaders

#pyspark libraryb
from pyspark import SparkConf, SparkContext

def sort_dict(src, non_zero=False):
    lst = sorted(src.items(), key=lambda x: -abs(x[1]))
    if non_zero:
        lst = [x for x in lst if x[1] != 0]
    return lst

def explanation_correctness(qiis, user_profile):
    qii_list = sort_dict(qiis)
    abs_qii_list = [(f, abs(q)) for (f, q) in qii_list]
    top_two_qii = abs_qii_list[0][1] + abs_qii_list[1][1]
    if top_two_qii == 0:
        top_two_qii = 1e-30
    qiis = collections.defaultdict(lambda: 0.0, qiis)
    pos_qii = abs(qiis[user_profile["pos"]])
    neg_qii = abs(qiis[user_profile["neg"]])
    corr = (pos_qii + neg_qii) / float(top_two_qii)
    return corr

def one_rating_correctness(user, movie, user_features, all_trees, indicators,
                           user_profile, feature_names,
                           iterations=10, indicator_distributions=None,
                           used_features=None, debug=False, model=None):
    if model is not None and debug:
        r = model.predict(user, movie)
        print "Original predicted rating:", r
        sys.stdout.flush()
    qiis = shadow_model_qii.shadow_model_qii(user, movie,
            user_features, all_trees, indicators, iterations,
             indicator_distributions, used_features, False)
    if debug:
        print "User profile:", user_profile
        print "Non-zero qiis:"
        for f, q in sort_dict(qiis, True):
            print "{} ({}): {}".format(feature_names[f], f, q)
    corr = explanation_correctness(qiis, user_profile)
    if debug:
        print "Correctness score:", corr
    return corr

def sample_correctness(user_product_pairs, user_features, all_trees,
                       indicators, users, profiles, feature_names, movies_dict, iterations=10,
                       indicator_distributions=None, used_features=None,
                       debug=False, model=None):
    res = []
    for (n, (u, m)) in enumerate(user_product_pairs):
        if debug:
            print "Processing user {}, movie {} ({})"\
                    .format(u, m, movies_dict[m])
            sys.stdout.flush()
        corr = one_rating_correctness(u, m, user_features, all_trees,
                indicators, profiles[users[u]], feature_names, iterations,
                indicator_distributions, used_features, debug, model)
        res.append(corr)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist-dir", action="store", type=str, help=\
                        "Path from which to loaad models and features to analyze")
    parser.add_argument("--dataset-dir", action="store", type=str, help=\
                        "Path from which to load user profiles")
    parser.add_argument("--qii-iterations", action="store", type=int, default=10, help=\
                        "Iterations to use with QII. 10 by default.")
    parser.add_argument("--sample-size", action="store", type=int, default=10,
            help="Number of user-movie pairs to evaluate correctness of")
    parser.add_argument("--movies-file", action="store", type=str, help=\
                        "File from which to load movie names")
    args = parser.parse_args()
    conf = SparkConf().setMaster("local[*]")
    sc = SparkContext(conf=conf)

    print "Loading results dict"
    results = rating_explanation.load_results_dict(args.persist_dir)

    print "Original recommender relative to ground truth on training set mean absolute error: {}, RMSE: {}"\
            .format(results["baseline_rec_eval"]["mean_abs_err"],results["baseline_rec_eval"]["rmse"])
    print "Shadow model relative to the baseline recommender on test set MAE: {}, RMSE: {}"\
            .format(results["all_replaced_rec_eval_test"]["mean_abs_err"],results["all_replaced_rec_eval_test"]["rmse"])
    print "Randomized model relative to the baseline recommender on test set MAE: {}, RMSE: {}"\
            .format(results["all_random_rec_eval_test"]["mean_abs_err"],results["all_random_rec_eval_test"]["rmse"])
    print "Shadow model is {} times better than random on the test set"\
            .format(results["all_random_rec_eval_test"]["mean_abs_err"]/results["all_replaced_rec_eval_test"]["mean_abs_err"])
    sys.stdout.flush()

    if args.movies_file is not None:
        if ".csv" in args.movies_file:
            msep = ","
            movies_remove_first_line = True
        else:
            msep = "::"
            movies_remove_first_line = False
        print "Loading movies"
        movies_rdd = sc.parallelize(
            parsers_and_loaders.loadCSV(
                args.movies_file,
                remove_first_line=movies_remove_first_line
            )
            ).cache()
        movies_dict = dict(movies_rdd.map(lambda x: parsers_and_loaders.parseMovie(x,\
            sep=msep)).collect())
    else:
        movies = product_features.keys().collect()
        movies_dict = {x: str(x) for x in movies}

    print "Loading ALS model"
    model = rating_qii.load_als_model(sc, os.path.join(args.persist_dir,
                                                       "als_model.pkl"))
    rank = model.rank
    user_features = model.userFeatures()

    print "Loading decision trees"
    all_trees = {}
    for i in xrange(rank):
        all_trees[i] = rating_explanation.load_regression_tree_model(sc, args.persist_dir, i)
    all_used_features = shadow_model_qii.get_all_used_features(all_trees)

    print "{} features are used: {}".format(len(all_used_features), ", "\
            .join("{}: {}".format(f, results["feature_names"][f])
                for f in all_used_features))

    sys.stdout.flush()
    print "Loading indicators"
    (training_movies, test_movies, features_test, features_training,
             features_original_test, features_original_training,
             indicators_training, indicators_test) =\
        tree_qii.load_features_indicators(os.path.join(args.persist_dir,
                                              "features_training_test.pkl"), sc, 7)
    print "Building indicator distributions"
    indicators = indicators_training.union(indicators_test).sortByKey().cache()
    products = indicators.keys().collect()
    all_indicator_distributions = shadow_model_qii.get_all_feature_distributions(indicators,
            all_used_features)

    sys.stdout.flush()

    print "Loading user profiles"
    profiles, users = pickle.load(open(os.path.join(args.dataset_dir,
                                                    "profiles.pkl"),
                                       "rb"))
    sys.stdout.flush()
    users_list = users.keys()

    user_product_pairs = [(random.choice(users_list), random.choice(products))
            for _ in xrange(args.sample_size)]

    corr = sample_correctness(user_product_pairs, user_features, all_trees,
                       indicators, users, profiles, results["feature_names"],
                       movies_dict, args.qii_iterations,
                       all_indicator_distributions, all_used_features,
                       debug=True, model=model)

    print "Correctness scores:", corr
    print "Average correctness:", sum(corr)/float(len(corr))

    sys.stdout.flush()
if __name__ == "__main__":
    main()
