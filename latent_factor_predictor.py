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
import internal_feature_predictor

#pyspark libraryb
from pyspark import SparkConf, SparkContext


def logger_init():
    """
    Initializes logger object for logging.
    """

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s '+\
        '- %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

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
    pos_qii = abs(qiis[user_profile["pos_name"]])
    neg_qii = abs(qiis[user_profile["neg_name"]])
    corr = (pos_qii + neg_qii) / float(top_two_qii)
    return corr

def one_rating_correctness(user, movie, user_features, all_trees, indicators,
                           user_profile, feature_names,
                           iterations=10, indicator_distributions=None,
                           used_features=None, debug=False, model=None,
                           user_profile_semi_random=None,
                           user_profile_random=None):
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
    qiis = {feature_names[f]: q for (f, q) in qiis.items()}
    if debug:
        print "New qiis:", qiis
    corr = explanation_correctness(qiis, user_profile)
    if user_profile_semi_random is not None\
            and user_profile_random is not None:
                corr_semi_random = explanation_correctness(qiis,
                        user_profile_semi_random)
                corr_random = explanation_correctness(qiis,
                        user_profile_random)
    else:
        corr_semi_random = None
        corr_random = None
    if debug:
        print "Correctness score:", corr
        if user_profile_semi_random is not None\
            and user_profile_random is not None:
            print "Correctness semirandom score:", corr_semi_random
            print "Correctness random score:", corr_random
    return corr, corr_semi_random, corr_random

def sample_correctness(user_product_pairs, user_features, all_trees,
                       indicators, users, profiles, feature_names, movies_dict, iterations=10,
                       indicator_distributions=None, used_features=None,
                       debug=False, model=None, profiles_semi_random=None,
                       profiles_random=None):
    res = []
    res_random = []
    res_semi_random = []
    for (n, (u, m)) in enumerate(user_product_pairs):
        if debug:
            print "Processing user {}, movie {} ({})"\
                    .format(u, m, movies_dict[m])
            sys.stdout.flush()
        if profiles_semi_random is not None and profiles_random is not None:
            user_profile_semi_random = profiles_semi_random[users[u]]
            user_profile_random = profiles_random[users[u]]
        else:
            user_profile_semi_random = None
            user_profile_random = None
        corr, corr_semi_random, corr_random = one_rating_correctness(u, m, user_features, all_trees,
                indicators, profiles[users[u]], feature_names, iterations,
                indicator_distributions, used_features, debug, model,
                user_profile_semi_random, user_profile_random)
        res.append(corr)
        res_semi_random.append(corr_semi_random)
        res_random.append(corr_random)
    return res, res_semi_random, res_random


def main():
    logger = logger_init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist-dir", action="store", type=str, help=\
                        "Path from which to loaad models and features to analyze")
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
    if results["all_replaced_rec_eval_test"]["mean_abs_err"] != 0:
        print "Shadow model is {} times better than random on the test set"\
            .format(results["all_random_rec_eval_test"]["mean_abs_err"]/results["all_replaced_rec_eval_test"]["mean_abs_err"])
    else:
        print "Shadow model is inf times better than random on the test set"
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
    product_features = model.productFeatures()

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

    (model, predictions, observations) = internal_feature_predictor.predict_internal_feature(product_features, indicators_training, 0, "regression_tree",
                             categorical_features={}, max_bins=32, logger=None,
                             no_threshold=False, is_classifier=False,
                             num_classes=None, max_depth=None)
    
    reg_eval = common_utils.evaluate_regression(predictions,
                                                        observations,
                                                        logger,
                                                        32,
                                                        bin_range=None,
                     model_name = "Training feature 0 without other latents")


    indicators_training = indicators_training.join(product_features)\
           .map(lambda (x, (y, z[1:])): (x, y+z))

    (model, predictions, observations) = internal_feature_predictor.predict_internal_feature(product_features, indicators_training, 0, "regression_tree",
                             categorical_features={}, max_bins=32, logger=None,
                             no_threshold=False, is_classifier=False,
                             num_classes=None, max_depth=None)
    
    reg_eval = common_utils.evaluate_regression(predictions,
                                                        observations,
                                                        None,
                                                        32,
                                                        bin_range=None,
                     model_name = "Training feature 0 with other latents")
    print reg_eval


    sys.stdout.flush()
if __name__ == "__main__":
    main()
