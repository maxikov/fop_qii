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

def shadow_model_qii(sc, user, movie, user_features, all_trees, indicators,
                     iterations=10, indicator_distributions=None,
                     used_features=None, debug=False):
    if used_features is None:
        used_features = get_all_used_features(all_trees)
    if indicator_distributions is None:
        indicator_distributions = get_all_feature_ditributions(idicators,
                                                               used_features)
    rank = len(all_trees)
    qiis = {}
    filter_f = functools.partial(lambda movie, (mid, _): mid == movie, movie)
    indicators = indicators.filter(filter_f)
    original_values = indicators.lookup(movie)[0]
    original_prediction = shadow_predict(user, movie, rank, user_features,
                                         all_trees, indicators)
    original_predicted_product_features =\
        [all_trees[f].predict(original_values) for f in xrange(rank)]
    this_user_features = user_features.lookup(user)[0]
    for i_f, f in enumerate(used_features):
        if debug:
            print "Processing feature {} ({} out of {})".format(
                    f, i_f, len(used_features))
        cur_qii = 0.0
        cur_signed_qii = 0.0
        sampled_values = [random.choice(indicator_distributions[f]) for _ in\
                xrange(iterations)]
        perturbed_indicators = [common_utils.set_list_value(original_values,
                                f, sv) for sv in sampled_values]
        pirdd = sc.parallelize(perturbed_indicators, 8)
        predicted_product_features = None
        for lf in xrange(rank):
            cur_f = all_trees[lf].predict(pirdd).map(lambda x: [x])
            if predicted_product_features is None:
                predicted_product_features = cur_f
            else:
                predicted_product_features = common_utils.safe_zip(
                        predicted_product_features,
                        cur_f).map(lambda (x, y): x+y)
        predicted_ratings = shadow_predict_all(this_user_features,
            predicted_product_features)
        signed_errors = predicted_ratings.map(functools.partial(
            lambda op, pp: op-pp, original_prediction))
        abs_errors = signed_errors.map(abs)
        cur_qii = abs_errors.sum()
        cur_signed_qii = signed_errors.sum()
        cur_qii = cur_qii / float(iterations)
        cur_signed_qii = cur_signed_qii / float(iterations)
        if cur_signed_qii == 0:
            sign = 1
        else:
            sign = cur_signed_qii / abs(cur_signed_qii)
        cur_qii *= sign
        qiis[f] = cur_qii
    return qiis

def exact_histogram(distr):
    uniques = set(distr)
    res = {u:0 for u in uniques}
    for i in distr:
        res[i] += 1
    return res

def get_all_feature_distributions(features, used_features):
    used_features = sorted(list(used_features))
    map_f = functools.partial(lambda used_features, (mid, ftrs):
            (mid, {f:[ftrs[f]] for f in used_features}), used_features)
    reduce_f = functools.partial(lambda used_features, foo, bar:\
            {f:foo[f]+bar[f] for f in used_features}, used_features)
    res = features.map(map_f).values().reduce(reduce_f)
    return res

def get_all_used_features(all_trees):
    res = set()
    for tree in all_trees.values():
        res = res.union(set(tree_qii.get_used_features(tree)))
    return res

def shadow_predict(user, movie, rank, user_features, all_trees, indicators):
    res = 0
    this_user_features = user_features.lookup(user)[0]
    this_movie_indicators = indicators.lookup(movie)[0]
    for f in xrange(rank):
        uf = this_user_features[f]
        tree_model = all_trees[f]
        pf = tree_model.predict(this_movie_indicators)
        res += uf * pf
    return res

def shadow_predict_all(this_user_features, all_product_features):
    res = all_product_features.map(
            functools.partial(
                lambda tuf, pf: sum(x*y for (x,y) in zip(tuf, pf)),
                this_user_features))
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist-dir", action="store", type=str, help=\
                        "Path from which to loaad models and features to analyze")
    parser.add_argument("--movie", action="store", type=int, help=\
                        "Movie for which to compute the QII")
    parser.add_argument("--user", action="store", type=int, help=\
                        "User for who to compute QII")
    parser.add_argument("--qii-iterations", action="store", type=int, default=10, help=\
                        "Iterations to use with QII. 10 by default.")
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

    print "Loading ALS model"
    model = rating_qii.load_als_model(sc, os.path.join(args.persist_dir,
                                                       "als_model.pkl"))
    rank = model.rank
    user_features = model.userFeatures()
    print "Loaded rank {} model".format(rank)
    print "Cur user features:", user_features.lookup(args.user)
    print "Cur product features:", model.productFeatures().lookup(args.movie)
    print "Loading decision trees"
    all_trees = {}
    for i in xrange(rank):
        all_trees[i] = rating_explanation.load_regression_tree_model(sc, args.persist_dir, i)
    all_used_features = get_all_used_features(all_trees)
    print "{} features are used: {}".format(len(all_used_features), ", "\
            .join("{}: {}".format(f, results["feature_names"][f])
                for f in all_used_features))
    original_predicted_rating = model.predict(args.user, args.movie)
    print "Original predicted rating:", original_predicted_rating
    print "Loading indicators"
    (training_movies, test_movies, features_test, features_training,
             features_original_test, features_original_training,
             indicators_training, indicators_test) =\
        tree_qii.load_features_indicators(os.path.join(args.persist_dir,
                                              "features_training_test.pkl"), sc, 7)
    indicators = indicators_training.union(indicators_test).sortByKey().cache()
    all_indicator_distributions = get_all_feature_distributions(indicators,
            all_used_features)
    print "Used indicator distributions:"
    for f in sorted(list(all_used_features)):
        hist = exact_histogram(all_indicator_distributions[f])
        print "{} ({}): {}".format(results["feature_names"][f], f, hist)
    shadow_predicted_rating = shadow_predict(args.user, args.movie, rank,
                                              user_features, all_trees,
                                              indicators)
    print "Shadow predicted rating: {} ({} away from original)"\
            .format(shadow_predicted_rating, abs(shadow_predicted_rating -\
                    original_predicted_rating))

    qiis =  shadow_model_qii(sc, args.user, args.movie, user_features, all_trees, indicators,
                     iterations=args.qii_iterations,
                     indicator_distributions=all_indicator_distributions,
                     used_features=all_used_features, debug=True)

    print "Feature influences:"
    qiis_list = sorted(qiis.items(), key=lambda x: -abs(x[1]))
    for f, q in qiis_list:
        print "{} ({}): {}".format(results["feature_names"][f], f, q)
if __name__ == "__main__":
    main()
