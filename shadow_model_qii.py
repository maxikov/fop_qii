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

class PrintLogger:
    def debug(self, x):
        print x

def shadow_model_qii_all_movies(user, user_features, all_trees, indicators,
                     indicator_distributions=None,
                     used_features=None, debug=False):
    if used_features is None:
        used_features = get_all_used_features(all_trees)
    if indicator_distributions is None:
        indicator_distributions = get_all_feature_ditributions(idicators,
                                                               used_features)
    rank = len(all_trees)
    qiis = {}
    original_predictions = shadow_predict_all_movies(user, rank, user_features,
                                                     all_trees, indicators)
    for i_f, f in enumerate(used_features):
        print "Replacing feature {} ({} out of {})".format(f, i_f,
                len(used_features))
        perturbed_indicators = common_utils.perturb_feature(indicators,
                                                            f, None)
        new_predictions = shadow_predict_all_movies(user, rank, user_features,
                                                     all_trees, perturbed_indicators)
        diffs = [original_predictions[m] - new_predictions[m] for m in
                original_predictions.keys()]
        abs_diffs = map(abs, diffs)
        cur_qii = sum(abs_diffs)/float(len(abs_diffs))
        cur_signed_qii = sum(diffs)/float(len(diffs))
        if cur_signed_qii == 0:
            sign = 1
        else:
            sign = cur_signed_qii / abs(cur_signed_qii)
        cur_qii *= sign
        qiis[f] = cur_qii
    return qiis

def shadow_model_qii(user, movie, user_features, all_trees, indicators,
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
    for i_f, f in enumerate(used_features):
        cur_qii = 0.0
        cur_signed_qii = 0.0
        for i in xrange(iterations):
            perturbed_value = random.choice(indicator_distributions[f])
            map_f = functools.partial(lambda pv, f, (mid, ftrs):\
                    (mid, common_utils.set_list_value(ftrs, f, pv)),
                    perturbed_value, f)
            cur_indicators = indicators.map(map_f)
            perturbed_prediction = shadow_predict(user, movie, rank, user_features,
                                                  all_trees, cur_indicators)
            abs_err = abs(original_prediction - perturbed_prediction)
            signed_err = original_prediction - perturbed_prediction
            cur_qii += abs_err
            cur_signed_qii += signed_err
            if debug:
                print "Processed feature {} ({} out of {}), iteration {} out of {}"\
                        .format(f, i_f, len(used_features), i, iterations) +\
                      ". Sampled feature value: {} (vs original {}),".format(perturbed_value, original_values[f]) +\
                      " new rating: {} (vs original {})".format(perturbed_prediction, original_prediction)
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
        print "\tPredicting feature {}".format(f)
        uf = this_user_features[f]
        tree_model = all_trees[f]
        pf = tree_model.predict(this_movie_indicators)
        res += uf * pf
    return res

def shadow_predict_all_movies(user, rank, user_features, all_trees, indicators):
    all_movies = indicators.keys().collect()
    res = {mid: 0.0 for mid in all_movies}
    this_user_features = user_features.lookup(user)[0]
    for f in xrange(rank):
        uf = this_user_features[f]
        tree_model = all_trees[f]
        pfs = tree_model.predict(indicators.values()).collect()
        for i, m in enumerate(indicators.keys().collect()):
            res[m] += pfs[i]
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist-dir", action="store", type=str, help=\
                        "Path from which to load models and features to analyze")
    parser.add_argument("--movie", action="store", type=int, default=None, help=\
                        "Movie for which to compute the QII")
    parser.add_argument("--all-movies", action="store_true", help=\
                        "Compute average QII for all movies for a given user")
    parser.add_argument("--user", action="store", type=int, help=\
                        "User for whom to compute QII")
    parser.add_argument("--qii-iterations", action="store",
                        type=int, default=10, help=\
                        "Iterations to use with QII. 10 by default.")
    parser.add_argument("--output", action="store",
                        type=str, default=None, help=\
                        "Output the QII measurement data to the given file as TSV.")
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
    if not args.all_movies:
        print "Cur product features:", model.productFeatures().lookup(args.movie)
    print "Loading decision trees"
    all_trees = {}
    for i in xrange(rank):
        all_trees[i] = rating_explanation.load_regression_tree_model(sc, args.persist_dir, i)
    all_used_features = get_all_used_features(all_trees)
    print "{} features are used: {}".format(len(all_used_features), ", "\
            .join("{}: {}".format(f, results["feature_names"][f])
                for f in all_used_features))
    if not args.all_movies:
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
    if not args.all_movies:
        shadow_predicted_rating = shadow_predict(args.user, args.movie, rank,
                                              user_features, all_trees,
                                              indicators)
        print "Shadow predicted rating: {} ({} away from original)"\
            .format(shadow_predicted_rating, abs(shadow_predicted_rating -\
                    original_predicted_rating))

        qiis =  shadow_model_qii(args.user, args.movie, user_features, all_trees, indicators,
                     iterations=args.qii_iterations,
                     indicator_distributions=all_indicator_distributions,
                     used_features=all_used_features, debug=True)
    else:
        qiis = shadow_model_qii_all_movies(args.user, user_features, all_trees, indicators,
                     indicator_distributions=all_indicator_distributions,
                     used_features=all_used_features, debug=True)
    print "Feature influences:"
    qiis_list = sorted(qiis.items(), key=lambda x: -abs(x[1]))
    for f, q in qiis_list:
        print "{} ({}): {}".format(results["feature_names"][f], f, q)

    # Write out the measurements to a TSV file if requested by the
    # output command line argument.
    if args.output is not None:
        print "writing qii measurements to {}".format(args.output)

        fout = open(args.output, 'w')
        fout.write("\t".join(['user','movie',
                              'feature_index','feature_name',
                              'influence']) + "\n")
        for f, q in qiis_list:
            fout.write("\t".join(
                map(lambda s: str(s),
                    [args.user, args.movie, f, results["feature_names"][f], q]
                    )) + "\n")

        fout.close()

if __name__ == "__main__":
    main()
