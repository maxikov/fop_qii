#standard library
import argparse
import os.path

#project files
import rating_explanation
import rating_qii
import tree_qii

#pyspark libraryb
from pyspark import SparkConf, SparkContext

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist-dir", action="store", type=str, help=\
                        "Path from which to loaad models and features to analyze")
    parser.add_argument("--movie", action="store", type=int, help=\
                        "Movie for which to compute the QII")
    parser.add_argument("--user", action="store", type=int, help=\
                        "User for who to compute QII")
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
    print "Loading decision trees"
    all_trees = {}
    for i in xrange(rank):
        all_trees[i] = rating_explanation.load_regression_tree_model(sc, args.persist_dir, i)
    original_predicted_rating = model.predict(args.user, args.movie)
    print "Original predicted rating:", original_predicted_rating
    print "Loading indicators"
    (training_movies, test_movies, features_test, features_training,
             features_original_test, features_original_training,
             indicators_training, indicators_test) =\
        tree_qii.load_features_indicators(os.path.join(args.persist_dir,
                                              "features_training_test.pkl"), sc, 7)
    indicators = indicators_training.union(indicators_test).sortByKey().cache()
    shadow_predicted_rating = shadow_predict(args.user, args.movie, rank,
                                              user_features, all_trees,
                                              indicators)
    print "Shadow predicted rating: {} ({} away from original)"\
            .format(shadow_predicted_rating, abs(shadow_predicted_rating -\
                    original_predicted_rating))

if __name__ == "__main__":
    main()
