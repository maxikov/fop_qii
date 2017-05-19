#standard library
import random
import argparse
import os.path
import pickle

#project files
import common_utils
import rating_qii
import parsers_and_loaders

#pyspark library
from pyspark import SparkConf, SparkContext
import pyspark.mllib.tree

def load_regression_predictions(state_path, feature):
    fname = os.path.join(state_path, "predictions_{}.pkl".format(feature))
    ifile = open(fname, "rb")
    res = dict(pickle.load(ifile)[0])
    ifile.close()
    return res

def load_regression_tree_model(sc, state_path, feature):
    fname = os.path.join(state_path, "lr_model_{}.pkl".format(feature))
    res = pyspark.mllib.tree.DecisionTreeModel.load(sc, fname)
    return res

def load_results_dict(state_path):
    fname = os.path.join(state_path, "results.pkl")
    ifile = open(fname, "rb")
    res = pickle.load(ifile)
    ifile.close()
    return res

def get_predicted_product_features(sc, state_path, product, rank):
    res = {}
    for f in xrange(rank):
        preds = load_regression_predictions(state_path, f)
        cur_pred = preds[product]
        res[f] = cur_pred
    return res

def get_all_predicted_product_features(sc, state_path, rank):
    res = None
    for f in xrange(rank):
        preds = load_regression_predictions(state_path, f)
        if res is None:
            res = {mid: [pred] for (mid, pred) in preds.items()}
        else:
            for mid in res:
                res[mid].append(preds[mid])
    return res

def compute_feature_prediction_errors(features, predicted_features):
    res = {}
    features = dict(features.collect())
    for mid in predicted_features:
        res[mid] = sum(abs(x-y) for (x,y) in zip(features[mid],
            predicted_features[mid]))/float(len(features[mid]))
    return res

def get_model_string(lr_model, feature_names):
    res = lr_model.toDebugString()
    res = common_utils.substitute_feature_names(res, feature_names)
    return res

def get_branch_depth(string):
    return len(string) - len(string.lstrip(" "))

def get_relevant_branch(model_string, prediction):
    lines = model_string.split("\n")
    if "DecisionTreeModel regressor" in lines[0]:
        lines = lines[1:]
    p_str = "{}".format(prediction)[:8]
    line_to_find = "Predict: {}".format(p_str)
    line_index = next(x[0] for x in enumerate(lines) if line_to_find in x[1])
    last_depth = get_branch_depth(lines[line_index])
    res = [lines[line_index]]
    for i in xrange(line_index-1, -1, -1):
        cur_line = lines[i]
        cur_depth = get_branch_depth(cur_line)
        if cur_depth == last_depth - 1:
            cur_line = cur_line.replace("Else", "If")
            res.insert(0, cur_line)
            last_depth -= 1
    res_str = "\n".join(res)
    return res_str

def get_all_prediction_branches(sc, predicted_product_features, state_path,
                                feature_names):
    res = {}
    for f, prediction in predicted_product_features.items():
        lr_model = load_regression_tree_model(sc, state_path, f)
        model_string = get_model_string(lr_model, feature_names)
        branch = get_relevant_branch(model_string, prediction)
        res[f] = branch
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-path", action="store", type=str, help=\
                        "Path from which to load models to analyze")
    parser.add_argument("--user", action="store", type=int, help=\
                        "User ID to analyze.")
    parser.add_argument("--product", action="store", type=int, help=\
                        "Product ID to analyze.")
    parser.add_argument("--user-or-product", action="store", type=str,
                        default="user", choices=["user", "product", "both"],
                        help="Compute QII by perturbing user or product "+\
                             "features. \"product\" by default.")
    parser.add_argument("--qii-iterations", action="store", type=int,
                        default=10, help="Number of QII iterations. "+\
                                         "10 by default.")
    parser.add_argument("--display-top-movies", action="store", type=int,
                        help="If passed, will display the specified number"+\
                             " of movies, for which product features are"+\
                             " are predicted most accurately, and exit.")
    parser.add_argument("--movies-file", action="store", type=str, help=\
                        "If passed, a file with movie titles will be loaded"+\
                        " to dispaly titles instead of IDs.")
    args = parser.parse_args()
    conf = SparkConf().setMaster("local[*]").set("spark.default.parallelism",
                                                 4)
    sc = SparkContext(conf=conf)

    results_dict = load_results_dict(args.state_path)
    feature_names = results_dict["feature_names"]
    als_model = rating_qii.load_als_model(sc, os.path.join(args.state_path,
                                                           "als_model.pkl"))
    user_features = als_model.userFeatures()
    product_features = als_model.productFeatures()
    rank = als_model.rank

    if args.movies_file is not None:
        if ".csv" in args.movies_file:
            msep = ","
            movies_remove_first_line = True
        else:
            msep = "::"
            movies_remove_first_line = False
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

    if args.display_top_movies is not None:
        all_predicted_features = get_all_predicted_product_features(sc, args.state_path, rank)
        errors = compute_feature_prediction_errors(product_features,
                                                  all_predicted_features)
        res = sorted(errors.items(), key=lambda x:
                abs(x[1]))[:args.display_top_movies]
        for mid, err in res:
            print "Movie:", movies_dict[mid], "(", mid, ")", ", error:", err
        return

    cur_product_features = rating_qii.get_features_by_id(product_features,
                                                         args.product)
    cur_user_features = rating_qii.get_features_by_id(user_features,
                                                      args.user)

    predicted_product_features = get_predicted_product_features(sc, args.state_path,
                                                                args.product, rank)
    predicted_product_features_list = [x[1] for x in sorted(predicted_product_features.items(),
                                                            key=lambda x: x[0])]

    print "Movie:", movies_dict[args.product]
    print "Real product features:", list(cur_product_features)
    print "Predicted product features:",\
        predicted_product_features_list


    qiis, _ = rating_qii.rating_qii(user_features, product_features, args.user,
                                    args.product, args.user_or_product,
                                    args.qii_iterations)

    qii_order = [x[0] for x in sorted(qiis.items(), key=lambda x: -abs(x[1]))]

    actual_predicted_rating = als_model.predict(args.user, args.product)
    regression_predicted_rating = rating_qii.predict_one_rating(\
            cur_user_features, predicted_product_features_list)
    print "Rating predicted by the actual recommender:",\
        actual_predicted_rating
    print "Rating predicted from product features estimated with regression:",\
        regression_predicted_rating
    all_prediction_branches = get_all_prediction_branches(sc,
                                                          predicted_product_features,
                                                          args.state_path,
                                                          feature_names)
    print "Features (from most to least influential):"
    for f in qii_order:
        branch = all_prediction_branches[f]
        print "Product feature:", f
        print "Influence on the rating:", qiis[f]
        print "Actual value:", cur_product_features[f]
        print "Predicted value:", predicted_product_features_list[f]
        print "Prediction branch:\n", branch
        print "\n"


if __name__ == "__main__":
    main()
