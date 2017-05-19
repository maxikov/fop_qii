#standard library
import random
import argparse
import os.path
import pickle

#project files
import common_utils
import rating_qii

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
    args = parser.parse_args()
    conf = SparkConf().setMaster("local[*]")
    sc = SparkContext(conf=conf)

    results_dict = load_results_dict(args.state_path)
    feature_names = results_dict["feature_names"]
    als_model = rating_qii.load_als_model(sc, os.path.join(args.state_path,
                                                           "als_model.pkl"))
    user_features = als_model.userFeatures()
    product_features = als_model.productFeatures()
    rank = als_model.rank

    cur_product_features = rating_qii.get_features_by_id(product_features,
                                                         args.product)
    cur_user_features = rating_qii.get_features_by_id(user_features,
                                                      args.user)

    predicted_product_features = get_predicted_product_features(sc, args.state_path,
                                                                args.product, rank)
    predicted_product_features_list = [x[1] for x in sorted(predicted_product_features.items(),
                                                            key=lambda x: x[0])]

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
