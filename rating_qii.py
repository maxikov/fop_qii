#standard library
import random
import argparse
import os.path

#project files
import common_utils
import CustomFeaturesRecommender

#pyspark library
from pyspark.mllib.recommendation import MatrixFactorizationModel
from pyspark import SparkConf, SparkContext

def load_als_model(sc, fname):
    if "als" in fname:
        if os.path.exists(fname):
            res =  MatrixFactorizationModel.load(sc, fname)
            return res
        else:
            old_suffix = "als_model.pkl"
            new_suffix = "upr_model.pkl"
            fname = fname[:len(old_suffix)] + new_suffix
    res = CustomFeaturesRecommender.load(sc, fname)
    return res

def get_features_by_id(features, _id):
    res = features\
          .lookup(_id)[0]
    return res

def predict_one_rating(cur_user_features, cur_product_features):
    res = sum(x*y for (x, y) in zip(cur_user_features, cur_product_features))
    return res

def rating_qii(user_features, product_features, user_id, product_id,
               user_or_feature_qii, iterations=10):
    res = {}
    rank = len(product_features.values().take(1)[0])
    cur_user_features = get_features_by_id(user_features, user_id)
    cur_product_features = get_features_by_id(product_features, user_id)
    original_rating = predict_one_rating(cur_user_features,
                                         cur_product_features)
    for f in xrange(rank):
        if user_or_feature_qii in ["user", "both"]:
            user_distr = common_utils.get_feature_distribution(user_features,
                                                               f)
        if user_or_feature_qii in ["product", "both"]:
            product_distr = common_utils.get_feature_distribution(product_features,
                                                                  f)
        mean_change = 0.0
        mean_abs_change = 0.0
        for i in xrange(iterations):
            if user_or_feature_qii in ["user", "both"]:
                new_user_features =\
                        common_utils.set_list_value(cur_user_features, f,
                                                    random.choice(user_distr))
            else:
                new_user_features = cur_user_features
            if user_or_feature_qii in ["product", "both"]:
                new_product_features =\
                        common_utils.set_list_value(cur_product_features, f,
                                                    random.choice(product_distr))
            else:
                new_product_features = cur_user_features
            new_result = predict_one_rating(new_user_features,
                                            new_product_features)
            change = new_result - original_rating
            abs_change = abs(change)
            mean_change += change
            mean_abs_change += abs_change
        sign = 1 if mean_change == 0 else mean_change/float(abs(mean_change))
        mean_abs_change *= sign
        mean_abs_change = mean_abs_change / float(iterations)
        res[f] = mean_abs_change
    return res, original_rating

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--als-model", action="store", type=str, help=\
                        "Path from which to load ALS model to analyze")
    parser.add_argument("--user", action="store", type=int, help=\
                        "User ID for whom to compute QII.")
    parser.add_argument("--product", action="store", type=int, help=\
                        "Product ID for which to compute QII.")
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

    model = load_als_model(sc, args.als_model)
    user_features = model.userFeatures()
    product_features = model.productFeatures()

    qiis, original_rating = rating_qii(user_features, product_features, args.user,
                                       args.product, args.user_or_product,
                                       args.qii_iterations)
    print "Original rating:", original_rating
    print "Per feature QIIs:"
    qiis = sorted(qiis.items(), key=lambda x: x[0])
    for f, qii in qiis:
        print f, ":", qii

if __name__ == "__main__":
    main()
