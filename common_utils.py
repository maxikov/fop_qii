#standard library
from operator import add
import time
import random

#pyspark library
from pyspark.mllib.evaluation import RegressionMetrics

def recommender_mean_error(model, data, power=1.0):
    predictions = model.predictAll(data.map(lambda x: (x[0], x[1])))
    predictionsAndRatings = predictions.map(lambda x: ((x[0], x[1]), x[2])) \
      .join(data.map(lambda x: ((x[0], x[1]), x[2]))) \
      .values()
    return mean_error(predictionsAndRatings, power)

def manual_predict_all(data, user_features, product_features):
    user_products = data.map(lambda x: (x[0], x[1]))
    user_features = dict(user_features.collect())
    product_features = dict(product_features.collect())
    res = user_products.map(lambda (user, product):
            (user, product, sum(map(lambda (x, y): x*y,
                zip(user_features[user],
                    product_features[product])))))
    return res

def manual_recommender_mean_error(user_features, product_features, data, power=1.0):
    predictions = manual_predict_all(data, user_features, product_features)
    predictionsAndRatings = predictions.map(lambda x: ((x[0], x[1]), x[2])) \
      .join(data.map(lambda x: ((x[0], x[1]), x[2]))) \
      .values()
    return mean_error(predictionsAndRatings, power)

def mean_error(predictionsAndRatings, power):
    return (predictionsAndRatings.map(lambda x: abs(x[0] - x[1]) **\
        power).reduce(add) / float(predictionsAndRatings.count())) ** (1.0/power)

def manual_diff_models(model_1, model_2, user_product_pairs, power=1.0):
    predictions_1 = manual_predict_all(user_product_pairs, *model_1)
    predictions_2 = manual_predict_all(user_product_pairs, *model_2)
    predictionsAndRatings = predictions_1.map(lambda x: ((x[0], x[1]), x[2])) \
      .join(predictions_2.map(lambda x: ((x[0], x[1]), x[2]))) \
      .values()
    return mean_error(predictionsAndRatings, power)

def set_list_value(lst, ind, val):
    new_lst = list(lst)
    new_lst[ind] = val
    return new_lst

def set_list_values(*args):
    return set_list_value(*args) #OOPS

def get_feature_distribution(features, f):
    res = features.map(lambda (_, arr): arr[f]).collect()
    return res

def perturb_feature(features, f):
    dist = get_feature_distribution(features, f)
    res = features.map(lambda (x, arr):
            (x, set_list_value(arr, f, random.choice(dist))))
    return res

def mean_feature_values(features, logger):
        logger.debug("Computing mean feature values")
        start = time.time()
        mean_p_feat_vals = features\
                .values()\
                .reduce(lambda x, y: (map(sum, zip(x, y))))
        nmovies = float(features.count())
        mpfv = {x: mean_p_feat_vals[x]/nmovies for x in xrange(len(mean_p_feat_vals))}
        logger.debug("Done in {} seconds".format(time.time() - start))
        logger.debug("Mean product feature values: {}".format(mpfv))
        return mpfv

def mean_relative_absolute_error(predobs):
    """
    Compute mean relative absolute error of regression.

    Parameters:
        predobs (mandatory) - RDD of (prediction, observation) pairs.

    Returns:
        MRAE (float)
    """
    res = predobs.\
        map(lambda (pred, obs):
            abs(pred-obs)/float(abs(obs if obs != 0 else 1))).\
                sum()/predobs.count()
    return res

def evaluate_regression(predictions, observations, logger = None):
    if logger is None:
        print "Evaluating the model"
    else:
        logger.debug("Evaluating the model")
    start = time.time()
    predobs = predictions\
            .join(observations)\
            .values()
    metrics = RegressionMetrics(predobs)
    mrae = mean_relative_absolute_error(predobs)
    if logger is None:
        print "Done in {} seconds".format(time.time() - start)
        print "RMSE: {}, variance explained: {}, mean absolute error: {},".\
                    format(metrics.explainedVariance,\
                    metrics.rootMeanSquaredError,
                    metrics.meanAbsoluteError)
        print "MRAE: {}".format(mrae)
    else:
        logger.debug("Done in {} seconds".format(time.time() - start))
        logger.debug("RMSE: {}, variance explained: {}, mean absolute error: {},".\
                    format(metrics.explainedVariance,\
                    metrics.rootMeanSquaredError,
                    metrics.meanAbsoluteError))
        logger.debug("MRAE: {}".format(mrae))
    res = {"mre": metrics.meanAbsoluteError, "mrae": mrae}
    return res
