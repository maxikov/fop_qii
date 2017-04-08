#standard library
from operator import add
import time
import random

#pyspark library
from pyspark.mllib.evaluation import RegressionMetrics

#numpy library
import numpy as np

def substitute_feature_names(string, feature_names):
    for fid, fname in feature_names.items():
        fname = fname.decode("ascii", errors="ignore")
        string = string.replace("feature {} ".format(fid),
                                "{} ".format(fname))
    return string

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
                            (user, product, sum(map(lambda (x, y): x*y,\
                                zip(user_features[user],\
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

def perturb_feature(features, f, perturbed_subset=None):
    if perturbed_subset is not None:
        features_intact = features.filter(lambda x: x[0] not in
                perturbed_subset)
        features_perturbed = features.filter(lambda x: x[0] in
                perturbed_subset)
        features = features_perturbed
    dist = get_feature_distribution(features, f)
    res = features.map(lambda (x, arr):\
            (x, set_list_value(arr, f, random.choice(dist))))
    if perturbed_subset is not None:
        res = res.union(features_intact)
    return res

def mean_feature_values(features, logger):
    logger.debug("Computing mean feature values")
    start = time.time()
    mean_p_feat_vals = features\
       .values()\
       .reduce(lambda x, y: (map(sum, zip(x, y))))
    nmovies = float(features.count())
    mpfv = {x: mean_p_feat_vals[x]/nmovies for x in xrange(len(mean_p_feat_vals))}
    logger.debug("Done in %f seconds", time.time() - start)
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

def evaluate_recommender(baseline_predictions, predictions, logger=None,
                         nbins=32):
    nbins = list(np.linspace(1.0, 5.0, nbins+1))
    predictionsAndRatings = predictions.map(lambda x: ((x[0], x[1]), x[2])) \
        .join(baseline_predictions.map(lambda x: ((x[0], x[1]), x[2])))
    predictions = predictionsAndRatings.map(lambda x: (x[0], x[1][0]))
    observations = predictionsAndRatings.map(lambda x: (x[0], x[1][1]))
    return evaluate_regression(predictions, observations, logger, nbins)

def evaluate_regression(predictions, observations, logger=None, nbins=32):
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
    obs = predobs.map(lambda (_, o): o)
    preds = predobs.map(lambda (p, _): p)
    preds_histogram = preds.histogram(nbins)
    obs_histogram = obs.histogram(nbins)
    errors = predobs.map(lambda (p, o): p - o)
    errors_histogram = errors.histogram(nbins)
    abs_errors_histogram = errors\
            .map(lambda x: abs(x))\
            .histogram(nbins)
    sq_errors_histogram = errors\
            .map(lambda x: x*x)\
            .histogram(nbins)
    logger.debug("Done in %f seconds", time.time() - start)
    logger.debug("RMSE: {}, variance explained: {}, mean absolute error: {},".\
        format(metrics.explainedVariance,\
               metrics.rootMeanSquaredError,
            metrics.meanAbsoluteError))
    logger.debug("MRAE: {}".format(mrae))
    logger.debug("Errors histogram: {}".format(errors_histogram))
    logger.debug("Absolute errors histogram: {}".format(abs_errors_histogram))
    logger.debug("Squared errors histogram: {}:".format(sq_errors_histogram))
    logger.debug("Predictions histogram: {}".format(preds_histogram))
    logger.debug("Observations histogram: {}".format(obs_histogram))
    res = {"mre": metrics.meanAbsoluteError,
           "mrae": mrae,
           "errors_histogram": errors_histogram,
           "abs_errors_histogram": abs_errors_histogram,
           "sq_errors_histogram": sq_errors_histogram,
           "preds_histogram": preds_histogram,
           "obs_histogram": obs_histogram}
    return res
