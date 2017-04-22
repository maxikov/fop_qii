#standard library
from operator import add
import time
import random
import bisect

#pyspark library
from pyspark.mllib.evaluation import RegressionMetrics,\
     BinaryClassificationMetrics

#numpy library
import numpy as np

def shift_drop_dict(src, ids_to_drop):
    ids_to_drop = sorted(list(ids_to_drop))
    res = {}
    for key, value in src.items():
        key_offset = bisect.bisect_left(ids_to_drop, key)
        res[key - key_offset] = value
    return res

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

def mean_error(predictionsAndRatings, power, abs_function=abs):
    return (predictionsAndRatings.map(lambda x: abs_function(x[0] - x[1]) **\
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
    predictionsAndRatings = predictions.map(lambda x: ((x[0], x[1]),
        float(x[2]))) \
        .join(baseline_predictions.map(lambda x: ((x[0], x[1]), float(x[2]))))
    predictions = predictionsAndRatings.map(lambda x: (x[0], x[1][0]))
    observations = predictionsAndRatings.map(lambda x: (x[0], x[1][1]))
    return evaluate_regression(predictions, observations, logger, nbins,
                               (0.0, 5.5))

def make_bins(bin_range, nbins):
    return map(float, list(np.linspace(bin_range[0], bin_range[1], nbins+2)))

def evaluate_binary_classifier(predictions, observations, logger,
                               no_threshold=True):
    logger.debug("Evaluating the model")
    predobs = predictions\
            .join(observations)\
            .values()
    n_pos = predobs.filter(lambda x: x[1] == 1).count()
    prate = float(n_pos)/float(predobs.count())
    if no_threshold:
        metrics = BinaryClassificationMetrics(predobs)
        auroc = metrics.areaUnderROC
        aupr = metrics.areaUnderPR
        better = (1.0 - prate)/(1.0 - aupr)
        res = {"auroc": auroc, "auprc": aupr, "prate": prate, "better": better}
    else:
        pass #TODO
    logger.debug("{}".format(res))
    return res


def evaluate_regression(predictions, observations, logger=None, nbins=32,
                        bin_range=None):
    if logger is None:
        print "Evaluating the model"
    else:
        logger.debug("Evaluating the model")
    start = time.time()
    predobs = predictions\
            .join(observations)\
            .values()\
            .map(lambda (a, b): (float(a), float(b)))

    if bin_range is None:
        bin_range = (-1.0, 1.0)
    else:
        bin_range = (float(bin_range[0]), float(bin_range[1]))
    normal_bins = make_bins(bin_range, nbins)
    max_magnitude = max(map(abs, bin_range))
    total_min = min(map(lambda x: -abs(x), bin_range))
    abs_bins = make_bins((0, max_magnitude), nbins)
    err_bins = make_bins(
                         ( total_min,
                           max_magnitude
                         ),
                         nbins)
    sq_bins = make_bins((0, max_magnitude**2), nbins)

    metrics = RegressionMetrics(predobs)
    mrae = mean_relative_absolute_error(predobs)
    mean_abs_err = mean_error(predobs, power=1.0, abs_function=abs)
    mean_err = mean_error(predobs, power=1.0, abs_function=float)

    obs = predobs.map(lambda (_, o): o)
    preds = predobs.map(lambda (p, _): p)
    errors = predobs.map(lambda (p, o): p - o)

    preds_histogram = preds.histogram(normal_bins)
    obs_histogram = obs.histogram(normal_bins)
    errors_histogram = errors.histogram(err_bins)
    abs_errors_histogram = errors\
            .map(lambda x: abs(x))\
            .histogram(abs_bins)
    sq_errors_histogram = errors\
            .map(lambda x: x*x)\
            .histogram(sq_bins)

    logger.debug("Done in %f seconds", time.time() - start)
    logger.debug("Mean error: {}, mean absolute error: {}".\
            format(mean_err, mean_abs_err))
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
           "obs_histogram": obs_histogram,
           "mean_err": mean_err,
           "mean_abs_err": mean_abs_err}
    return res
