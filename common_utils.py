#standard library
from operator import add
import time
import random
import bisect
import os.path
import pickle
import functools

#pyspark library
from pyspark.mllib.evaluation import RegressionMetrics,\
     BinaryClassificationMetrics

#numpy library
import numpy as np

def dict_join(foo, bar, join_f = lambda x, y: (x, y)):
    keys = set(foo.keys()) & set(bar.keys())
    res = {key: join_f(foo[key], bar[key]) for key in keys}
    return res

def drop_rare_movies(features, indicators, nmovies):
    filter_f = functools.partial(
        lambda nmovies, (mid, (ftrs, inds)):
                    sum(1 for x in inds if abs(x) > 0) >= nmovies,
        nmovies)
    joined = features\
            .join(indicators)\
            .filter(filter_f)
    features = joined.map(lambda (mid, (ftrs, inds)): (mid, ftrs))
    indicators = joined.map(lambda (mid, (ftrs, inds)): (mid, inds))
    return features, indicators

def safe_zip(foo, bar):
    foo = foo\
            .zipWithIndex()\
            .map(lambda (x, y): (y, x))
    bar = bar\
            .zipWithIndex()\
            .map(lambda (x, y): (y, x))
    res = foo\
            .join(bar)\
            .values()
    return res

def load_if_available(persist_dir, fname, logger):
    if persist_dir is not None:
        fname = os.path.join(persist_dir, fname)
        if not os.path.isfile(fname):
            logger.debug("%s not found, loading new", fname)
            objects = None
            store = True
        else:
            ifile = open(fname, "rb")
            objects = pickle.load(ifile)
            ifile.close()
            store = False
    else:
        objects = None
        store = False
    return objects, store

def save_if_needed(persist_dir, fname, objects, store, logger):
    if store:
        fname = os.path.join(persist_dir, fname)
        logger.debug("Storing in %s", fname)
        ofile = open(fname, "wb")
        pickle.dump(objects, ofile)
        ofile.close()


def compute_regression_qii(lr_model, input_features, target_variable,
                           logger, original_predictions=None, rank=None,
                           features_to_test=None):
    if logger is not None:
        logger.debug("Measuring model QII")
    if original_predictions is None:
        original_predictions = lr_model\
                               .predict(\
                                   input_features\
                                   .values())\
                               .cache()
    else:
        original_predictions = original_predictions.values().cache()
    if features_to_test is None:
        if rank is None:
            rank = len(input_features.take(1)[0][1])
            if logger is not None:
                logger.debug("%d features detected", rank)
        features_to_test = xrange(rank)
    res = []
    for f in features_to_test:
        if logger is not None:
            logger.debug("Processing feature %d", f)
        perturbed_features = perturb_feature(input_features, f,
                                             None).values()
        new_predictions = lr_model.predict(perturbed_features)
        predobs = safe_zip(new_predictions, original_predictions)
        cur_qii = mean_error(predobs, 1.0, abs)
        signed_error = mean_error(predobs, 1.0, float)
        if signed_error == 0:
            sign = 1
        else:
            sign = signed_error/abs(signed_error)
        signed_cur_qii = cur_qii * sign
        if logger is not None:
            logger.debug("QII: {}, signed QII: {}".format(cur_qii, signed_cur_qii))
        res.append(signed_cur_qii)
    return res

def shift_drop_dict(src, ids_to_drop):
    ids_to_drop = set(ids_to_drop)
    res = {}
    items = src.items()
    offset = 0
    for key, value in items:
        if key in ids_to_drop:
            offset += 1
        else:
            res[key-offset] = value
    return res

def substitute_belong_predicate(line):
    if_pos = line.find("If (")
    else_pos = line.find("Else (")
    if if_pos == -1 and else_pos == -1:
        return line
    if if_pos != -1:
        subline = line[if_pos+len("If ("):]
        before_line = line[:if_pos+len("If (")]
    elif else_pos != -1:
        subline = line[else_pos+len("Else ("):]
        before_line = line[:else_pos+len("Else (")]
    else:
        return line
    in_pos = subline.find(" not in ")
    if in_pos == -1:
        in_pos = subline.find(" in ")
    if in_pos == -1:
        return line
    feature = subline[:in_pos]
    if " not in {0.0}" in subline:
        predicate = "Is "
    elif " not in {1.0}" in subline:
        predicate = "Isn't "
    elif " in {1.0}" in subline:
        predicate = "Is "
    elif " in {0.0}" in subline:
        predicate = "Isn't "
    else:
        return line
    res = before_line + predicate + feature + ")"
    return res

def substitute_feature_names(string, feature_names):
    for fid, fname in feature_names.items():
        fname = fname.decode("ascii", errors="ignore")
        string = string.replace("feature {} ".format(fid),
                                "{} ".format(fname))
    string = "\n".join(substitute_belong_predicate(line) for line in\
            string.split("\n"))
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
    map_f = functools.partial(lambda abs_function, power, x:
            abs_function(x[0] - x[1]) ** power,
            abs_function, power)
    return (predictionsAndRatings.map(map_f).reduce(add) / float(predictionsAndRatings.count())) ** (1.0/power)

def manual_diff_models(model_1, model_2, user_product_pairs, power=1.0):
    predictions_1 = manual_predict_all(user_product_pairs, *model_1)
    predictions_2 = manual_predict_all(user_product_pairs, *model_2)
    predictionsAndRatings = predictions_1.map(lambda x: ((x[0], x[1]), x[2])) \
      .join(predictions_2.map(lambda x: ((x[0], x[1]), x[2]))) \
      .values()
    return mean_error(predictionsAndRatings, power)

def set_list_value(lst, ind, val, logger=None):
    lst = list(lst)
    new_lst = lst[:ind] + [val] + lst[ind+1:]
    if logger is not None:
        print "set_list_value(lst={}, ind={}, val={})\nOld list: {}\nNew list:{}".\
                format(lst, ind, val, lst, new_lst)
    return new_lst

def set_list_values(*args):
    return set_list_value(*args) #OOPS

def get_feature_distribution(features, f):
    map_f = functools.partial(lambda f, (_, arr):
            arr[f], f)
    res = features.map(map_f).collect()
    return res

def perturb_feature(features, f, perturbed_subset=None):
    if perturbed_subset is not None:
        filter_f = functools.partial(lambda perturbed_subset, x:
                x[0] not in perturbed_subset, perturbed_subset)
        features_intact = features.filter(filter_f)
        filter_f = functools.partial(lambda perturbed_subset, x:
                x[0] in perturbed_subset, perturbed_subset)
        features_perturbed = features.filter(filter_f)
        features = features_perturbed
    dist = get_feature_distribution(features, f)
    random.shuffle(dist)
    all_xs = features.keys().collect()
    ddist = {k: v for (k, v) in zip(all_xs, dist)}
    map_f = functools.partial(lambda f, ddist, (x, arr):
            (x, set_list_value(arr, f, ddist[x])),
            f, ddist)
    res = features.map(map_f)
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

def feature_ranges(features, logger):
    logger.debug("Computing feature ranges")
    res = {}
    rank = len(features.take(1)[0][1])
    logger.debug("Detected %d features", rank)
    maxes = features\
       .values()\
       .reduce(lambda x, y: (map(max, zip(x, y))))
    mins = features\
       .values()\
       .reduce(lambda x, y: (map(min, zip(x, y))))
    for f in xrange(rank):
        _min = mins[f]
        _max = maxes[f]
        res[f] = {"min": _min, "max": _max}
    logger.debug("Feature ranges: {}".format(res))
    return res

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
                         nbins=32, model_name=""):
    predictionsAndRatings = predictions.map(lambda x: ((x[0], x[1]),
        float(x[2]))) \
        .join(baseline_predictions.map(lambda x: ((x[0], x[1]), float(x[2]))))
    predictions = predictionsAndRatings.map(lambda x: (x[0], x[1][0]))
    observations = predictionsAndRatings.map(lambda x: (x[0], x[1][1]))
    return evaluate_regression(predictions, observations, logger, nbins,
                               (0.0, 5.5), model_name)

def make_bins(bin_range, nbins):
    return map(float, list(np.linspace(bin_range[0], bin_range[1], nbins+2)))

def evaluate_binary_classifier(predictions, observations, logger,
                               no_threshold=True):
    logger.debug("Evaluating the model, no_threshold: {}".\
            format(no_threshold))
    predobs = predictions\
            .join(observations)\
            .values()
    n_pos = predobs.filter(lambda x: int(x[1]) == 1).count()
    prate = float(n_pos)/float(predobs.count())
    if no_threshold:
        metrics = BinaryClassificationMetrics(predobs)
        auroc = metrics.areaUnderROC
        aupr = metrics.areaUnderPR
        if aupr == 1:
            better = 0
        else:
            better = (1.0 - prate)/(1.0 - aupr)
        res = {"auroc": auroc, "auprc": aupr, "prate": prate, "better": better}
    else:
        relevants = predobs.filter(lambda x: int(x[1]) == 1)
        true_positives = predobs.filter(lambda x: int(x[0]) == 1\
                and int(x[1]) == 1)
        all_positives = predobs.filter(lambda x: int(x[0]) == 1)
        rcount = float(relevants.count())
        tpcount = float(true_positives.count())
        apcount = float(all_positives.count())
        if rcount != 0:
            recall = tpcount/rcount
        else:
            recall = 0
        if apcount != 0:
            precision = tpcount/apcount
        else:
            precision = 0
        res = {"recall": recall, "precision": precision}
    logger.debug("{}".format(res))
    return res


def evaluate_regression(predictions, observations, logger=None, nbins=32,
                        bin_range=None, model_name=""):
    logger.debug("{} Evaluating the model".format(model_name))
    start = time.time()
    predobs = predictions\
            .join(observations)\
            .values()\
            .map(lambda (a, b): (float(a), float(b)))\
            .cache()

    if bin_range is None:
        _min = min(predictions.values().min(), observations.values().min())
        _max = max(predictions.values().max(), observations.values().max())
        bin_range = (_min, _max)
    else:
        bin_range = (float(bin_range[0]), float(bin_range[1]))
    logger.debug("{} Bin range: {}".format(model_name, bin_range))
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

    obs = predobs.map(lambda (_, o): o).cache()
    preds = predobs.map(lambda (p, _): p).cache()
    errors = predobs.map(lambda (p, o): p - o).cache()

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
    logger.debug("{} Mean error: {}, mean absolute error: {}".\
            format(model_name, mean_err, mean_abs_err))
    logger.debug("{} RMSE: {}, variance explained: {}, mean absolute error: {}, r2: {}".\
                 format(model_name,
                        metrics.rootMeanSquaredError,# these are switched?
                        metrics.explainedVariance,   #
                        metrics.meanAbsoluteError,
                        metrics.r2))

    logger.debug("{} MRAE: {}".format(model_name, mrae))
    logger.debug("{} Errors histogram: {}".format(model_name, errors_histogram))
    logger.debug("{} Absolute errors histogram: {}".format(model_name, abs_errors_histogram))
    logger.debug("{} Squared errors histogram: {}:".format(model_name, sq_errors_histogram))
    logger.debug("{} Predictions histogram: {}".format(model_name, preds_histogram))
    logger.debug("{} Observations histogram: {}".format(model_name, obs_histogram))
    res = {"mre": metrics.meanAbsoluteError,
           "rmse": metrics.rootMeanSquaredError,
           "r2": metrics.r2,
           "explained_variance": metrics.explainedVariance,
           "mrae": mrae,
           "errors_histogram": errors_histogram,
           "abs_errors_histogram": abs_errors_histogram,
           "sq_errors_histogram": sq_errors_histogram,
           "preds_histogram": preds_histogram,
           "obs_histogram": obs_histogram,
           "mean_err": mean_err,
           "mean_abs_err": mean_abs_err}
    return res
