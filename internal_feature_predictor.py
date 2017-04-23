#standard library
import time
import math
import random
from collections import defaultdict

#prettytable library
from prettytable import PrettyTable

#pyspark library
from pyspark.mllib.recommendation import ALS
import pyspark.mllib.recommendation
from pyspark.mllib.classification import LabeledPoint, NaiveBayes
import pyspark.mllib.regression
import pyspark.mllib.tree
from pyspark.mllib.regression import LinearRegressionWithSGD

#numpy library
import numpy as np

#project files
import AverageRatingRecommender
import RandomizedRecommender
import parsers_and_loaders
import common_utils
import TrimmedFeatureRecommender
import HashTableRegression

regression_models = ["regression_tree", "random_forest", "linear",
                     "naive_bayes", "logistic"]
metadata_sources = ["name", "genres", "tags", "imdb_keywords",
                    "imdb_genres", "imdb_director", "imdb_producer",
                    "tvtropes", "average_rating", "users", "years"]

def discretize_single_feature(data, nbins, logger):
    """
    data: RDD of ID: value
    """
    cur_f_values = data.values()
    _max = cur_f_values.max()
    _min = cur_f_values.min()
    logger.debug("Range from {} to {}".format(_min, _max))
    bin_step = (_max - _min)/float(nbins)
    logger.debug("{} bins, step: {}".format(nbins, bin_step))
    bin_edges = np.linspace(_min, _max, nbins+1)
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2.0
    bin_centers = np.insert(bin_centers,
                            [0, len(bin_centers)],
                            [bin_centers[0]-bin_step,
                                bin_centers[-1]+bin_step])
    data = data.map(lambda (mid, val):
                           (mid, int(np.digitize(val, bin_edges))))
    return data, bin_centers

def undiscretize_signle_feature(data, bin_centers):
    """
    data: RDD of ID: bin_n
    """
    data = data.map(lambda (mid, bin_n):
                           (mid, float(bin_centers[bin_n])))
    return data

def discretize_features(data, nbins, dont_discretize, logger):
    """
    data: RDD of ID: [values]
    dont_discretize: set of feature IDs to be left intact
    """
    logger.debug("Discretizing features")
    start = time.time()
    rank = len(data.take(1)[0][1])
    logger.debug("{} features found".format(rank))
    all_bin_centers = {}
    for f in set(xrange(rank)) - dont_discretize:
        logger.debug("Processing feature {}".format(f))
        cur_f_data = data.map(lambda (mid, ftrs):
                                     (mid, ftrs[f]))
        cur_f_data, bin_centers = discretize_single_feature(cur_f_data,
                                                            nbins,
                                                            logger)
        data = data\
               .join(cur_f_data)\
               .map(lambda (mid, (ftrs, new_f_val)):
                           (mid, common_utils.set_list_value(ftrs, f,
                                                             new_f_val)))
        all_bin_centers[f] = bin_centers
    logger.debug("Done in %f seconds", time.time() - start)
    return data, all_bin_centers

def train_regression_model(data, regression_model="regression_tree",
                           categorical_features={}, max_bins=32, max_depth=None,
                           num_trees=128, logger=None, is_classifier=False,
                           num_classes=None):
    """
    Train a regression model on the input data.

    Parameters:

        data (mandatory) - a training set RDD of pyspark.mllib.classification.LabeledPoint.

        regression_model (optional) - string, which model to train. Possible
            values:
                "regression_tree"
                "random_forest"
                "linear"
            Default:
                "regression_tree"

        categorical_features (optional) - dict,
            {feature_index: number_of_categories}. For "regression_tree" and
            "random_forest" models, contains information about which of the
            input features should be treated as taking only a fixed set of
            possible values with no meaningful order. Otherwise, all features
            are considered to be ordered.
            Default:
                {}

        max_bins (optional) - int, the maximum number of bins into which the
            target variable is split for "regression_tree" and "random_forest".
            Default:
                32

        max_depth (optional) - int, maximum depth of the tree.
            Default: int(math.ceil(math.log(max_bins, 2)))

        num_trees (optional) - int, number of trees in the random forest.
            Default:
                128

        logger (optional) - logging.Logger, the object into which debug
            information will be send. If None, printing to stdout will be used.
            Default:
                None

    Returns:
        trained_regression_model
    """

    if num_classes is None:
        num_classes = max_bins

    if max_depth is None:
        max_depth = int(math.ceil(math.log(max_bins, 2)))

    start = time.time()

    if logger is None:
        print "Training", regression_model
    else:
        logger.debug("Training " + regression_model)
    if is_classifier:
        logger.debug("Actually training a classifier")

    if regression_model == "regression_tree":
        if is_classifier:
            lr_model = pyspark.\
                    mllib.\
                    tree.\
                    DecisionTree.\
                    trainClassifier(
                        data,
                        numClasses=num_classes,
                        categoricalFeaturesInfo=categorical_features,
                        impurity="gini",
                        maxDepth=max_depth,
                        maxBins=max_bins)
        else:
            lr_model = pyspark.\
                    mllib.\
                    tree.\
                    DecisionTree.\
                    trainRegressor(
                        data,
                        categoricalFeaturesInfo=categorical_features,
                        impurity="variance",
                        maxDepth=max_depth,
                        maxBins=max_bins)

    elif regression_model == "random_forest":
        if is_classifier:
            lr_model = pyspark.\
                    mllib.\
                    tree.\
                    RandomForest.\
                    trainClassifier(
                        data,
                        numClasses=num_classes,
                        categoricalFeaturesInfo=categorical_features,
                        impurity="gini",
                        maxDepth=max_depth,
                        maxBins=max_bins,
                        numTrees=num_trees)
        else:
            lr_model = pyspark.\
                    mllib.\
                    tree.\
                    RandomForest.\
                    trainRegressor(
                        data,
                        categoricalFeaturesInfo=categorical_features,
                        numTrees=num_trees,
                        maxDepth=max_depth,
                        maxBins=max_bins)
    elif regression_model == "linear":
        lr_model = LinearRegressionWithSGD.train(data)
    elif regression_model == "naive_bayes":
        lr_model = NaiveBayes.train(data)
    elif regression_model == "hash_table":
        lr_model = HashTableRegression.train(data)
    elif regression_model == "logistic":
        lr_model = pyspark.\
                   mllib.\
                   classification.\
                   LogisticRegressionWithLBFGS.\
                   train(data)



    if logger is None:
        print "Done in {} seconds".format(time.time() - start)
    else:
        logger.debug("Done in {} seconds".format(time.time() - start))

    return lr_model


def build_meta_data_set(sc, sources, all_ids=None, logger=None):
    """
    Load specified types of metadata for users or products, and turn them into
    an RDD of
        ID: [features]
    and dict of categorical feature info (see documentation for
    train_regression_model for details).

    Parameters:

        sc (mandatory) - SparkContext

        sources (mandatory) - list of dicts:
            [
                {
                    name: str, name of the feature(s) being loaded,
                    src_rdd: a function that takes no arguments and returns
                        RDD of parseable strings,
                    loader: function that takes src_rdd and returns
                        (RDD(ID: [features],
                        number_of_features,
                        categorical_feature_info
                        )
                }
            ]

        all_ids (optional) - set of ints, all user and product IDs for which
            features should be loaded. If a certain source of data is missing
            records for some of the passed IDs, empty records of matching
            dimensionality will be added. If None, it's assumed that all
            sources will have the same sets of IDs.
            Default:
                None

        logger (optional) - logging.Logger, the object into which debug
            information will be send. If None, printing to stdout will be used.
            Default:
                None

    Returns:
        (RDD (ID: [features]), number_of_features, categorical_features_info)
    """

    feature_offset = 0
    categorical_features_info = {}
    feature_names = {}
    res_rdd = None

    for source in sources:
        logger.debug("Loading " + source["name"])

        start = time.time()

        cur_rdd, nof, cfi, fnames = source["loader"](source["src_rdd"]())
        rdd_count = cur_rdd.count()

        logger.debug("Done in {} seconds".format(time.time() - start))
        logger.debug(str(rdd_count) + " records of " +\
                    str(nof) + " features loaded")

        if all_ids is not None:
            cur_ids = set(cur_rdd.keys().collect())
            missing_ids = all_ids - cur_ids
            if len(missing_ids) == 0:
                logger.debug("No missing IDs")
            else:
                logger.debug(str(len(missing_ids)) + " IDs are missing. "+\
                            "Adding empty records for them")

                start = time.time()

                empty_records = [(_id, [0 for _ in xrange(nof)]) for _id in
                                 missing_ids]
                empty_records = sc.parallelize(empty_records)
                cur_rdd = cur_rdd.union(empty_records).cache()
                logger.debug("Done in {} seconds".format(time.time() - start))

        shifted_cfi = {f+feature_offset: v for (f, v) in cfi.items()}
        shifted_fnames = {f+feature_offset: v for (f, v) in fnames.items()}
        categorical_features_info = dict(categorical_features_info,
                                         **shifted_cfi)
        feature_names = dict(feature_names,
                             **shifted_fnames)

        if res_rdd is None:
            res_rdd = cur_rdd
        else:
            res_rdd = res_rdd\
                    .join(cur_rdd)\
                    .map(lambda (x, (y, z)): (x, y+z))\
                    .cache()
        feature_offset += nof
    return (res_rdd, feature_offset, categorical_features_info, feature_names)

def drop_rare_features(indicators, nof, categorical_features, feature_names,
                       drop_threshold, logger):
    logger.debug("Dropping features with less than %d" +\
                 " non-zero values", drop_threshold)
    start = time.time()
    feature_counts = indicators\
                     .map(lambda (mid, ind_lst):
                             (mid, [1 if x > 0 else 0 for x in ind_lst]))\
                     .values()\
                     .reduce(lambda a, b: map(sum, zip(a, b)))
    features_to_drop = set(x[0] for x in enumerate(feature_counts) if x[1] <=
                           drop_threshold)
    logger.debug("Dropping %d features", len(features_to_drop))
    res_rdd = indicators.map(lambda (mid, ind_list): (mid,
        [x[1] for x in enumerate(ind_list) if x[0] not in features_to_drop]))
    res_rdd = res_rdd.cache()
    features_to_drop = sorted(list(features_to_drop))
    #logger.debug("Old catf: {}".format(sorted(categorical_features.items(),
    #    key=lambda x: x[0])))
    categorical_features = common_utils.shift_drop_dict(categorical_features,
                                                        features_to_drop)
    #logger.debug("New catf: {}".format(sorted(categorical_features.items(),
    #    key=lambda x: x[0])))
    #logger.debug("Old fn: {}".format(sorted(feature_names.items(),
    #    key=lambda x: x[0])))
    feature_names = common_utils.shift_drop_dict(feature_names,
                                                 features_to_drop)
    #logger.debug("New fn: {}".format(sorted(feature_names.items(),
    #    key=lambda x: x[0])))
    nof = nof - len(features_to_drop)
    logger.debug("%d features remaining", nof)
    logger.debug("Done in %f seconds", time.time() - start)
    return (res_rdd, nof, categorical_features, feature_names)

def predict_internal_feature(features, indicators, f, regression_model,
                             categorical_features, max_bins, logger=None,
                             no_threshold=False, is_classifier=False,
                             num_classes=None):
    """
    Predict the values of an internal feature based on metadata.

    Parameters:

        features (mandatory) - RDD of (ID: [float]) IDs and internal features.

        indicators (mandatory) - RDD of (ID: [float]) IDs and metadata.

        f (mandatory) - int, feature index to predict.

        regression_model (mandatory) - string.

        categorical_features (mandatory) - dict.

        max_bins (mandatory) - int.

        logger (optional) - logging.Logger, the object into which debug
            information will be send. If None, printing to stdout will be used.
            Default:
                None

    For more parameter descriptions see documentation for
    train_regression_model.

    Returns:
        (model - the model created,
        observations - RDD of (ID: float) true feature values,
        predictions - RDD of (ID: float) predicted values)
    """

    if logger is None:
        print "Processing feature {}".format(f)
        print "Building data set"
    else:
        logger.debug("Processing feature {}".format(f))
        logger.debug("Building data set")

    start = time.time()

    joined = features.join(indicators)
    data = joined.map(
        lambda (_id, (ftrs, inds)):
        LabeledPoint(float(ftrs[f]), inds))\
                .cache()
    ids = joined.map(lambda (_id, _): _id)

    if logger is None:
        print "Done in {} seconds".format(time.time() - start)
    else:
        logger.debug("Done in {} seconds".format(time.time() - start))

    lr_model = train_regression_model(data,
                                      regression_model=regression_model,
                                      categorical_features=categorical_features,
                                      max_bins=max_bins,
                                      logger=logger,
                                      is_classifier=is_classifier,
                                      num_classes=num_classes)

    if no_threshold and regression_model == "logistic":
        lr_model.clearThreshold()

    observations = ids.zip(data.map(lambda x: float(x.label))).cache()
    predictions = ids.zip(
        lr_model\
        .predict(
            data.map(lambda x: x.features)
            )\
        .map(float)
        ).cache()

    return (lr_model, observations, predictions)


def replace_feature_with_predicted(features, f, predictions, logger):
    logger.debug("Replacing original feature "+\
            "{} with predicted values".format(f))
    start = time.time()
    features_ids = set(features.keys().collect())
    predictions_ids = set(predictions.keys().collect())
    not_predicted_ids = features_ids - predictions_ids
    features_intact = features.filter(lambda x: x[0] in not_predicted_ids)
    joined = features.join(predictions)
    replaced = joined.map(lambda (mid, (feats, pred)):\
            (mid, common_utils.set_list_value(feats, f, pred)))
    replaced = replaced.union(features_intact).cache()
    logger.debug("Done in {} seconds".format(time.time() - start))
    return replaced

def compare_baseline_to_replaced(baseline_predictions, uf, pf, logger, power):
    start = time.time()
    replaced_predictions = common_utils.manual_predict_all(\
        baseline_predictions.map(lambda x: (x[0], x[1])), uf, pf)
    logger.debug("Done in %f seconds", time.time() - start)
    logger.debug("Computing replaced mean error relative to the baseline model")
    start = time.time()
    predictionsAndRatings = replaced_predictions.map(lambda x: ((x[0], x[1]), x[2])) \
        .join(baseline_predictions.map(lambda x: ((x[0], x[1]), x[2]))) \
        .values().cache()
    replaced_mean_error_baseline = common_utils.mean_error(predictionsAndRatings, power)
    logger.debug("Done in %f seconds", time.time() - start)
    return replaced_mean_error_baseline, replaced_predictions

def compare_with_all_replaced_features(features, other_features,
                                       user_or_product_features,
                                       all_predicted_features, rank,
                                       baseline_predictions, logger, power,
                                       args):
    logger.debug("Computing predictions of the model with all "+\
        "replaced features")
    start = time.time()
    for f in xrange(rank):
        logger.debug("Replacing feature %d", f)
        features = replace_feature_with_predicted(features, f,
                                                   all_predicted_features[f],
                                                   logger)

    if user_or_product_features == "product":
        uf, pf = other_features, features
    else:
        uf, pf = features, other_features

    replaced_mean_error_baseline, replaced_predictions = compare_baseline_to_replaced(\
                        baseline_predictions, uf, pf, logger, power)

    logger.debug("Replaced mean error baseline: "+\
                 "%f", replaced_mean_error_baseline)

    logger.debug("Evaluating replaced model")
    replaced_rec_eval =\
                    common_utils.evaluate_recommender(baseline_predictions,\
                        replaced_predictions, logger, args.nbins)
    logger.debug("Done in %f seconds", time.time() - start)
    return replaced_mean_error_baseline, replaced_rec_eval

def compare_with_all_randomized(baseline_model, rank, perturbed_subset,
                                baseline_predictions, logger, power,
                                args):
    logger.debug("Evaluating a completely randomized model")
    start = time.time()
    model = RandomizedRecommender.RandomizedRecommender(\
        baseline_model, rank, perturbed_subset, logger)
    model.randomize()
    randomized_predictions = model.predictAll(baseline_predictions)
    replaced_rec_eval =\
                    common_utils.evaluate_recommender(baseline_predictions,\
                        randomized_predictions, logger, args.nbins)
    logger.debug("Done in %f seconds", time.time() - start)
    return replaced_rec_eval

def measure_associativity(input_features, target_features, f, logger):
    logger.debug("Measuring associativity")
    start = time.time()
    joined = input_features\
             .join(target_features)\
             .map(lambda (mid, (inds, ftrs)): (inds, ftrs[f]))
    res = defaultdict(set)
    for inds, ftr in joined.collect():
        inds = tuple(inds)
        res[inds].add(ftr)
    avg_card = float(
                    sum(
                        (0.0 if len(x) <= 1 else np.var(list(x))) for x in res.values()
                       )
                    )/len(res)
    logger.debug("Done in %f seconds", time.time() - start)
    return avg_card

def internal_feature_predictor(sc, training, rank, numIter, lmbda,
                               args, all_movies, metadata_sources,
                               user_or_product_features, eval_regression,
                               compare_with_replaced_feature,
                               compare_with_randomized_feature, logger,
                               train_ratio = 0):

    results = {}
    power = 1.0

    if "average_rating" in args.metadata_sources:
        arc = AverageRatingRecommender.AverageRatingRecommender(logger)
        arc.train(training)
        arc_ratings = sc.parallelize(arc.ratings.items())
        metadata_sources.append(
            {"name": "average_rating",
             "src_rdd": lambda: arc_ratings,
             "loader": parsers_and_loaders.load_average_ratings})

    cur_mtdt_srcs = filter(lambda x: x["name"] in args.metadata_sources, metadata_sources)
    if args.drop_missing_movies:
        indicators, nof, categorical_features, feature_names =\
            build_meta_data_set(sc, cur_mtdt_srcs, None, logger)
        old_all_movies = all_movies
        all_movies = set(indicators.keys().collect())
        logger.debug("{} movies loaded, data for {} is missing, purging them"\
                .format(len(all_movies), len(old_all_movies - all_movies)))
        training = training.filter(lambda x: x[1] in all_movies)
        logger.debug("{} items left in the training set"\
                .format(training.count()))
    else:
        indicators, nof, categorical_features, feature_names =\
            build_meta_data_set(sc, cur_mtdt_srcs, all_movies, logger)
    logger.debug("%d features loaded", nof)

    if args.drop_rare_features > 0:
        indicators, nof, categorical_features, feature_names =\
            drop_rare_features(indicators, nof, categorical_features,
                               feature_names, args.drop_rare_features,
                               logger)

    logger.debug("Training ALS recommender")
    start = time.time()
    model = ALS.train(training, rank=rank, iterations=numIter,
                      lambda_=lmbda, nonnegative=args.non_negative)
    logger.debug("Done in %f seconds", time.time() - start)

    logger.debug("Fetching all products from the training set")
    start = time.time()
    training_set_products = set(training.map(lambda x: x[1]).collect())
    logger.debug("Done in %f seconds", time.time() - start)
    logger.debug("%d products collected", len(training_set_products))

    logger.debug("Fetching all products in model")
    start = time.time()
    model_products = set(model.productFeatures().keys().collect())
    logger.debug("Done in %f seconds", time.time() - start)
    logger.debug("%d products collected", len(model_products))
    logger.debug("%d products are missing",
                 len(training_set_products - model_products))

    if compare_with_replaced_feature or compare_with_randomized_feature or\
            args.features_trim_percentile:
        logger.debug("Computing model predictions")
        start = time.time()
        baseline_predictions = model.predictAll(training.map(lambda x:\
            (x[0], x[1]))).cache()
        logger.debug("Done in %f seconds", time.time() - start)

        logger.debug("Computing mean error")
        start = time.time()
        predictionsAndRatings = baseline_predictions\
            .map(lambda x: ((x[0], x[1]), x[2])) \
            .join(training.map(lambda x: ((x[0], x[1]), x[2]))) \
            .values().cache()
        baseline_mean_error = common_utils.mean_error(predictionsAndRatings,
                                                      power)
        baseline_rmse = common_utils.mean_error(predictionsAndRatings, power=2.0)
        logger.debug("Done in %f seconds", time.time() - start)
        logger.debug("Mean error: {}, RMSE: {}".format(baseline_mean_error,
                                                       baseline_rmse))
        results["baseline_mean_error"] = baseline_mean_error
        results["baseline_rmse"] = baseline_rmse

        baseline_rec_eval = common_utils.evaluate_recommender(\
               training, baseline_predictions, logger, args.nbins)
        results["baseline_rec_eval"] = baseline_rec_eval

    if args.features_trim_percentile:
        old_model = model
        old_productFeatures = old_model.productFeatures()
        old_userFeatures = old_model.userFeatures()
        model = TrimmedFeatureRecommender.TrimmedFeatureRecommender(\
                rank, old_userFeatures, old_productFeatures,\
                args.features_trim_percentile, logger).train()
        logger.debug("Computing trimmed predictions")
        trimmed_predictions = model.predictAll(training)
        results["trimmed_rec_eval"] = common_utils.evaluate_recommender(\
                baseline_predictions, trimmed_predictions, logger, args.nbins)

    features = model.productFeatures()
    other_features = model.userFeatures()
    if user_or_product_features == "user":
        features, other_features = other_features, features
    if args.regression_model == "naive_bayes":
        features, feature_bin_centers = discretize_features(features,
                                                            args.nbins,
                                                            set([]),
                                                            logger)
        indicators, _ = discretize_features(indicators, args.nbins,
                                            set(categorical_features.keys()),
                                            logger)

    results["mean_feature_values"] = common_utils.mean_feature_values(features, logger)

    results["features"] = {}

    results["train_ratio"] = train_ratio

    all_predicted_features = {}
    all_predicted_features_test = {}

    for f in xrange(rank):
        results["features"][f] = {}
        if train_ratio > 0:
            logger.debug("Building training and test sets for regression")
            all_movies = features.keys().collect()
            n_movies = len(all_movies)
            training_size = int(math.floor(n_movies * train_ratio / 100.0)) + 1
            logger.debug("{} items in training and {} in test set"\
                    .format(training_size, n_movies - training_size))
            random.shuffle(all_movies)
            training_movies = set(all_movies[:training_size])
            test_movies = set(all_movies[training_size:])
            features_training = features.filter(lambda x: x[0] in
                    training_movies).cache()
            features_test = features.filter(lambda x: x[0] in
                    test_movies).cache()
            indicators_training = indicators.filter(lambda x: x[0] in
                    training_movies).cache()
            indicators_test = indicators.filter(lambda x: x[0] in
                    test_movies).cache()
        else:
            features_training = features
            indicators_training = indicators

        ht_model, observations_ht, predictions_ht = predict_internal_feature(\
            features_training,
            indicators_training,
            f,
            "hash_table",
            categorical_features,
            args.nbins,
            logger)
        #logger.debug("Hash table: {}".format(ht_model.table))
        _min = features_training.map(lambda x: x[1][f]).min()
        _max = features_training.map(lambda x: x[1][f]).max()
        bin_range = (_min, _max)
        reg_eval = common_utils.evaluate_regression(predictions_ht,
                                                    observations_ht,
                                                    logger,
                                                    args.nbins,
                                                    bin_range)
        results["features"][f]["regression_evaluation_ht"] = reg_eval

        lr_model, observations, predictions = predict_internal_feature(\
            features_training,
            indicators_training,
            f,
            args.regression_model,
            categorical_features,
            args.nbins,
            logger)
        results["features"][f]["model"] = lr_model
        if args.regression_model == "naive_bayes":
            predictions = undiscretize_signle_feature(predictions,
                                                      feature_bin_centers[f])
            observations = undiscretize_signle_feature(observations,
                                                       feature_bin_centers[f])
        all_predicted_features[f] = predictions

        if args.regression_model == "regression_tree":
            model_debug_string = lr_model.toDebugString()
            model_debug_string = common_utils.substitute_feature_names(\
                    model_debug_string, feature_names)
            logger.info(model_debug_string)

        if train_ratio > 0:
            _, observations_ht, predictions_ht = predict_internal_feature(\
                features_test,
                indicators_test,
                f,
                "hash_table",
                categorical_features,
                args.nbins,
                logger)
            _min = features_test.map(lambda x: x[1][f]).min()
            _max = features_test.map(lambda x: x[1][f]).max()
            bin_range = (_min, _max)
            reg_eval = common_utils.evaluate_regression(predictions_ht,
                                                    observations_ht,
                                                    logger,
                                                    args.nbins,
                                                    bin_range)
            results["features"][f]["regression_evaluation_ht_test"] = reg_eval
            logger.debug("Computing predictions on the test set")
            ids_test = indicators_test.keys()
            input_test = indicators_test.values()
            predictions_test = ids_test.zip(lr_model\
                                            .predict(input_test)\
                                            .map(float))
            observations_test = features_test.map(lambda (mid, ftrs):
                    (mid, ftrs[f]))
            if args.regression_model == "naive_bayes":
                predictionsi_test =\
                    undiscretize_signle_feature(predictions_test,
                                                feature_bin_centers[f])
                observations_test =\
                        undiscretize_signle_feature(observations_test,
                                                    feature_bin_centers[f])
            all_predicted_features_test[f] = predictions_test

        predictions_training = predictions

        if eval_regression:
            if args.features_trim_percentile:
                bin_range = model.feature_threshold(f)
            else:
                _min = features_training.map(lambda x: x[1][f]).min()
                _max = features_training.map(lambda x: x[1][f]).max()
                bin_range = (_min, _max)
            logger.debug("Bin range: {}".format(bin_range))
            reg_eval = common_utils.evaluate_regression(predictions,
                                                        observations,
                                                        logger,
                                                        args.nbins,
                                                        bin_range)
            results["features"][f]["regression_evaluation"] = reg_eval
            if train_ratio > 0:
                logger.debug("Evaluating regression on the test set")
                reg_eval_test = common_utils.evaluate_regression(\
                        predictions_test, observations_test, logger,\
                        args.nbins, bin_range)
                results["features"][f]["regression_evaluation_test"] =\
                    reg_eval_test

        if compare_with_replaced_feature:
            logger.debug("Computing predictions of the model with replaced "+\
                "feature %d", f)
            replaced_features = replace_feature_with_predicted(features, f,
                                                               predictions_training,
                                                               logger)

            start = time.time()

            if user_or_product_features == "product":
                uf, pf = other_features, replaced_features
            else:
                uf, pf = replaced_features, other_features

            replaced_mean_error_baseline, replaced_predictions = compare_baseline_to_replaced(\
                        baseline_predictions, uf, pf, logger, power)

            logger.debug("Replaced mean error baseline: "+\
                    "%f", replaced_mean_error_baseline)
            results["features"][f]["replaced_mean_error_baseline"] =\
                replaced_mean_error_baseline

            logger.debug("Evaluating replaced model")
            results["features"][f]["replaced_rec_eval"] =\
                    common_utils.evaluate_recommender(baseline_predictions,\
                        replaced_predictions, logger, args.nbins)


            if train_ratio > 0:
                logger.debug("Computing predictions of the model with replaced "+\
                    "feature %d test", f)
                replaced_features_test = replace_feature_with_predicted(features, f,
                                                                        predictions_test,
                                                                        logger)

                start = time.time()

                if user_or_product_features == "product":
                    uf, pf = other_features, replaced_features_test
                else:
                    uf, pf = replaced_features_test, other_features

                replaced_mean_error_baseline_test, replaced_predictions_test = compare_baseline_to_replaced(\
                           baseline_predictions, uf, pf, logger, power)

                logger.debug("Replaced mean error baseline test: "+\
                    "%f", replaced_mean_error_baseline_test)
                results["features"][f]["replaced_mean_error_baseline_test"] =\
                    replaced_mean_error_baseline_test

                logger.debug("Evaluating replaced model test")
                results["features"][f]["replaced_rec_eval_test"] =\
                    common_utils.evaluate_recommender(baseline_predictions,\
                        replaced_predictions_test, logger, args.nbins)

        if compare_with_randomized_feature:
            logger.debug("Randomizing feature %d", f)
            start = time.time()
            if train_ratio > 0:
                perturbed_subset = training_movies
            else:
                perturbed_subset = None
            replaced_features = common_utils.perturb_feature(features, f,
                    perturbed_subset)
            logger.debug("Done in %f seconds", time.time()-start)
            if user_or_product_features == "product":
                uf, pf = other_features, replaced_features
            else:
                uf, pf = replaced_features, other_features

            logger.debug("Computing predictions of the model with randomized"+\
                " feature %d", f)
            randomized_mean_error_baseline, randomized_predictions = compare_baseline_to_replaced(\
                baseline_predictions, uf, pf, logger, power)

            logger.debug("Radnomized mean error baseline: "+\
                "%f", randomized_mean_error_baseline)
            results["features"][f]["randomized_mean_error_baseline"] =\
                randomized_mean_error_baseline

            logger.debug("Substitution is %f times better than "+\
                         "randomization on the training set",
                         randomized_mean_error_baseline/\
                         replaced_mean_error_baseline)

            logger.debug("Evaluating randomized model")
            results["features"][f]["randomized_rec_eval"] =\
                    common_utils.evaluate_recommender(baseline_predictions,\
                        randomized_predictions, logger, args.nbins)

            if train_ratio > 0:
                logger.debug("Randomizing feature %d test", f)
                start = time.time()
                perturbed_subset = test_movies
                replaced_features = common_utils.perturb_feature(features, f,
                    perturbed_subset)
                logger.debug("Done in %f seconds", time.time()-start)
                if user_or_product_features == "product":
                    uf, pf = other_features, replaced_features
                else:
                    uf, pf = replaced_features, other_features

                logger.debug("Computing predictions of the model with randomized"+\
                    " feature %d test", f)
                randomized_mean_error_baseline_test, randomized_predictions_test\
                        = compare_baseline_to_replaced(\
                            baseline_predictions, uf, pf, logger, power)

                logger.debug("Radnomized mean error baseline test: "+\
                    "%f", randomized_mean_error_baseline_test)
                results["features"][f]["randomized_mean_error_baseline_test"] =\
                    randomized_mean_error_baseline_test

                logger.debug("Substitution is %f times better than "+\
                             "randomization on the test set",
                             randomized_mean_error_baseline/\
                             replaced_mean_error_baseline)

                logger.debug("Evaluating randomized model test")
                results["features"][f]["randomized_rec_eval_test"] =\
                    common_utils.evaluate_recommender(baseline_predictions,\
                        randomized_predictions_test, logger, args.nbins)

    if train_ratio <= 0:
        training_movies = all_movies

    replaced_mean_error_baseline, replaced_rec_eval =\
        compare_with_all_replaced_features(features, other_features,\
            user_or_product_features, all_predicted_features, rank,\
            baseline_predictions, logger, power, args)
    results["all_replaced_mean_error_baseline"] = replaced_mean_error_baseline
    results["all_replaced_rec_eval"] = replaced_rec_eval
    results["all_random_rec_eval"] = compare_with_all_randomized(\
        model, rank, training_movies, baseline_predictions,\
        logger, power, args)

    if train_ratio > 0:
        replaced_mean_error_baseline, replaced_rec_eval =\
            compare_with_all_replaced_features(features, other_features,\
                user_or_product_features, all_predicted_features_test, rank,\
                baseline_predictions, logger, power, args)
        results["all_replaced_mean_error_baseline_test"] = replaced_mean_error_baseline
        results["all_replaced_rec_eval_test"] = replaced_rec_eval
        results["all_random_rec_eval_test"] = compare_with_all_randomized(\
            model, rank, test_movies, baseline_predictions,\
            logger, power, args)

    return results

def display_internal_feature_predictor(results, logger):
    logger.info("Overall results dict: {}".format(results))
    logger.info("Baseline mean error: {}".format(
        results["baseline_mean_error"]))
    logger.info("baseline RMSE: {}".format(
        results["baseline_rmse"]))

    if results["train_ratio"] > 0:
        logger.info("Train ratio: %f %%", results["train_ratio"])

    feature_results = sorted(
        results["features"].items(),
        key=lambda x: x[1]["regression_evaluation"]["mrae"])

    if results["train_ratio"] > 0:
        for f, r in feature_results:
            logger.info("Evaluation of recommender "+\
                    "with replaced feature {}".format(f)+\
                    " on test set: {}".format(r["replaced_rec_eval_test"]))

    header = ["Feature",
              "MRAE",
              "Mean absolute error",
              "Mean error",
              "Mean absolute error HT",
              "MRAE HT"]
    if results["train_ratio"] > 0:
        header += ["MRAE test",
                   "Mean absolute error test",
                   "Mean error test",
                   "Mean absolute error HT test",
                   "MRAE HT test"]
    header += ["Mean feature value",
               "Replaced MERR Baseline",
               "Random MERR Baseline",
               "x better than random"]
    if results["train_ratio"] >0:
        header += ["Replaced MERR Baseline test",
                   "Random MERR Baseline test",
                   "x better than random test"]
    table = PrettyTable(header)
    for f, r in feature_results:
        row = [f,
               r["regression_evaluation"]["mrae"],
               r["regression_evaluation"]["mre"],
               r["regression_evaluation"]["mean_err"],
               r["regression_evaluation_ht"]["mre"],
               r["regression_evaluation_ht"]["mrae"]]
        if results["train_ratio"] > 0:
            row +=  [r["regression_evaluation_test"]["mrae"],
                     r["regression_evaluation_test"]["mre"],
                     r["regression_evaluation_test"]["mean_err"],
                     r["regression_evaluation_ht_test"]["mre"],
                     r["regression_evaluation_ht_test"]["mrae"]]
        row += [results["mean_feature_values"][f],
                r["replaced_mean_error_baseline"],
                r["randomized_mean_error_baseline"],
                float(r["randomized_mean_error_baseline"])/\
                r["replaced_mean_error_baseline"]]
        if results["train_ratio"] > 0:
            row += [r["replaced_mean_error_baseline_test"],
                    r["randomized_mean_error_baseline_test"],
                    float(r["randomized_mean_error_baseline_test"])/\
                            r["replaced_mean_error_baseline_test"]]
        table.add_row(row)
    logger.info("\n" + str(table))
