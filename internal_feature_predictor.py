#standard library
import time
from collections import defaultdict
import math

#pyspark library
from pyspark.mllib.recommendation import ALS
import pyspark.mllib.recommendation
from pyspark.mllib.classification import LabeledPoint
import pyspark.mllib.regression
import pyspark.mllib.tree
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.evaluation import RegressionMetrics,\
        BinaryClassificationMetrics

#prettytable library
from prettytable import PrettyTable

#project files
import AverageRatingRecommender
import parsers_and_loaders
import common_utils

def train_regression_model(data, regression_model="regression_tree",
                           categorical_features={}, max_bins=32, max_depth=None,
                           num_trees=128, logger=None):
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

    if max_depth is None:
        max_depth = int(math.ceil(math.log(max_bins, 2)))

    start = time.time()

    if logger is None:
        print "Training", regression_model
    else:
        logger.debug("Training " + regression_model)

    if regression_model == "regression_tree":
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
    res_rdd = None

    for source in sources:
        if logger is None:
            print "Loading", source["name"]
        else:
            logger.debug("Loading " + source["name"])

        start = time.time()

        cur_rdd, nof, cfi = source["loader"](source["src_rdd"]())
        rdd_count = cur_rdd.count()

        if logger is None:
            print "Done in {} seconds".format(time.time() - start)
            print rdd_count, "records of", nof, "features loaded"
        else:
            logger.debug("Done in {} seconds".format(time.time() - start))
            logger.debug(str(rdd_count) + " records of " +\
                    str(nof) + " features loaded")

        if all_ids is not None:
            cur_ids = set(cur_rdd.keys().collect())
            missing_ids = all_ids - cur_ids
            if len(missing_ids) == 0:
                if logger is None:
                    print "No missing IDs"
                else:
                    logger.debug("No missing IDs")
            else:
                if logger is None:
                    print len(missing_ids), "IDs are missing. "+\
                            "Adding empty records for them"
                else:
                    logger.debug(str(len(missing_ids)) + " IDs are missing. "+\
                            "Adding empty records for them")

                start = time.time()

                empty_records = [(_id, [0 for _ in xrange(nof)]) for _id in
                                 missing_ids]
                empty_records = sc.parallelize(empty_records)
                cur_rdd = cur_rdd.union(empty_records)
                if logger is None:
                    print "Done in {} seconds".format(time.time() - start)
                else:
                    logger.debug("Done in {} seconds".format(time.time() - start))

        shifted_cfi = {f+feature_offset: v for (f, v) in cfi.items()}
        categorical_features_info = dict(categorical_features_info,
                                         **shifted_cfi)

        if res_rdd is None:
            res_rdd = cur_rdd
        else:
            res_rdd = res_rdd\
                    .join(cur_rdd)\
                    .map(lambda (x, (y, z)): (x, y+z))

        feature_offset += nof

    return (res_rdd, feature_offset, categorical_features_info)



def predict_internal_feature(features, indicators, f, regression_model,
                             categorical_features, max_bins, logger=None):
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
        LabeledPoint(ftrs[f], inds))
    ids = joined.map(lambda (_id, _): _id)

    if logger is None:
        print "Done in {} seconds".format(time.time() - start)
    else:
        logger.debug("Done in {} seconds".format(time.time() - start))

    lr_model = train_regression_model(data,
                                      regression_model=regression_model,
                                      categorical_features=categorical_features,
                                      max_bins=max_bins,
                                      logger=logger)

    observations = ids.zip(data.map(lambda x: float(x.label)))
    predictions = ids.zip(
        lr_model\
        .predict(
            data.map(lambda x: x.features)
            )\
        .map(lambda x: float(x))
        )

    return (lr_model, observations, predictions)


def replace_feature_with_predicted(features, f, predictions, logger):
    logger.debug("Replacing original feature "+\
            "{} with predicted values".format(f))
    start = time.time()
    joined = features.join(predictions)
    replaced = joined.map(lambda (mid, (feats, pred)):\
            (mid, common_utils.set_list_value(feats, f, pred)))
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
        .values()
    replaced_mean_error_baseline = common_utils.mean_error(predictionsAndRatings, power)
    logger.debug("Done in %f seconds", time.time() - start)
    return replaced_mean_error_baseline

def internal_feature_predictor(sc, training, rank, numIter, lmbda,
    args, all_movies, metadata_sources, user_or_product_features, eval_regression,
    compare_with_replaced_feature, compare_with_randomized_feature, logger):

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
    indicators, number_of_features, categorical_features =\
            build_meta_data_set(sc, cur_mtdt_srcs, all_movies, logger)

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

    if compare_with_replaced_feature or compare_with_randomized_feature:
        logger.debug("Computing model predictions")
        start = time.time()
        baseline_predictions = model.predictAll(training.map(lambda x:\
            (x[0], x[1])))
        logger.debug("Done in %f seconds", time.time() - start)

        logger.debug("Computing mean error")
        start = time.time()
        predictionsAndRatings = baseline_predictions\
            .map(lambda x: ((x[0], x[1]), x[2])) \
            .join(training.map(lambda x: ((x[0], x[1]), x[2]))) \
            .values()
        baseline_mean_error = common_utils.mean_error(predictionsAndRatings,
                                                      power)
        baseline_rmse = common_utils.mean_error(predictionsAndRatings, power=2.0)
        logger.debug("Done in %f seconds", time.time() - start)
        logger.debug("Mean error: {}, RMSE: {}".format(baseline_mean_error,
                                                       baseline_rmse))
        results["baseline_mean_error"] = baseline_mean_error
        results["baseline_rmse"] = baseline_rmse

    features = model.productFeatures()
    other_features = model.userFeatures()
    if user_or_product_features == "user":
        features, other_features = other_features, features

    results["mean_feature_values"] = common_utils.mean_feature_values(features, logger)

    results["features"] = {}

    for f in xrange(rank):
        lr_model, observations, predictions = predict_internal_feature(\
            features,
            indicators,
            f,
            args.regression_model,
            categorical_features,
            args.nbins,
            logger)
        results["features"][f] = {"model": lr_model}

        if eval_regression:
            reg_eval = common_utils.evaluate_regression(predictions,
                                                        observations, logger)
            results["features"][f]["regression_evaluation"] = reg_eval

        if compare_with_replaced_feature:
            logger.debug("Computing predictions of the model with replaced "+\
                "feature %d", f)
            replaced_features = replace_feature_with_predicted(features, f,
                                                               predictions,
                                                               logger)

            start = time.time()

            if user_or_product_features == "product":
                uf, pf = other_features, replaced_features
            else:
                uf, pf = replaced_features, other_features

            replaced_mean_error_baseline = compare_baseline_to_replaced(\
                        baseline_predictions, uf, pf, logger, power)

            logger.debug("Replaced mean error baseline: "+\
                    "%f", replaced_mean_error_baseline)
            results["features"][f]["replaced_mean_error_baseline"] =\
                replaced_mean_error_baseline

        if compare_with_randomized_feature:
            logger.debug("Randomizing feature %d", f)
            start = time.time()
            replaced_features = common_utils.perturb_feature(features, f)
            logger.debug("Done in %f seconds", time.time()-start)
            if user_or_product_features == "product":
                uf, pf = other_features, replaced_features
            else:
                uf, pf = replaced_features, other_features

            logger.debug("Computing predictions of the model with randomized"+\
                " feature %d", f)
            randomized_mean_error_baseline = compare_baseline_to_replaced(\
                baseline_predictions, uf, pf, logger, power)

            logger.debug("Radnomized mean error baseline: "+\
                "%f", randomized_mean_error_baseline)
            results["features"][f]["randomized_mean_error_baseline"] =\
                randomized_mean_error_baseline
    return results

def display_internal_feature_predictor(results, logger):
    logger.info("Baseline mean error: {}".format(
        results["baseline_mean_error"]))
    logger.info("baseline RMSE: {}".format(
        results["baseline_rmse"]))

    feature_results = sorted(
        results["features"].items(),
        key=lambda x:
            x[1]["regression_evaluation"]["mrae"])

    table = PrettyTable(["Feature",
            "MRAE",
            "Mean absolute error",
            "Mean feature value",
            "Replaced MERR Baseline",
            "Random MERR Baseline",
            "x better than random"])
    for f, r in feature_results:
        table.add_row([f,
            r["regression_evaluation"]["mrae"],
            r["regression_evaluation"]["mre"],
            results["mean_feature_values"][f],
            r["replaced_mean_error_baseline"],
            r["randomized_mean_error_baseline"],
            float(r["randomized_mean_error_baseline"])/r["replaced_mean_error_baseline"]
            ])
    logger.info("\n" + str(table))
