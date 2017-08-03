#standard library
import time
import math
import random
from collections import defaultdict
import os.path
import pickle
import functools
import shutil

#prettytable library
from prettytable import PrettyTable

#pyspark library
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel
import pyspark.mllib.recommendation
from pyspark.mllib.classification import LabeledPoint, NaiveBayes
import pyspark.mllib.regression
import pyspark.mllib.classification
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
import CustomFeaturesRecommender
import product_topics


regression_models = ["regression_tree", "random_forest", "linear",
                     "naive_bayes", "logistic"]
metadata_sources = ["name", "genres", "tags", "imdb_keywords",
                    "imdb_genres", "imdb_director", "imdb_producer",
                    "tvtropes", "average_rating", "users", "years",
                    "imdb_year", "imdb_rating", "imdb_cast",
                    "imdb_cinematographer", "imdb_composer",
                    "imdb_languages", "imdb_production_companies",
                    "imdb_writer", "topics"]

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
    map_f = functools.partial(lambda bin_edges, (mid, val):
            (mid, int(np.digitize(val, bin_edges))),
            bin_edges)
    data = data.map(map_f)
    return data, bin_centers

def undiscretize_signle_feature(data, bin_centers):
    """
    data: RDD of ID: bin_n
    """
    map_f = functools.partial(lambda bin_centers, (mid, bin_n):
            (mid, float(bin_centers[bin_n])),
            bin_centers)
    data = data.map(map_f)
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
        map_f = functools.partial(lambda f, (mid, ftrs):
                (mid, ftrs[f]),
                f)
        cur_f_data = data.map(map_f)
        cur_f_data, bin_centers = discretize_single_feature(cur_f_data,
                                                            nbins,
                                                            logger)
        map_f = functools.partial(lambda f, (mid, (ftrs, new_f_val)):
                           (mid, common_utils.set_list_value(ftrs, f,
                                                             new_f_val)),
                           f)
        data = data\
               .join(cur_f_data)\
               .map(map_f)
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

    logger.debug("train_regression_model(data.count={},".format(data.count())+\
                 " regression_model=\"{}\",".format(regression_model)+\
                 " len(categorical_features)={},".format(len(categorical_features))+\
                 " max_bins={},".format(max_bins)+\
                 " max_depth={},".format(max_depth)+\
                 " num_trees={},".format(num_trees)+\
                 " is_classifier={},".format(is_classifier)+\
                 " num_classes={})".format(num_classes))

    start = time.time()
    if num_classes is None:
        num_classes = max_bins
        logger.debug("num_classes is None, setting to max_bins={}".format(max_bins))

    if max_depth is None:
        max_depth = int(math.ceil(math.log(max_bins, 2)))
        logger.debug("max_depth is None, setting to {}".format(max_depth))

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
        lr_model = LinearRegressionWithSGD.train(data, step=0.01)
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


def build_meta_data_set(sc, sources, all_ids=None, logger=None,
                        drop_threshold=None):
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

        src_rdd = source["src_rdd"]()
        cur_rdd, nof, cfi, fnames = source["loader"](src_rdd)
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

        if drop_threshold is not None:
            cur_rdd, nof, cfi, fnames =\
                    drop_rare_features(cur_rdd, nof, cfi, fnames,
                                       drop_threshold, logger)

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
                             num_classes=None, max_depth=None):
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
    logger.debug("predict_internal_feature("+\
                 "features.count()={},".format(features.count())+\
                 " indicators.count()={},".format(indicators.count())+\
                 " f={},".format(f)+\
                 " regression_model=\"{}\",".format(regression_model)+\
                 " len(categorical_features)={},".format(len(categorical_features))+\
                 " max_bins={},".format(max_bins)+\
                 " no_threshold={},".format(no_threshold)+\
                 " is_classifier={},".format(is_classifier)+\
                 " num_classes={},".format(num_classes)+\
                 " max_depth={})".format(max_depth))
    if logger is None:
        print "Processing feature {}".format(f)
        print "Building data set"
    else:
        logger.debug("Processing feature {}".format(f))
        logger.debug("Building data set")

    start = time.time()

    joined = features.join(indicators)
    map_f = functools.partial( ( lambda f, (_id, (ftrs, inds)):
        LabeledPoint(float(ftrs[f]), inds) ), f)
    data = joined.map(map_f).cache()
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
                                      num_classes=num_classes,
                                      max_depth=max_depth)
    #    if regression_model in ["linear", "logistic"]:
    #    logger.debug("Model weights: {}".format(lr_model.weights))
    if no_threshold and regression_model == "logistic":
        lr_model.clearThreshold()

    observations = common_utils.safe_zip(ids, data.map(lambda x: float(x.label))).cache()
    predictions = common_utils.safe_zip(ids,
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
    filter_f = functools.partial(lambda not_predicted_ids, x: x[0] in
            not_predicted_ids, not_predicted_ids)
    features_intact = features.filter(filter_f)
    logger.debug("No predictions for {} items out of {}, leaving as is".\
            format(features_intact.count(), features.count()))
    joined = features.join(predictions)
    map_f = functools.partial(lambda f, (mid, (feats, pred)):\
            (mid, common_utils.set_list_value(feats, f, pred)), f)
    replaced = joined.map(map_f)
    replaced = replaced.union(features_intact).cache()
    logger.debug("Done in {} seconds".format(time.time() - start))
    return replaced

def replace_all_features_with_predicted(sc, features, all_predictions, logger,
                                        args):
    logger.debug("Replacing all original features "+\
            "with predicted values")
    start = time.time()
    logger.debug("Creating predictions RDD")
    rank = len(all_predictions)
    predictions = None
    for f in xrange(rank):
        logger.debug("Adding feature {}".format(f))
        cur_data = dict(all_predictions[f].map(lambda (x, y): (x, [y])).collect())
        if predictions is None:
            predictions = cur_data
        else:
            predictions = common_utils.dict_join(predictions,
                                                 cur_data,
                                                 join_f = lambda x, y: x+y)
    logger.debug("Predictions for {} products created".format(len(predictions)))
    logger.debug("Turning into an RDD")
    predictions =\
        sc.parallelize(predictions.items()).repartition(args.num_partitions).cache()
    features_ids = set(features.keys().collect())
    predictions_ids = set(predictions.keys().collect())
    not_predicted_ids = features_ids - predictions_ids
    filter_f = functools.partial(lambda not_predicted_ids, x: x[0] in
            not_predicted_ids, not_predicted_ids)
    features_intact = features.filter(filter_f)
    logger.debug("No predictions for {} items out of {}, leaving as is".\
            format(features_intact.count(), features.count()))
    joined = features.join(predictions)
    replaced = joined.map(lambda (mid, (feats, preds)): (mid, preds))
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

def compare_with_all_replaced_features(sc, features, other_features,
                                       user_or_product_features,
                                       all_predicted_features, rank,
                                       baseline_predictions, logger, power,
                                       args):
    logger.debug("Computing predictions of the model with all "+\
        "replaced features")
    start = time.time()
    features = replace_all_features_with_predicted(sc, features,
            all_predicted_features, logger, args)

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
                        replaced_predictions, logger, args.nbins,
                        "All replaced features")
    logger.debug("Done in %f seconds", time.time() - start)
    return replaced_mean_error_baseline, replaced_rec_eval

def compare_with_all_randomized(sc, baseline_model, rank, perturbed_subset,
                                baseline_predictions, logger, power,
                                args):
    logger.debug("Evaluating a completely randomized model")
    start = time.time()
    model = RandomizedRecommender.RandomizedRecommender(\
        sc, baseline_model, rank, perturbed_subset, logger)
    model.randomize()
    randomized_predictions = model.predictAll(baseline_predictions)
    replaced_rec_eval =\
                    common_utils.evaluate_recommender(baseline_predictions,\
                        randomized_predictions, logger, args.nbins,
                        "All randomized features")
    logger.debug("Done in %f seconds", time.time() - start)
    return replaced_rec_eval

def measure_associativity(input_features, target_features, f, logger):
    logger.debug("Measuring associativity")
    start = time.time()
    map_f = functools.partial(lambda f, (mid, (inds, ftrs)): (inds, ftrs[f]),
            f)
    joined = input_features\
             .join(target_features)\
             .map(map_f)
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

def load_or_train_ALS(training, rank, numIter, lmbda, args, sc, logger):
    if args.persist_dir is not None:
        fname = os.path.join(args.persist_dir, "als_model.pkl")
        fname2 = os.path.join(args.persist_dir, "upr_model.pkl")
        if not os.path.exists(fname):
            if os.path.exists(fname2):
                need_new_model = False
                write_model = False
                logger.debug("Loading %s", fname2)
                model = CustomFeaturesRecommender.load(sc, fname2)
            else:
                logger.debug("%s not found, bulding a new model", fname)
                need_new_model = True
                write_model = True
        else:
            need_new_model = False
            write_model = False
            logger.debug("Loading %s", fname)
            model = MatrixFactorizationModel.load(sc, fname)
    else:
        need_new_model = True
        write_model = False
    if need_new_model:
        logger.debug("Training ALS recommender")
        start = time.time()
        model = ALS.train(training, rank=rank, iterations=numIter,
                      lambda_=lmbda, nonnegative=args.non_negative)
        logger.debug("Done in %f seconds", time.time() - start)
    if write_model:
        if os.path.exists(fname):
            logger.debug("%a already exists, removing")
            shutil.rmtree(fname)
        logger.debug("Saving model to %s", fname)
        model.save(sc, fname)
    return model

def load_metadata_process_training(sc, args, metadata_sources, training,
                                   logger, all_movies):
    if args.persist_dir is not None:
        fname = os.path.join(args.persist_dir, "indicators.pkl")
        if os.path.exists(fname):
            logger.debug("Loading %s", fname)
            ifile = open(fname, "rb")
            objects = pickle.load(ifile)
            ifile.close()
            (indicators_c, nof, categorical_features, feature_names,
                  all_movies, training_c) = objects
            indicators = sc.parallelize(indicators_c)\
                    .repartition(args.num_partitions)\
                    .cache()
            training = sc.parallelize(training_c)\
                    .repartition(args.num_partitions)\
                    .cache()
            need_new_model = False
            write_model = False
        else:
            logger.debug("{} not found, building new features".format(fname))
            need_new_model = True
            write_model = True
    else:
        need_new_model = True
        write_model = False

    if need_new_model:
        cur_mtdt_srcs = filter(lambda x: x["name"] in args.metadata_sources, metadata_sources)
        if args.drop_missing_movies:
            indicators, nof, categorical_features, feature_names =\
                build_meta_data_set(sc, cur_mtdt_srcs, None, logger)
            old_all_movies = all_movies
            all_movies = set(indicators.keys().collect())
            logger.debug("{} movies loaded, data for {} is missing, purging them"\
                .format(len(all_movies), len(old_all_movies - all_movies)))
            filter_f = functools.partial(lambda all_movies, x: x[1] in
                    all_movies, all_movies)
            training = training.filter(filter_f)
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

    if write_model:
        logger.debug("Writing %s", fname)
        indicators_c = indicators.collect()
        training_c = training.collect()
        objects = (indicators_c, nof, categorical_features, feature_names,
                  all_movies, training_c)
        ofile = open(fname, "wb")
        pickle.dump(objects, ofile)
        ofile.close()
    return (indicators, nof, categorical_features, feature_names, all_movies,
           training)

def load_or_build_baseline_predictions(sc, model, power, results, logger, args,
        training):
    if args.persist_dir is not None:
        fname = os.path.join(args.persist_dir, "baseline_predictions.pkl")
        if os.path.exists(fname):
            logger.debug("Loading %s", fname)
            ifile = open(fname, "rb")
            objects = pickle.load(ifile)
            ifile.close()
            (baseline_predictions_c, _) = objects
            baseline_predictions = sc.parallelize(baseline_predictions_c)\
                    .repartition(args.num_partitions)\
                    .cache()
            need_new_model = False
            write_model = False
        else:
            logger.debug("{} not found, building new predictions".format(fname))
            need_new_model = True
            write_model = True
    else:
        need_new_model = True
        write_model = False

    if need_new_model:
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
               training, baseline_predictions, logger, args.nbins,
               "Original recommender")
        results["baseline_rec_eval"] = baseline_rec_eval

    if write_model:
        logger.debug("Writing %s", fname)
        baseline_predictions_c = baseline_predictions.collect()
        objects = (baseline_predictions_c, results)
        ofile = open(fname, "wb")
        pickle.dump(objects, ofile)
        ofile.close()
        fname = os.path.join(args.persist_dir, "results.pkl")
        logger.debug("Writing %s", fname)
        ofile = open(fname, "wb")
        pickle.dump(results, ofile)
        ofile.close
    return baseline_predictions, results

def load_or_train_trimmed_recommender(model, args, sc, results, rank, logger,
        training, baseline_predictions):
    old_model = model
    old_productFeatures = old_model.productFeatures()
    old_userFeatures = old_model.userFeatures()
    if args.persist_dir is not None:
        fname_model = os.path.join(args.persist_dir, "trimmed_recommender.pkl")
        fname_results = os.path.join(args.persist_dir, "results.pkl")
        if not os.path.exists(fname_model) or not os.path.exists(fname_results):
           logger.debug("%s or %s not found, bulding a new model",
                        fname_model, fname_results)
           need_new_model = True
           write_model = True
        else:
           need_new_model = False
           write_model = False
           logger.debug("Loading %s", fname_model)
           model = TrimmedFeatureRecommender.load(fname_model, sc, args.num_partitions)
           logger.debug("Loading %s", fname_results)
           ifile = open(fname_results, "rb")
           results_ = pickle.load(ifile)
           ifile.close()
    else:
       need_new_model = True
       write_model = False
    if need_new_model:
       logger.debug("Training trimmed recommender")
       start = time.time()
       model = TrimmedFeatureRecommender.TrimmedFeatureRecommender(\
           rank, old_userFeatures, old_productFeatures,\
           args.features_trim_percentile, logger).train()
       logger.debug("Computing trimmed predictions")
       trimmed_predictions = model.predictAll(training)
       results["trimmed_rec_eval"] = common_utils.evaluate_recommender(\
           baseline_predictions, trimmed_predictions, logger, args.nbins,
           "Thresholded features recommender")
       logger.debug("Done in %f seconds", time.time() - start)
    if write_model:
       logger.debug("Saving model to %s", fname_model)
       TrimmedFeatureRecommender.save(model, fname_model)
       if os.path.exists(fname_model):
           logger.debug("%a already exists, removing")
           shutil.rmtree(fname_model)
       logger.debug("Saving results to %s", fname_results)
       ofile = open(fname_results, "wb")
       pickle.dump(results, ofile)
       ofile.close()
    return old_model, old_productFeatures, old_userFeatures, model, results

def compute_or_load_mean_feature_values(args, features, results, logger):
    if args.persist_dir is not None:
        fname = os.path.join(args.persist_dir, "results.pkl")
        if "mean_feature_values" not in results:
           logger.debug("mean feature values not found, computing")
           need_new_model = True
           write_model = True
        else:
           need_new_model = False
           write_model = False
    else:
       need_new_model = True
       write_model = False
    if need_new_model:
       logger.debug("Computing mean feature values")
       results["mean_feature_values"] = common_utils.mean_feature_values(features, logger)
    if write_model:
       logger.debug("Saving results to %s", fname)
       ofile = open(fname, "wb")
       pickle.dump(results, ofile)
       ofile.close()
    return results

def compute_or_load_discrete_features(features, indicators, results, logger,
                                      args, categorical_features):
    if args.persist_dir is not None:
        fname = os.path.join(args.persist_dir, "discrete_features.pkl")
        if not os.path.exists(fname):
           logger.debug("%s or %s not found, discretizing features",
                        fname)
           need_new_model = True
           write_model = True
        else:
           need_new_model = False
           write_model = False
           logger.debug("Loading %s", fname)
           ifile = open(fname, "rb")
           (features_c, indicators_c, feature_bin_centers, _) = pickle.load(fname)
           ifile.close()
           indicators = sc.parallelize(indicators_c)\
                    .repartition(args.num_partitions)\
                    .cache()
           features = sc.parallelize(features_c)\
                    .repartition(args.num_partitions)\
                    .cache()
    else:
       need_new_model = True
       write_model = False
    if need_new_model:
        logger.debug("Discretizing features for naive bayes")
        features, feature_bin_centers = discretize_features(features,
                                                            args.nbins,
                                                            set([]),
                                                            logger)
        logger.debug("Feature bin centers: {}".format(feature_bin_centers))
        results["mean_discrete_feature_values"] = common_utils.mean_feature_values(features, logger)
        indicators, _ = discretize_features(indicators, args.nbins,
                                            set(categorical_features.keys()),
                                            logger)
    if write_model:
       logger.debug("Saving results to %s", fname)
       ofile = open(fname, "wb")
       features_c = features.collect()
       indicators_c = indicators.collect()
       pickle.dump((features_c, indicators_c, feature_bin_centers, results), ofile)
       ofile.close()
    return (features, indicators, feature_bin_centers, results)

def split_or_load_training_test_sets(train_ratio, all_movies, features,
                                     indicators, features_original, n_movies,
                                     args, logger, sc):
    if args.persist_dir is not None:
        fname = os.path.join(args.persist_dir, "features_training_test.pkl")
        if os.path.exists(fname):
            logger.debug("Loading %s", fname)
            ifile = open(fname, "rb")
            objects = pickle.load(ifile)
            ifile.close()
            (training_movies, test_movies, features_test_c, features_training_c,
             features_original_test_c, features_original_training_c,
             indicators_training_c, indicators_test_c) = objects
            indicators_training = sc.parallelize(indicators_training_c)\
                    .repartition(args.num_partitions)\
                    .cache()
            indicators_training = sc.parallelize(indicators_training_c)\
                    .repartition(args.num_partitions)\
                    .cache()
            features_training = sc.parallelize(features_training_c)\
                    .repartition(args.num_partitions)\
                    .cache()
            features_original_training = sc.parallelize(features_original_training_c)\
                    .repartition(args.num_partitions)\
                    .cache()
            if indicators_test_c is not None:
                indicators_test = sc.parallelize(indicators_test_c)\
                    .repartition(args.num_partitions)\
                    .cache()
            else:
                features_test = None
            if features_test_c is not None:
                features_test = sc.parallelize(features_test_c)\
                    .repartition(args.num_partitions)\
                    .cache()
            else:
                features_test = None
            if features_original_test_c is not None:
                features_original_test = sc.parallelize(features_original_test_c)\
                    .repartition(args.num_partitions)\
                    .cache()
            else:
                features_original_test = None
            if indicators_test_c is not None:
                indicators_test = sc.parallelize(indicators_test_c)\
                    .repartition(args.num_partitions)\
                    .cache()
            else:
                indicators_test = None
            need_new_model = False
            write_model = False
        else:
            logger.debug("{} not found, building new split".format(fname))
            need_new_model = True
            write_model = True
    else:
        need_new_model = True
        write_model = False
    if need_new_model:
         logger.debug("Building training and test sets for regression")
         training_size = int(math.floor(n_movies * train_ratio / 100.0)) + 1
         logger.debug("{} items in training and {} in test set"\
                 .format(training_size, n_movies - training_size))
         all_movies = list(all_movies)
         random.shuffle(all_movies)
         training_movies = set(all_movies[:training_size])
         test_movies = set(all_movies[training_size:])
         training_filter_f = functools.partial(lambda training_movies, x: x[0]
                 in training_movies, training_movies)
         test_filter_f = functools.partial(lambda test_movies, x: x[0]
                 in test_movies, test_movies)
         features_training = features.filter(training_filter_f).cache()
         features_test = features.filter(test_filter_f).cache()
         indicators_training = indicators.filter(training_filter_f).cache()
         indicators_test = indicators.filter(test_filter_f).cache()
         features_original_training =\
            features_original.filter(training_filter_f).cache()
         features_original_test = features_original.filter(test_filter_f).cache()
    if write_model:
         logger.debug("Writing %s", fname)
         features_test_c = features_test.collect()
         features_training_c = features_training.collect()
         features_original_test_c = features_original_test.collect()
         features_original_training_c = features_original_training.collect()
         indicators_training_c = indicators_training.collect()
         indicators_test_c = indicators_test.collect()
         objects = (training_movies, test_movies, features_test_c, features_training_c,
             features_original_test_c, features_original_training_c,
             indicators_training_c, indicators_test_c)
         ofile = open(fname, "wb")
         pickle.dump(objects, ofile)
         ofile.close()
    return (training_movies, test_movies, features_test, features_training,
             features_original_test, features_original_training,
             indicators_training, indicators_test)

def normalize_features(features, categorical_features, feature_names, logger):
    logger.debug("Normalizing features")
    features_to_normalize = set(feature_names.keys()) -\
                            set(categorical_features.keys())
    for f in features_to_normalize:
        logger.debug("Normalizing feature {} ({})".format(f, feature_names[f]))
        map_f = functools.partial(lambda f, (mid, ftrs):
            (mid, [ftrs[f]]), f)
        cur_feature = features.map(map_f)
        ranges = common_utils.feature_ranges(cur_feature, logger)
        _min, _max = ranges[0]["min"], ranges[0]["max"]
        factor = float(_max - _min)
        logger.debug("Scaling factor for feature {} ({}): {}".format(f, feature_names[f],
                    factor))
        map_f = functools.partial(
                lambda f, factor, _min, (mid, ftrs):
                    (mid, common_utils.set_list_value(ftrs, f,
                        float(ftrs[f]-_min)/factor, True)),
                f, factor, _min)
        features = features.map(map_f)
    return features

def internal_feature_predictor(sc, training, rank, numIter, lmbda,
                               args, all_movies, metadata_sources,
                               user_or_product_features, eval_regression,
                               compare_with_replaced_feature,
                               compare_with_randomized_feature, logger,
                               train_ratio = 0):
    logger.debug("Started internal_feature_predictor")
    if args.persist_dir is not None:
        logger.debug("Trying to load previous results")
        fname = os.path.join(args.persist_dir, "results.pkl")
        if os.path.exists(fname):
            logger.debug("Loading %s", fname)
            ifile = open(fname, "rb")
            results = pickle.load(ifile)
            ifile.close()
            if "features" in results:
                logger.debug("{} features already processed".\
                    format(len(results["features"])))
            else:
                logger.debug("No information about features in results")
                results["features"] = {}
        else:
            logger.debug("%s not found", fname)
            results = {}
            results["features"] = {}
    else:
        logger.debug("Not trying to load previous results")
        results = {}
        results["features"] = {}
    power = 1.0

    if "average_rating" in args.metadata_sources:
        arc = AverageRatingRecommender.AverageRatingRecommender(logger)
        arc.train(training)
        arc_ratings =\
            sc.parallelize(arc.ratings.items()).repartition(args.num_partitions)
        metadata_sources.append(
            {"name": "average_rating",
             "src_rdd": functools.partial(lambda x: x, arc_ratings),
             "loader": parsers_and_loaders.load_average_ratings})

    model = load_or_train_ALS(training, rank, numIter, lmbda, args, sc, logger)
    if compare_with_replaced_feature or compare_with_randomized_feature or\
            args.features_trim_percentile:
        baseline_predictions, results =\
            load_or_build_baseline_predictions(sc, model, power, results,
                    logger, args, training)
    if args.features_trim_percentile:
        old_model, old_productFeatures, old_userFeatures, model, results =\
            load_or_train_trimmed_recommender(model, args, sc, results, rank,
                                              logger, training,
                                              baseline_predictions)
    else:
        old_model = model

    features = model.productFeatures()
    other_features = model.userFeatures()
    logger.debug("user_or_product_features: {}"\
            .format(user_or_product_features))
    if user_or_product_features == "user":
        features, other_features = other_features, features
    features_original = features


    indicators, nof, categorical_features, feature_names, all_movies, training=\
        load_metadata_process_training(sc, args, metadata_sources, training,
        logger, all_movies)

    if args.drop_rare_movies > 0:
        logger.debug("Dropping movies with fewer than %d non-zero "+\
                "features", args.drop_rare_movies)
        features, indicators = common_utils.drop_rare_movies(features,
                                                             indicators,
                                                             args.drop_rare_movies)
        logger.debug("%d movies left", features.count())
        all_movies = set(indicators.keys().collect())
        filter_f = functools.partial(lambda all_movies, x: x[1] in all_movies,
                all_movies)
        training = training.filter(filter_f)
        baseline_predictions = baseline_predictions.filter(filter_f)
        logger.debug("{} items left in the training set"\
            .format(training.count()))
        all_movies = list(all_movies)

    if args.topic_modeling:
        movies_dict = {m: str(m) for m in all_movies}
        indicators, feature_names, categorical_features = product_topics.topicize_indicators(sc, movies_dict, indicators, feature_names,
                categorical_features, num_topics=15, num_words=10, passes=100)
        logger.debug("Indicators sample after topic modeling: {}".format(indicators.take(2)))
    if args.normalize:
        indicators = normalize_features(indicators, categorical_features,
                feature_names, logger)

    results["feature_names"] = feature_names
    results["categorical_features"] = categorical_features


    results = compute_or_load_mean_feature_values(args, features, results, logger)

    if args.regression_model == "naive_bayes":
        (features, indicators, feature_bin_centers, results) =\
             compute_or_load_discrete_features(features, indicators, results,
                                               logger, args, categorical_features)


    results["train_ratio"] = train_ratio

    all_predicted_features = {}
    all_predicted_features_test = {}
    all_lr_models = {}

    models_by_name = {"regression_tree": pyspark.mllib.tree.DecisionTreeModel,
                      "random_forest": pyspark.mllib.tree.RandomForestModel,
                      "linear": pyspark.mllib.regression.LinearRegressionModel,
                      "logistic": pyspark.mllib.classification.LogisticRegressionModel,
                      "naive_bayes": pyspark.mllib.classification.NaiveBayesModel}

    n_movies = len(all_movies)
    if train_ratio > 0:
         (training_movies, test_movies, features_test, features_training,
             features_original_test, features_original_training,
             indicators_training, indicators_test) =\
                  split_or_load_training_test_sets(train_ratio, all_movies, features,
                                     indicators, features_original, n_movies,
                                     args, logger, sc)
         filter_f = functools.partial(lambda training_movies, x: x[1] in
                 training_movies, training_movies)
         baseline_predictions_training = baseline_predictions.filter(filter_f)
         filter_f = functools.partial(lambda test_movies, x: x[1] in
                 test_movies, test_movies)
         baseline_predictions_test = baseline_predictions.filter(filter_f)
         logger.debug("{} feature rows, {} indicator rows,  and {} ratings in the training set".\
                 format(features_training.count(), indicators_training.count(), baseline_predictions_training.count()) +\
                 ", {} feature rows, {} indicators rows, and {} ratings in the test set".\
                 format(features_test.count(), indicators_test.count(), baseline_predictions_test.count()))
    else:
         features_training = features
         indicators_training = indicators
         features_original_training = features_original
         training_movies = all_movies
         test_movies = None
         indicators_test = None
         features_test = None
         baseline_predictions_training = baseline_predictions
         baseline_predictions_test = None

    for f in xrange(rank):
        logger.debug("Processing {} out of {}"\
                .format(f, rank))
        if "features" not in results:
            logger.debug("Features dict not found, adding")
            results["features"] = {}
        if f in results["features"]:
            fname = os.path.join(args.persist_dir,
                                 "lr_model_{}.pkl".format(f))
            logger.debug("Already processed, loading %s", fname)
            lr_model = models_by_name[args.regression_model].load(sc, fname)
            all_lr_models[f] = lr_model
            fname = os.path.join(args.persist_dir,
                                 "predictions_{}.pkl".format(f))
            logger.debug("Loading %s", fname)
            ifile = open(fname, "rb")
            (predictions_training_c, predictions_test_c) = pickle.load(ifile)
            ifile.close()
            predictions_training= sc.parallelize(predictions_training_c)\
                .repartition(args.num_partitions)\
                .cache()
            if predictions_test_c is not None:
                predictions_test= sc.parallelize(predictions_test_c)\
                    .repartition(args.num_partitions)\
                    .cache()
            else:
                predictions_test = None
            all_predicted_features[f] = predictions_training
            all_predicted_features_test[f] = predictions_test
            continue
        logger.debug("Feature {} not in {}, processing".format(f,
            results["features"].keys()))
        results["features"][f] = {}
        map_f = functools.partial(lambda f, (mid, ftrs): (mid, ftrs[f]), f)
        observations_training = features_original_training.map(map_f)
        observations_test = features_original_test.map(map_f)

        if not args.no_ht:
            ht_model, _, predictions_ht = predict_internal_feature(\
                features_training,
                indicators_training,
                f,
                "hash_table",
                categorical_features,
                args.nbins,
                logger)
            if args.regression_model == "naive_bayes":
                logger.debug("Undiscretizing values for hast table")
                predictions_ht = undiscretize_signle_feature(predictions_ht,
                                                      feature_bin_centers[f])
                #logger.debug("Hash table: {}".format(ht_model.table))
            reg_eval = common_utils.evaluate_regression(predictions_ht,
                                                    observations_training,
                                                    logger,
                                                    args.nbins,
                                                    bin_range=None,
                     model_name = "Hash table training feature {}".format(f))
            results["features"][f]["regression_evaluation_ht"] = reg_eval

        lr_model, _, predictions = predict_internal_feature(\
            features=features_training,
            indicators=indicators_training,
            f=f,
            regression_model=args.regression_model,
            categorical_features=categorical_features,
            max_bins=args.nbins,
            logger=logger,
            no_threshold=True,
            max_depth=args.max_depth)

        all_lr_models[f] = lr_model
        results["features"][f]["model"] = args.regression_model
        if args.regression_model == "naive_bayes":
            logger.debug("Undiscretizing values for training predictions")
            predictions = undiscretize_signle_feature(predictions,
                                                      feature_bin_centers[f])
        all_predicted_features[f] = predictions
        predictions_training = predictions

        if args.regression_model == "regression_tree":
            model_debug_string = lr_model.toDebugString()
            model_debug_string = common_utils.substitute_feature_names(\
                    model_debug_string, feature_names)
            logger.info(model_debug_string)

        if train_ratio > 0:
            if not args.no_ht:
                _, _, predictions_ht_test = predict_internal_feature(\
                    features_test,
                    indicators_test,
                    f,
                    "hash_table",
                    categorical_features,
                    args.nbins,
                    logger)
                if args.regression_model == "naive_bayes":
                    logger.debug("Undiscretizing values for hash table test")
                    predictions_ht_test =\
                        undiscretize_signle_feature(predictions_ht_test,
                                                      feature_bin_centers[f])
                reg_eval = common_utils.evaluate_regression(predictions_ht_test,
                                                    observations_test,
                                                    logger,
                                                    args.nbins,
                                                    bin_range=None,
                     model_name = "Hash table test feature {}".format(f))
                results["features"][f]["regression_evaluation_ht_test"] = reg_eval
            logger.debug("Computing predictions on the test set")
            ids_test = indicators_test.keys()
            input_test = indicators_test.values()
            predictions_test = common_utils.safe_zip(ids_test, lr_model\
                                            .predict(input_test)\
                                            .map(float))
            if args.regression_model == "naive_bayes":
                logger.debug("Undiscretizing values for test predictions")
                predictions_test =\
                    undiscretize_signle_feature(predictions_test,
                                                feature_bin_centers[f])
            all_predicted_features_test[f] = predictions_test

        if eval_regression:
            reg_eval = common_utils.evaluate_regression(predictions_training,
                                                        observations_training,
                                                        logger,
                                                        args.nbins,
                                                        bin_range=None,
                     model_name = "Training feature {}".format(f))
            results["features"][f]["regression_evaluation"] = reg_eval
            if train_ratio > 0:
                logger.debug("Evaluating regression on the test set")
                reg_eval_test = common_utils.evaluate_regression(\
                        predictions_test, observations_test, logger,\
                        args.nbins, bin_range=None,
                     model_name = "Test feature {}".format(f))
                results["features"][f]["regression_evaluation_test"] =\
                    reg_eval_test

        if compare_with_replaced_feature:
            logger.debug("Computing predictions of the model with replaced "+\
                "feature %d", f)
            replaced_features =\
                replace_feature_with_predicted(features_original_training, f,
                                                               predictions_training,
                                                               logger)

            start = time.time()

            if user_or_product_features == "product":
                uf, pf = other_features, replaced_features
            else:
                uf, pf = replaced_features, other_features

            replaced_mean_error_baseline, replaced_predictions = compare_baseline_to_replaced(\
                        baseline_predictions_training, uf, pf, logger, power)

            logger.debug("Replaced mean error baseline: "+\
                    "%f", replaced_mean_error_baseline)
            results["features"][f]["replaced_mean_error_baseline"] =\
                replaced_mean_error_baseline

            logger.debug("Evaluating replaced model")
            results["features"][f]["replaced_rec_eval"] =\
                    common_utils.evaluate_recommender(baseline_predictions_training,\
                        replaced_predictions, logger, args.nbins,
                        "Replaced feature {} training".format(f))


            if train_ratio > 0:
                logger.debug("Computing predictions of the model with replaced "+\
                    "feature %d test", f)
                replaced_features_test =\
                    replace_feature_with_predicted(features_original_test, f,
                                                                        predictions_test,
                                                                        logger)
                start = time.time()

                if user_or_product_features == "product":
                    uf, pf = other_features, replaced_features_test
                else:
                    uf, pf = replaced_features_test, other_features
                replaced_mean_error_baseline_test, replaced_predictions_test = compare_baseline_to_replaced(\
                           baseline_predictions_test, uf, pf, logger, power)

                logger.debug("Replaced mean error baseline test: "+\
                    "%f", replaced_mean_error_baseline_test)
                results["features"][f]["replaced_mean_error_baseline_test"] =\
                    replaced_mean_error_baseline_test

                logger.debug("Evaluating replaced model test")
                results["features"][f]["replaced_rec_eval_test"] =\
                    common_utils.evaluate_recommender(baseline_predictions_test,\
                        replaced_predictions_test, logger, args.nbins,
                        "Replaced feature {} test".format(f))

        if compare_with_randomized_feature:
            logger.debug("Randomizing feature %d", f)
            start = time.time()
            if train_ratio > 0:
                perturbed_subset = training_movies
            else:
                perturbed_subset = None
            replaced_features =\
                common_utils.perturb_feature(features_original_training, f,
                    perturbed_subset)
            logger.debug("Done in %f seconds", time.time()-start)
            if user_or_product_features == "product":
                uf, pf = other_features, replaced_features
            else:
                uf, pf = replaced_features, other_features

            logger.debug("Computing predictions of the model with randomized"+\
                " feature %d", f)
            randomized_mean_error_baseline, randomized_predictions = compare_baseline_to_replaced(\
                baseline_predictions_training, uf, pf, logger, power)

            logger.debug("Radnomized mean error baseline: "+\
                "%f", randomized_mean_error_baseline)
            results["features"][f]["randomized_mean_error_baseline"] =\
                randomized_mean_error_baseline
            if replaced_mean_error_baseline != 0:
                logger.debug("Substitution is %f times better than "+\
                         "randomization on the training set",
                         randomized_mean_error_baseline/\
                         replaced_mean_error_baseline)
            else:
                logger.debug("Substitution is inf times better than "+\
                        "randomization on the training set")
            logger.debug("Evaluating randomized model")
            results["features"][f]["randomized_rec_eval"] =\
                    common_utils.evaluate_recommender(baseline_predictions_training,\
                        randomized_predictions, logger, args.nbins,
                        "Randomized feature {} training".format(f))

            if train_ratio > 0:
                logger.debug("Randomizing feature %d test", f)
                start = time.time()
                perturbed_subset = test_movies
                replaced_features =\
                    common_utils.perturb_feature(features_original_test, f,
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
                            baseline_predictions_test, uf, pf, logger, power)

                logger.debug("Radnomized mean error baseline test: "+\
                    "%f", randomized_mean_error_baseline_test)
                results["features"][f]["randomized_mean_error_baseline_test"] =\
                    randomized_mean_error_baseline_test
                if replaced_mean_error_baseline_test != 0:
                    logger.debug("Substitution is %f times better than "+\
                             "randomization on the test set",
                             randomized_mean_error_baseline_test/\
                             replaced_mean_error_baseline_test)
                else:
                    logger.debug("Substitution is inf times better "+\
                            "than randomization on the test set")
                logger.debug("Evaluating randomized model test")
                results["features"][f]["randomized_rec_eval_test"] =\
                    common_utils.evaluate_recommender(baseline_predictions_test,\
                        randomized_predictions_test, logger, args.nbins,
                        "Randomized feature {} test".format(f))
        logger.debug("persist_dir: {}".format(args.persist_dir))
        if args.persist_dir is not None:
            logger.debug("Saving the state of current iteration")
            fname = os.path.join(args.persist_dir, "results.pkl")
            logger.debug("Saving %s", fname)
            ofile = open(fname, "wb")
            pickle.dump(results, ofile)
            ofile.close()
            fname = os.path.join(args.persist_dir,
                                 "lr_model_{}.pkl".format(f))
            if os.path.exists(fname):
                logger.debug("%a already exists, removing")
                shutil.rmtree(fname)
            logger.debug("Saving %s", fname)
            lr_model.save(sc, fname)
            fname = os.path.join(args.persist_dir,
                                 "predictions_{}.pkl".format(f))
            logger.debug("Saving %s", fname)
            predictions_training_c = predictions_training.collect()
            if train_ratio > 0:
                 predictions_test_c = predictions_test.collect()
            else:
                 predictions_test_c = None
            ofile = open(fname, "wb")
            pickle.dump((predictions_training_c, predictions_test_c), ofile)
            ofile.close()
    if train_ratio <= 0:
        training_movies = all_movies

    replaced_mean_error_baseline, replaced_rec_eval =\
                compare_with_all_replaced_features(sc, features_original_training, other_features,\
            user_or_product_features, all_predicted_features, rank,\
            baseline_predictions_training, logger, power, args)
    results["all_replaced_mean_error_baseline"] = replaced_mean_error_baseline
    results["all_replaced_rec_eval"] = replaced_rec_eval
    results["all_random_rec_eval"] = compare_with_all_randomized(\
        sc, model, rank, training_movies, baseline_predictions_training,\
        logger, power, args)

    if train_ratio > 0:
        replaced_mean_error_baseline, replaced_rec_eval =\
            compare_with_all_replaced_features(sc, features_original_test, other_features,\
                user_or_product_features, all_predicted_features_test, rank,\
                baseline_predictions_test, logger, power, args)
        results["all_replaced_mean_error_baseline_test"] = replaced_mean_error_baseline
        results["all_replaced_rec_eval_test"] = replaced_rec_eval
        results["all_random_rec_eval_test"] = compare_with_all_randomized(\
            sc, model, rank, test_movies, baseline_predictions_test,\
            logger, power, args)
    if args.persist_dir is not None:
        fname = os.path.join(args.persist_dir, "results.pkl")
        logger.debug("Writing %s", fname)
        ofile = open(fname, "wb")
        pickle.dump(results, ofile)
        ofile.close

    return results

def display_internal_feature_predictor(results, logger, no_ht=False):
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
              "Mean error"]
    if not no_ht:
        header +=[
              "Mean absolute error HT",
              "MRAE HT"]
    if results["train_ratio"] > 0:
        header += ["MRAE test",
                   "Mean absolute error test",
                   "Mean error test"]
        if not no_ht:
            header += [
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
               r["regression_evaluation"]["mean_err"]]
        if not no_ht:
            row += [
               r["regression_evaluation_ht"]["mre"],
               r["regression_evaluation_ht"]["mrae"]]
        if results["train_ratio"] > 0:
            row +=  [r["regression_evaluation_test"]["mrae"],
                     r["regression_evaluation_test"]["mre"],
                     r["regression_evaluation_test"]["mean_err"]]
            if not no_ht:
                row += [
                     r["regression_evaluation_ht_test"]["mre"],
                     r["regression_evaluation_ht_test"]["mrae"]]
        row += [results["mean_feature_values"][f],
                r["replaced_mean_error_baseline"],
                r["randomized_mean_error_baseline"]]
        if r["replaced_mean_error_baseline"] != 0:
            row += [float(r["randomized_mean_error_baseline"])/\
                r["replaced_mean_error_baseline"]]
        else:
            row += ["inf"]
        if results["train_ratio"] > 0:
            row += [r["replaced_mean_error_baseline_test"],
                    r["randomized_mean_error_baseline_test"]]
            if r["replaced_mean_error_baseline_test"] != 0:
                row += [float(r["randomized_mean_error_baseline_test"])/\
                            r["replaced_mean_error_baseline_test"]]
            else:
                row += ["inf"]
        table.add_row(row)
    logger.info("\n" + str(table))
