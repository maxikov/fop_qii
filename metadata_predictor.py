#standard library
import time
import math
import random

#prettytable library
from prettytable import PrettyTable

#pyspark library
from pyspark.mllib.recommendation import ALS

#project files
import internal_feature_predictor
import AverageRatingRecommender
import common_utils
import parsers_and_loaders


def metadata_predictor(sc, training, rank, numIter, lmbda,
                       args, all_movies, metadata_sources,
                       logger, train_ratio = 0):
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

    use_user_features = ("users" in args.metadata_sources)


    cur_mtdt_srcs = filter(lambda x: x["name"] in args.metadata_sources, metadata_sources)
    if args.drop_missing_movies:
        indicators, nof, categorical_features, feature_names =\
            internal_feature_predictor.build_meta_data_set(sc, cur_mtdt_srcs, None, logger)
        old_all_movies = all_movies
        all_movies = set(indicators.keys().collect())
        logger.debug("{} movies loaded, data for {} is missing, purging them"\
                .format(len(all_movies), len(old_all_movies - all_movies)))
        training = training.filter(lambda x: x[1] in all_movies)
        logger.debug("{} items left in the training set"\
                .format(training.count()))
    else:
        indicators, nof, categorical_features, feature_names =\
            internal_feature_predictor.\
            build_meta_data_set(sc, cur_mtdt_srcs, all_movies, logger)
    logger.debug("%d features loaded", nof)
    if args.drop_rare_features > 0:
        indicators, nof, categorical_features, feature_names =\
            internal_feature_predictor.drop_rare_features(indicators, nof, categorical_features,
                               feature_names, args.drop_rare_features,
                               logger)

    logger.debug("Training ALS recommender")
    start = time.time()
    model = ALS.train(training, rank=rank, iterations=numIter,
                      lambda_=lmbda, nonnegative=args.non_negative)
    logger.debug("Done in %f seconds", time.time() - start)

    features = model.productFeatures()
    other_features = model.userFeatures()
    if use_user_features:
        features, other_features = other_features, features

    results["train_ratio"] = train_ratio
    results["features"] = {}
    results["nof"] = nof
    results["categorical_features"] = categorical_features

    for f in xrange(nof):
        logger.debug("Processing {} ({} out of {})"\
                .format(feature_names[f], f, nof))
        results["features"][f] = {}
        results["features"][f]["name"] = feature_names[f]
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
                    training_movies)
            features_test = features.filter(lambda x: x[0] in test_movies)
            indicators_training = indicators.filter(lambda x: x[0] in
                    training_movies)
            indicators_test = indicators.filter(lambda x: x[0] in test_movies)
        else:
            features_training = features
            indicators_training = indicators

        if f in categorical_features:
            regression_model = "logistic"
            results["features"][f]["type"] = "classification"
        else:
            regression_model = "linear"
            results["features"][f]["type"] = "regression"

        lr_model, observations, predictions = internal_feature_predictor.predict_internal_feature(\
            indicators_training,
            features_training,
            f,
            regression_model,
            {},
            args.nbins,
            logger,
            True)

        weights = list(lr_model.weights)
        logger.debug("Model weights: {}".format(weights))
        results["features"][f]["weights"] = weights

        if f in categorical_features:
            results["features"][f]["eval"] =\
                common_utils.evaluate_binary_classifier(predictions,
                                                        observations,
                                                        logger)
        else:
            _min = indicators_training.map(lambda x: x[1][f]).min()
            _max = indicators_training.map(lambda x: x[1][f]).max()
            bin_range = (_min, _max)
            logger.debug("Bin range: {}".format(bin_range))
            reg_eval =\
                common_utils.evaluate_regression(predictions,
                                                 observations,
                                                logger,
                                                args.nbins,
                                                bin_range)
            results["features"][f]["eval"] = reg_eval

        if train_ratio > 0:
            logger.debug("Computing predictions on the test set")
            input_test = features_test.values()
            ids_test = features_test.keys()
            predictions = ids_test.zip(lr_model\
                                            .predict(input_test)\
                                            .map(float))
            observations = indicators_test.map(lambda (mid, ftrs):
                    (mid, float(ftrs[f])))
            if f in categorical_features:
                results["features"][f]["eval_test"] =\
                    common_utils.evaluate_binary_classifier(predictions,
                                                            observations,
                                                            logger)
            else:
                _min = indicators_test.map(lambda x: x[1][f]).min()
                _max = indicators_test.map(lambda x: x[1][f]).max()
                bin_range = (_min, _max)
                logger.debug("Bin range: {}".format(bin_range))
                reg_eval =\
                    common_utils.evaluate_regression(predictions,
                                                     observations,
                                                     logger,
                                                     args.nbins,
                                                     bin_range)
                results["features"][f]["eval_test"] = reg_eval

    return results


def display_regression_results(results, logger):
    rf = results["features"]
    rf = [(f, info) for (f, info) in rf.items() if info["type"]=="regression"]
    rf.sort(key=lambda x: -x[1]["eval"]["mrae"])

    header = ["Feature",
              "MRAE",
              "Mean abs err"]
    if results["train_ratio"] > 0:
        header += ["MRAE test",
                   "Mean abs err test"]
    table = PrettyTable(header)
    for f, info in rf:
        row = [info["name"], info["eval"]["mrae"], info["eval"]["mrae"]]
        if results["train_ratio"] > 0:
            row+= [info["eval_test"]["mrae"],
                      info["eval_test"]["mre"]]
        table.add_row(row)
    logger.info("\n{}".format(table))

def display_classification_results_no_threshold(results, logger):
    rf = results["features"]
    rf = [(f, info) for (f, info) in rf.items() if
            info["type"]=="classification"]
    rf.sort(key=lambda x: -x[1]["eval"]["better"])

    header = ["Feature",
              "AUPRC",
              "Prate",
              "x better"]
    if results["train_ratio"] > 0:
        header += ["AUPRC test",
                   "Prate test",
                   "x better test"]
    table = PrettyTable(header)
    for f, info in rf:
        row = [info["name"], info["eval"]["auprc"],
                info["eval"]["prate"],
                info["eval"]["better"]]
        if results["train_ratio"] > 0:
            row += [info["eval_test"]["auprc"],
                info["eval_test"]["prate"],
                info["eval_test"]["better"]]
        table.add_row(row)
    logger.info("\n{}".format(table))

def display_metadata_predictor(results, logger):
    logger.info("Overall results dict: {}".format(results))
    if results["train_ratio"] > 0:
        logger.info("Train ratio: %f %%", results["train_ratio"])
    if len(results["categorical_features"]) > 0:
        display_classification_results_no_threshold(results, logger)
    if len(results["categorical_features"]) < results["nof"]:
        display_regression_results(results, logger)

