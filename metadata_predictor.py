#standard library
import time
import math
import random
import os.path
import pickle
import functools

#prettytable library
from prettytable import PrettyTable

#pyspark library
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel
import pyspark

#project files
import internal_feature_predictor
import AverageRatingRecommender
import common_utils
import parsers_and_loaders


def metadata_predictor(sc, training, rank, numIter, lmbda,
                       args, all_movies, metadata_sources,
                       logger, train_ratio = 0):
    user_or_product_features = "product"
    logger.debug("Started metadata_predictor")
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

    model = internal_feature_predictor.load_or_train_ALS(training, rank, numIter, lmbda, args, sc, logger)

    features = model.productFeatures()
    other_features = model.userFeatures()
    logger.debug("user_or_product_features: {}"\
            .format(user_or_product_features))
    if user_or_product_features == "user":
        features, other_features = other_features, features
    features_original = features


    indicators, nof, categorical_features, feature_names, all_movies, training=\
        internal_feature_predictor.load_metadata_process_training(sc, args, metadata_sources, training,
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
        logger.debug("{} items left in the training set"\
            .format(training.count()))
        all_movies = list(all_movies)

    if args.normalize:
        indicators = internal_feature_predictor.normalize_features(indicators, categorical_features,
                feature_names, logger)

    results["feature_names"] = feature_names
    results["categorical_features"] = categorical_features


    results = internal_feature_predictor.compute_or_load_mean_feature_values(args, features, results, logger)

    results["train_ratio"] = train_ratio


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
                  internal_feature_predictor.split_or_load_training_test_sets(train_ratio, all_movies, features,
                                     indicators, features_original, n_movies,
                                     args, logger, sc)
         filter_f = functools.partial(lambda training_movies, x: x[1] in
                 training_movies, training_movies)
         training_training = training.filter(filter_f)
         filter_f = functools.partial(lambda test_movies, x: x[1] in
                 test_movies, test_movies)
         training_test = training.filter(filter_f)

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
         training_training = training
         trainin_test = None






    all_lr_models = {}
    for f in xrange(nof):
        logger.debug("Processing {} ({} out of {})"\
                .format(feature_names[f], f, nof))
        if "features" not in results:
            logger.debug("Features dict not found, adding")
            results["features"] = {}
        if f in results["features"]:
            fname = os.path.join(args.persist_dir,
                                 "lr_model_{}.pkl".format(f))
            logger.debug("Already processed, loading %s", fname)
            lr_model = models_by_name[args.regression_model].load(sc, fname)
            all_lr_models[f] = lr_model
            continue



        results["features"][f] = {}
        results["features"][f]["name"] = feature_names[f]
        linlog = args.regression_model in ["logistic", "linear"]
        if not linlog:
            results["not_linlog"] = True #ONLY SET IF TRUE!!!
        regression_model = args.regression_model
        if f in categorical_features:
            if linlog:
               regression_model = "logistic"
            results["features"][f]["type"] = "classification"
        else:
            if linlog:
               regression_model = "linear"
            results["features"][f]["type"] = "regression"

        lr_model, observations, predictions = internal_feature_predictor.predict_internal_feature(\
            features=indicators_training,
            indicators=features_training,
            f=f,
            regression_model=regression_model,
            categorical_features={},
            max_bins=args.nbins,
            logger=logger,
            no_threshold=False,
            is_classifier=(f in categorical_features),
            num_classes=(categorical_features[f] if f in categorical_features
                else None))
        all_lr_models[f] = lr_model

        if args.regression_model == "regression_tree":
            logger.info(lr_model.toDebugString())
        elif regression_model == "logistic":
            threshold = lr_model.threshold
            logger.debug("Model threshold %f", threshold)
            logger.debug("Clearing")
            lr_model.clearThreshold()
            logger.debug("Making no-threshold predictions")
            input_training = features_training.values()
            ids_training = features_training.keys()
            predictions_nt = common_utils.safe_zip(ids_training, lr_model\
                                            .predict(input_training)\
                                            .map(float))
            logger.debug("Restoring threshold")
            lr_model.setThreshold(threshold)
        select_cur_feature_map_f = functools.partial(lambda f, (mid, ftrs):
                (ftrs[f]), f)
        qii = common_utils\
                  .compute_regression_qii(lr_model,
                                          features_training,
                                          indicators_training\
                                                  .map(select_cur_feature_map_f),
                                          logger,
                                          predictions, rank)
        results["features"][f]["qii"] = qii
        if linlog and not args.force_qii:
            weights = list(lr_model.weights)
        else:
            weights = qii

        logger.debug("Model weights: {}".format(weights))
        results["features"][f]["weights"] = weights

        if f in categorical_features:
            results["features"][f]["eval"] =\
                common_utils.evaluate_binary_classifier(predictions,
                                                        observations,
                                                        logger,
                                                        False)
            if regression_model == "logistic":
                results["features"][f]["eval_nt"] =\
                    common_utils.evaluate_binary_classifier(predictions_nt,
                                                            observations,
                                                            logger,
                                                            True)
        else:
            map_f = functools.partial(lambda f, x: x[1][f], f)
            _min = indicators_training.map(map_f).min()
            _max = indicators_training.map(map_f).max()
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
            predictions = common_utils.safe_zip(ids_test, lr_model\
                                            .predict(input_test)\
                                            .map(float))
            map_f = functools.partial(lambda f, (mid, ftrs):
                    (mid, float(ftrs[f])), f)
            observations = indicators_test.map(map_f)

            if f in categorical_features:
                results["features"][f]["eval_test"] =\
                    common_utils.evaluate_binary_classifier(predictions,
                                                            observations,
                                                            logger,
                                                            False)
                if regression_model == "logistic":
                    logger.debug("Clearing threshold")
                    lr_model.clearThreshold()
                    logger.debug("Making no-threshold test predictions")
                    predictions_nt_test = common_utils.safe_zip(ids_test, lr_model\
                                            .predict(input_test)\
                                            .map(float))
                    logger.debug("Restoring threshold")
                    lr_model.setThreshold(threshold)
                    results["features"][f]["eval_test_nt"] =\
                        common_utils.evaluate_binary_classifier(predictions_nt_test,
                                                               observations,
                                                                logger,
                                                                True)
            else:
                map_f = functools.partial(lambda f, x: x[1][f], f)
                _min = indicators_training.map(map_f).min()
                _max = indicators_training.map(map_f).max()
                bin_range = (_min, _max)
                logger.debug("Bin range: {}".format(bin_range))
                reg_eval =\
                    common_utils.evaluate_regression(predictions,
                                                     observations,
                                                     logger,
                                                     args.nbins,
                                                     bin_range)
                results["features"][f]["eval_test"] = reg_eval

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
            ofile.close()



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

def display_classification_results_threshold(results, logger):
    rf = results["features"]
    rf = [(f, info) for (f, info) in rf.items() if
            info["type"]=="classification"]
    rf.sort(key=lambda x: -x[1]["eval"]["recall"])

    header = ["Feature",
              "Precision",
              "Recall"]
    if results["train_ratio"] > 0:
        header += ["Precision test",
                   "Recall test"]
    table = PrettyTable(header)
    for f, info in rf:
        row = [info["name"], info["eval"]["precision"],
                info["eval"]["recall"]]
        if results["train_ratio"] > 0:
            row += [info["eval_test"]["precision"],
                info["eval_test"]["recall"]]
        table.add_row(row)
    logger.info("\n{}".format(table))

def display_classification_results_no_threshold(results, logger):
    rf = results["features"]
    rf = [(f, info) for (f, info) in rf.items() if
            info["type"]=="classification"]
    rf.sort(key=lambda x: -x[1]["eval_nt"]["better"])

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
        row = [info["name"], info["eval_nt"]["auprc"],
                info["eval_nt"]["prate"],
                info["eval_nt"]["better"]]
        if results["train_ratio"] > 0:
            row += [info["eval_test_nt"]["auprc"],
                info["eval_test_nt"]["prate"],
                info["eval_test_nt"]["better"]]
        table.add_row(row)
    logger.info("\n{}".format(table))

def display_metadata_predictor(results, logger):
    logger.info("Overall results dict: {}".format(results))
    if results["train_ratio"] > 0:
        logger.info("Train ratio: %f %%", results["train_ratio"])
    if len(results["categorical_features"]) > 0:
        if "not_linlog" in results:
            display_classification_results_threshold(results, logger)
        else:
            display_classification_results_no_threshold(results, logger)
    if len(results["categorical_features"]) < results["nof"]:
        display_regression_results(results, logger)

