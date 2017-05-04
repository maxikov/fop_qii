#standard library
import time
import math
import random
import os.path
import pickle

#prettytable library
from prettytable import PrettyTable

#pyspark library
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel

#project files
import internal_feature_predictor
import AverageRatingRecommender
import common_utils
import parsers_and_loaders


def metadata_predictor(sc, training, rank, numIter, lmbda,
                       args, all_movies, metadata_sources,
                       logger, train_ratio = 0):
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

    if args.persist_dir is not None:
        fname = os.path.join(args.persist_dir, "als_model.pkl")
        if not os.path.exists(fname):
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
        logger.debug("Saving model to %s", fname)
        model.save(sc, fname)

    features = model.productFeatures()
    other_features = model.userFeatures()
    if use_user_features:
        features, other_features = other_features, features

    if args.persist_dir is not None:
        fname = os.path.join(args.persist_dir, "features_indicators.pkl")
        if os.path.exists(fname):
            logger.debug("Loading %s", fname)
            ifile = open(fname, "rb")
            objects = pickle.load(ifile)
            ifile.close()
            (indicators_c, nof, categorical_features, feature_names,
                  features_c, other_features_c, features_training_c,
                  indicators_training_c, features_test_c, indicators_test_c,
                  all_movies, n_movies, train_ratio, training_movies,
                  test_movies) = objects
            indicators = sc.parallelize(indicators_c)\
                    .repartition(args.num_partitions)\
                    .cache()
            features = sc.parallelize(features_c)\
                    .repartition(args.num_partitions)\
                    .cache()
            other_features = sc.parallelize(other_features_c)\
                    .repartition(args.num_partitions)\
                    .cache()
            indicators_training = sc.parallelize(indicators_training_c)\
                    .repartition(args.num_partitions)\
                    .cache()
            features_training = sc.parallelize(features_training_c)\
                    .repartition(args.num_partitions)\
                    .cache()
            if features_test_c is not None:
                features_test = sc.parallelize(features_test_c)\
                    .repartition(args.num_partitions)\
                    .cache()
            else:
                features_test = None
            if indicators_test_c is not None:
                indicators_test = sc.parallelize(indicators_test_c)\
                    .repartition(args.num_partitions)\
                    .cache()
            else:
                indicators_test = None
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
        if args.drop_rare_movies > 0:
            logger.debug("Dropping movies with fewer than %d non-zero "+\
                    "features", args.drop_rare_movies)
            features, indicators = common_utils.drop_rare_movies(features,
                                                                 indicators,
                                                                 args.drop_rare_movies)
            logger.debug("%d movies left", features.count())
        all_movies = features.keys().collect()
        n_movies = len(all_movies)
        if train_ratio > 0:
            logger.debug("Building training and test sets for regression")
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
            features_test = None
            indicators_test = None
            test_movies = None
            training_movies = all_movies


    if write_model:
        logger.debug("Writing %s", fname)
        indicators_c = indicators.collect()
        features_c = features.collect()
        other_features_c = other_features.collect()
        features_training_c = features_training.collect()
        indicators_training_c = indicators_training.collect()
        if features_test is not None:
            features_test_c = features_test.collect()
        else:
            features_test_c = None
        if indicators_test is not None:
            indicators_test_c = indicators_test.collect()
        else:
            indicators_test_c = None
        objects = (indicators_c, nof, categorical_features, feature_names,
                  features_c, other_features_c, features_training_c,
                  indicators_training_c, features_test_c, indicators_test_c,
                  all_movies, n_movies, train_ratio, training_movies,
                  test_movies)
        ofile = open(fname, "wb")
        pickle.dump(objects, ofile)
        ofile.close()

    results, store = common_utils.load_if_available(args.persist_dir,
                                                    "results.pkl",
                                                    logger)
    if args.persist_dir is not None:
        store = True
    if results is None:
        results = {}
        results["train_ratio"] = train_ratio
        results["features"] = {}
        results["nof"] = nof
        results["categorical_features"] = categorical_features
        results["mean_feature_values"] = common_utils.mean_feature_values(features,
            logger)
        results["feature_ranges"] = common_utils.feature_ranges(features, logger)
        results["mean_indicator_values"] = common_utils.mean_feature_values(\
            indicators, logger)
        results["indicator_ranges"] = common_utils.feature_ranges(indicators, logger)
    common_utils.save_if_needed(args.persist_dir, "results.pkl",
                    results, store, logger)

    for f in xrange(nof):
        logger.debug("Processing {} ({} out of {})"\
                .format(feature_names[f], f, nof))
        if f in results["features"]:
            logger.debug("Already computed, skipping")
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
        qii = common_utils\
                  .compute_regression_qii(lr_model,
                                          features_training,
                                          indicators_training\
                                                  .map(lambda (mid, ftrs):
                                                       (ftrs[f])),
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
            predictions = common_utils.safe_zip(ids_test, lr_model\
                                            .predict(input_test)\
                                            .map(float))
            observations = indicators_test.map(lambda (mid, ftrs):
                    (mid, float(ftrs[f])))

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
        common_utils.save_if_needed(args.persist_dir, "results.pkl",
                    results, store, logger)

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

