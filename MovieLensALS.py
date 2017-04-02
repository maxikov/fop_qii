#!/usr/bin/env python

#Standard library
import logging
import sys
import itertools
import copy
import random
from math import sqrt
from os.path import join, isfile, dirname
from collections import defaultdict
import time
import argparse
import math
import numpy
import itertools
import StringIO
import csv
import traceback

#prettytable library
from prettytable import PrettyTable

#matplotlib library
import matplotlib.pyplot as plt

#pyspark library
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS

#project files
import qii_ls_legacy
import parsers_and_loaders
import internal_feature_predictor
import metadata_predictor
import feature_global_influence

def args_init(logger):
    parser = argparse.ArgumentParser(description=u"Usage: " +\
            "/path/to/spark/bin/spark-submit --driver-memory 2g " +\
            "MovieLensALS.py [arguments]")

    parser.add_argument("--non-negative", action="store_true", help=\
            "Use non-negative factrorization for ALS")

    parser.add_argument("--rank", action="store", default=12, type=int,
            help="Rank for ALS algorithm. 12 by default")

    parser.add_argument("--lmbda", action="store", default=0.1, type=float,
            help="Lambda for ALS algorithm. 0.1 by default")

    parser.add_argument("--num-iter", action="store", default=20, type=int,
            help="Number of iterations for ALS algorithm. 20 by default")

    parser.add_argument("--num-partitions", action="store", default=4,
            type=int, help="Number of partitions for the RDD. 4 by default")

    parser.add_argument("--data-path", action="store",
            default="datasets/ml-1m/", type=str, help="Path to MovieLens " +\
                    "home directory. datasets/ml-1m/ by default")

    parser.add_argument("--checkpoint-dir", action="store",
            default="checkpoint", type=str, help="Path to checkpoint " +\
                    "directory. checkpoint by default")

    parser.add_argument("--regression-model", action="store", type=str,
            default="linear", help="Model used in genres-regression, "+\
                    "Possible values: linear, "+\
                    "regression_tree, "+\
                    "random_forest, "+\
                    "linear by default")
    parser.add_argument("--nbins", action="store", type=int, default=32, help=\
            "Number of bins for a regression tree. 32 by default. "+\
            "Maximum depth is ceil(log(nbins, 2)).")
 
    parser.add_argument("--regression-users", action="store_true", help=\
            "Predicting internal features based on user metadata")

    parser.add_argument("--predict-product-features", action="store_true",
            help = "Use regression to predict product features "+\
                    "based on product metadata")

    parser.add_argument("--metadata-sources", action = "store", type = str,
            nargs = "+", help = "Sources for user or product metadata "+\
                    "for feature explanations. Possible values: years, "+\
                    "genres, tags, average_rating, imdb_keywords.")

    args = parser.parse_args()

    logger.debug("rank: {}, lmbda: {}, num_iter: {}, num_partitions: {}".format(
        args.rank, args.lmbda, args.num_iter, args.num_partitions))
    logger.debug("data_path: {}, checkpoint_dir: {}".format(args.data_path,
        args.checkpoint_dir))

    logger.debug("regression_model: {}".format(args.regression_model))

    logger.debug("nbins: {}".format(args.nbins))
    logger.debug("regression_users: {}".format(args.regression_users))
    logger.debug("predict_product_features: {}"\
        .format(args.predict_product_features))
    logger.debug("metadata_sources: {}".format(args.metadata_sources))

    return args

def logger_init():

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s '+\
        '- %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

if __name__ == "__main__":

    logger = logger_init()
    args = args_init(logger)

    # set up environment
    conf = SparkConf() \
      .setMaster("local[*]") \
      .setAppName("MovieLensALS") \
      .set("spark.executor.memory", "2g")
    sc = SparkContext(conf=conf)


    sc.setCheckpointDir(args.checkpoint_dir)
    ALS.checkpointInterval = 2


    if "ml-20m" in args.data_path:
        sep = ","
        extension = ".csv"
        remove_first_line = True
    else:
        sep = "::"
        extension = ".dat"
        remove_first_line = False


    logger.debug("Loading ratings")
    start = time.time()
    ratings_rdd = sc.parallelize(
            parsers_and_loaders.loadCSV(
                join(args.data_path, "ratings" + extension),
                remove_first_line = remove_first_line
                )
            )
    ratings = ratings_rdd.map(lambda x: parsers_and_loaders.parseRating(x,
         sep=sep))
    logger.debug("Done in {} seconds".format(time.time() - start))

    logger.debug("Loading movies")
    start = time.time()
    movies_rdd = sc.parallelize(
            parsers_and_loaders.loadCSV(
                join(args.data_path, "movies" + extension),
                remove_first_line = remove_first_line
                )
            )
    movies = dict(movies_rdd.map(lambda x: parsers_and_loaders.parseMovie(x,
        sep=sep)).collect())
    all_movies = set(movies.keys())
    logger.debug("Done in {} seconds".format(time.time() - start))
    logger.debug("{} movies loaded".format(len(all_movies)))


    metadata_sources = [
        {
            "name": "years",
            "src_rdd": (lambda: movies_rdd),
            "loader": (lambda x: parsers_and_loaders.load_years(x, sep))
        },
        {
            "name": "genres",
            "src_rdd": (lambda: movies_rdd),
            "loader": (lambda x: parsers_and_loaders.load_genres(x,
                sep, parsers_and_loaders.parseGenre))
        },
        {
            "name": "tags",
            "src_rdd": (
                lambda: sc.parallelize(
                    parsers_and_loaders.loadCSV(
                        join(args.data_path, "tags" + extension),
                        remove_first_line = remove_first_line
                    )
                )
            ),
            "loader": (lambda x: parsers_and_loaders.load_tags(x, sep))
        },
        {
            "name": "imdb_keywords",
            "src_rdd": (
                lambda: sc.parallelize(
                    parsers_and_loaders.loadCSV(
                        join(args.data_path, "ml-20m.imdb.small.csv"),
                        remove_first_line = True
                    )
                )
            ),
            "loader": (lambda x: parsers_and_loaders.load_genres(x, sep=",",
                parser_function=parsers_and_loaders.parseIMDBKeywords))
        }
    ]



    training = ratings\
      .filter(lambda x: (x[1][1] in all_movies) and (True or x[0] < 3))\
      .values() \
      .repartition(args.num_partitions) \
      .cache()

    logger.debug("{} records in the training set".format(training.count()))
    logger.debug("{} unique movies in the training set"\
            .format(len(set(training.map(lambda x: x[1]).collect()))))

    if args.regression_users:
        args.metadata_sources = ["users"]
        logger.debug("Loading users")
        start = time.time()
        users_rdd = sc.parallelize(
            parsers_and_loaders.loadCSV(
                join(args.data_path, "users" + extension),
                remove_first_line = remove_first_line
                )
            )
        users = parsers_and_loaders.load_users(users_rdd, sep)
        all_users = set(users[0].keys().collect())
        logger.debug("Done in {} seconds".format(time.time() - start))
        metadata_sources = [
            {
                "name": "users",
                "src_rdd": (lambda: users),
                "loader": (lambda x: x)
            }]
        results = internal_feature_predictor.internal_feature_predictor(sc,
            training, args.rank,
            args.num_iter, args.lmbda,
            args, all_users, metadata_sources,
            user_or_product_features="user", eval_regression = True,
            compare_with_replaced_feature = True,
            compare_with_randomized_feature = True, logger = logger)

        internal_feature_predictor.display_internal_feature_predictor(
            results, logger)

    elif args.predict_product_features:
        results = internal_feature_predictor.internal_feature_predictor(
            sc, training, args.rank,
            args.num_iter, args.lmbda,
            args, all_movies, metadata_sources,
            user_or_product_features="product", eval_regression = True,
            compare_with_replaced_feature = True,
            compare_with_randomized_feature = True, logger = logger)

        internal_feature_predictor.display_internal_feature_predictor(
           results, logger)

    sc.stop()
