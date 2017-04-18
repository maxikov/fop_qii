#!/usr/bin/env python

#Standard library
import logging
import sys
from os.path import join
import time
import argparse

#pyspark library
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS

#project files
import parsers_and_loaders
import internal_feature_predictor

def args_init(logger):
    """
    Sets up command line arguments and outputs the values.
    """

    parser = argparse.ArgumentParser(description=u"Usage: " +\
            "/path/to/spark/bin/spark-submit --driver-memory 2g " +\
            "MovieLensALS.py [arguments]")

    parser.add_argument("--local-threads", action="store", type=str,
                        default="*", help="Argument passed to "+\
                                          ".setMaster(\"local[<ARG>]\"). "+\
                                          "* by default.")

    parser.add_argument("--spark-executor-memory", action="store",
                        type=str, default="16g")

    parser.add_argument("--non-negative", action="store_true", help=\
            "Use non-negative factrorization for ALS")

    parser.add_argument("--rank", action="store", default=12, type=int,
                        help="Rank for ALS algorithm. 12 by default")

    parser.add_argument("--lmbda", action="store", default=0.1, type=float,
                        help="Lambda for ALS algorithm. 0.1 by default")

    parser.add_argument("--num-iter", action="store", default=20, type=int,
                        help="Number of iterations for ALS algorithm. 20 by"+\
                             " default")

    parser.add_argument("--num-partitions", action="store", default=4,
                        type=int, help="Number of partitions for the RDD. "+\
                                       "4 by default")

    parser.add_argument("--data-path", action="store",
                        default="datasets/ml-1m/", type=str,
                        help="Path to MovieLens " +\
                             "home directory. datasets/ml-1m/ by default")

    parser.add_argument("--checkpoint-dir", action="store",
                        default="checkpoint", type=str,
                        help="Path to checkpoint " +\
                             "directory. checkpoint by default")

    parser.add_argument("--regression-model", action="store", type=str,
                        default="linear",
                        help="Model used in genres-regression, "+\
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
                        help="Use regression to predict product features "+\
                             "based on product metadata")

    parser.add_argument("--metadata-sources", action="store", type=str,
                        nargs="+",
                        help="Sources for user or product metadata "+\
                             "for feature explanations. Possible values: "+\
                             "years, genres, tags, average_rating, "+\
                             "imdb_keywords, tvtropes.")

    parser.add_argument("--movies-file", action="store", type=str,
                        default="movies", help="File from which to read "+\
                                "movie metadata. By default looks for "+\
                                "movies.(csv|dat) in the root data set "+\
                                "directory.")

    parser.add_argument("--cross-validation", action="store", type=int,
                        default=0, help="Percent of the data points that "+\
                                        "is to be used in the training "+\
                                        "set during cross-valudation. "+\
                                        "If 0 (default), no cross-"+\
                                        "validation is performed")

    parser.add_argument("--tvtropes-file", action="store", type=str)
    parser.add_argument("--features-trim-percentile", action="store",
                        type=int, help="Leave only the specified "+\
                                       "percentage of the internal "+\
                                       "feature distribution before "+\
                                       "running regression. Tails will "+\
                                       "be trimmed symmerically. If 0 "+\
                                       "(default) no trimming is done.",
                        default=0)

    parser.add_argument("--drop-missing-movies", action="store_true",
                        help="When there's no metadata for a given movie "+\
                             "(even in just one source), drop it from "+\
                             "the data set entirely instead of adding "+\
                             "empty records for it.")

    parser.add_argument("--drop-rare-features", action="store", type=int,
                        default=0, help="Drop features from the meta "+\
                                        "data set that have fewer than "+\
                                        "the specified value of non-zero "+\
                                        "values. 0 by default (no dropping).")

    args = parser.parse_args()

    logger.debug("rank: {}, lmbda: {}, num_iter: {}, num_partitions: {}"\
        .format(args.rank, args.lmbda, args.num_iter, args.num_partitions))
    logger.debug("data_path: {}, checkpoint_dir: {}".format(args.data_path,\
        args.checkpoint_dir))
    logger.debug("local_threads: {}".format(args.local_threads))
    logger.debug("spark_executor_memory: {}"\
            .format(args.spark_executor_memory))

    logger.debug("regression_model: {}".format(args.regression_model))

    logger.debug("nbins: {}".format(args.nbins))
    logger.debug("regression_users: {}".format(args.regression_users))
    logger.debug("predict_product_features: {}"\
        .format(args.predict_product_features))
    logger.debug("metadata_sources: {}".format(args.metadata_sources))
    logger.debug("movies_file: {}".format(args.movies_file))
    logger.debug("cross_validation: {}".format(args.cross_validation))
    logger.debug("tvtropes_file: {}".format(args.tvtropes_file))
    logger.debug("features_trim_percentile: {}"\
            .format(args.features_trim_percentile))
    logger.debug("drop_missing_movies: {}".format(args.drop_missing_movies))
    logger.debug("drop_rare_features: {}".format(args.drop_rare_features))

    return args

def logger_init():
    """
    Initializes logger object for logging.
    """

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s '+\
        '- %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def main():
    logger = logger_init()
    args = args_init(logger)

    # set up environment
    conf = SparkConf() \
      .setMaster("local[{}]".format(args.local_threads)) \
      .setAppName("MovieLensALS") \
      .set("spark.executor.memory", args.spark_executor_memory)\
      .set("spark.driver.maxResultSize", "4g")\
      .set("spark.python.worker.memory", "4g")\
      .set("spark.network.timeout", "360000s")\
      .set("spark.rpc.numRetries", "50")\
      .set("spark.cores.max", "8")\
      .set("spark.default.parallelism", str(args.num_partitions))\
      .set("spark.local.dir", args.checkpoint_dir)
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

    if args.movies_file == "movies":
        movies_file = join(args.data_path, "movies" + extension)
        msep = sep
        movies_remove_first_line = remove_first_line
    else:
        movies_file = args.movies_file
        if ".csv" in movies_file:
            msep = ","
            movies_remove_first_line = True
    logger.debug("msep: {}".format(msep))

    logger.debug("Loading ratings")
    start = time.time()
    ratings_rdd = sc.parallelize(
        parsers_and_loaders.loadCSV(
            join(args.data_path, "ratings" + extension),
            remove_first_line=remove_first_line
            )
        )
    ratings = ratings_rdd.map(lambda x: parsers_and_loaders.parseRating(x,\
         sep=sep))
    logger.debug("Done in %f seconds", time.time() - start)

    logger.debug("Loading movies")
    start = time.time()
    movies_rdd = sc.parallelize(
        parsers_and_loaders.loadCSV(
            movies_file,
            remove_first_line=movies_remove_first_line
            )
        )
    movies = dict(movies_rdd.map(lambda x: parsers_and_loaders.parseMovie(x,\
        sep=msep)).collect())
    all_movies = set(movies.keys())
    logger.debug("Done in %f seconds", time.time() - start)
    logger.debug("%d movies loaded", len(all_movies))

    metadata_sources = [
        {
            "name": "years",
            "src_rdd": (lambda: movies_rdd),
            "loader": (lambda x: parsers_and_loaders.load_years(x, msep))
        },
        {
            "name": "genres",
            "src_rdd": (lambda: movies_rdd),
            "loader": (lambda x: parsers_and_loaders.load_genres(x,\
                msep, parsers_and_loaders.parseGenre))
        },
        {
            "name": "tags",
            "src_rdd": (
                lambda: sc.parallelize(
                    parsers_and_loaders.loadCSV(
                        join(args.data_path, "tags" + extension),
                        remove_first_line=remove_first_line
                    )
                )
            ),
            "loader": (lambda x: parsers_and_loaders.load_tags(x, sep,
                       prefix="movielens_tags"))
        },
        {
            "name": "imdb_keywords",
            "src_rdd": (lambda: movies_rdd),
            "loader": (lambda x: parsers_and_loaders.load_genres(x, sep=msep,\
                parser_function=parsers_and_loaders.parseIMDBKeywords,
                    prefix="imdb_keywords"))
        },
        {
            "name": "imdb_genres",
            "src_rdd": (lambda: movies_rdd),
            "loader": (lambda x: parsers_and_loaders.load_genres(x, sep=msep,\
                parser_function=parsers_and_loaders.parseIMDBGenres,
                    prefix="imdb_genres"))
        },
        {
            "name": "imdb_director",
            "src_rdd": (lambda: movies_rdd),
            "loader": (lambda x: parsers_and_loaders.load_genres(x, sep=msep,\
                parser_function=parsers_and_loaders.parseIMDBDirector,
                    prefix="imdb_director"))
        },
        {
            "name": "imdb_producer",
            "src_rdd": (lambda: movies_rdd),
            "loader": (lambda x: parsers_and_loaders.load_genres(x, sep=msep,\
                parser_function=parsers_and_loaders.parseIMDBProducer,
                    prefix="imdb_producer"))
        },
        {
            "name": "tvtropes",
            "src_rdd": (
                lambda: sc.parallelize(
                    parsers_and_loaders.loadCSV(
                        args.tvtropes_file,
                        remove_first_line=True
                    )
                )
            ),
            "loader": (lambda x: parsers_and_loaders.load_genres(x, sep=",",\
                parser_function=parsers_and_loaders.parseTropes,
                    prefix="tvtropes"))
        }
    ]



    training = ratings\
      .filter(lambda x: (x[1][1] in all_movies) and (True or x[0] < 2))\
      .values() \
      .repartition(args.num_partitions) \
      .cache()

    logger.debug("%d records in the training set", training.count())
    logger.debug("%d unique movies in the training set",
                 len(set(training.map(lambda x: x[1]).collect())))

    if args.regression_users:
        args.metadata_sources = ["users"]
        logger.debug("Loading users")
        start = time.time()
        users_rdd = sc.parallelize(
            parsers_and_loaders.loadCSV(
                join(args.data_path, "users" + extension),
                remove_first_line=remove_first_line
                )
            )
        users = parsers_and_loaders.load_users(users_rdd, sep)
        all_users = set(users[0].keys().collect())
        logger.debug("Done in %f seconds", time.time() - start)
        metadata_sources = [
            {
                "name": "users",
                "src_rdd": (lambda: users),
                "loader": (lambda x: x)
            }]
        results = internal_feature_predictor.internal_feature_predictor(sc,\
            training, args.rank,\
            args.num_iter, args.lmbda,\
            args, all_users, metadata_sources,\
            user_or_product_features="user", eval_regression=True,\
            compare_with_replaced_feature=True,\
            compare_with_randomized_feature=True, logger=logger,\
            train_ratio=args.cross_validation)

        internal_feature_predictor.display_internal_feature_predictor(\
            results, logger)

    elif args.predict_product_features:
        results = internal_feature_predictor.internal_feature_predictor(\
            sc, training, args.rank,\
            args.num_iter, args.lmbda,\
            args, all_movies, metadata_sources,\
            user_or_product_features="product", eval_regression=True,\
            compare_with_replaced_feature=True,\
            compare_with_randomized_feature=True, logger=logger,\
            train_ratio=args.cross_validation)

        internal_feature_predictor.display_internal_feature_predictor(\
           results, logger)

    sc.stop()

if __name__ == "__main__":
    main()

