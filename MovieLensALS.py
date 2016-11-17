#!/usr/bin/env python

import sys
import itertools
import copy
import random
from math import sqrt
from operator import add
from os.path import join, isfile, dirname
from collections import defaultdict

from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS

def parseRating(line):
    """
    Parses a rating record in MovieLens format userId::movieId::rating::timestamp .
    """
    fields = line.strip().split("::")
    return long(fields[3]) % 10, (int(fields[0]), int(fields[1]), float(fields[2]))

def parseMovie(line):
    """
    Parses a movie record in MovieLens format movieId::movieTitle .
    """
    fields = line.strip().split("::")
    return int(fields[0]), fields[1]

def loadRatings(ratingsFile):
    """
    Load ratings from file.
    """
    if not isfile(ratingsFile):
        print "File %s does not exist." % ratingsFile
        sys.exit(1)
    f = open(ratingsFile, 'r')
    ratings = filter(lambda r: r[2] > 0, [parseRating(line)[1] for line in f])
    f.close()
    if not ratings:
        print "No ratings provided."
        sys.exit(1)
    else:
        return ratings

def computeRmse(model, data, n):
    """
    Compute RMSE (Root Mean Squared Error).
    """
    predictions = model.predictAll(data.map(lambda x: (x[0], x[1])))
    predictionsAndRatings = predictions.map(lambda x: ((x[0], x[1]), x[2])) \
      .join(data.map(lambda x: ((x[0], x[1]), x[2]))) \
      .values()
    return sqrt(predictionsAndRatings.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(n))


def create_data_sets(sc, myRatingsRDD):
    # load ratings and movie titles

    movieLensHomeDir = sys.argv[1]

    # ratings is an RDD of (last digit of timestamp, (userId, movieId, rating))
    ratings = sc.textFile(join(movieLensHomeDir, "ratings.dat")).map(parseRating)

    # movies is an RDD of (movieId, movieTitle)
    movies = dict(sc.textFile(join(movieLensHomeDir, "movies.dat")).map(parseMovie).collect())

    numRatings = ratings.count()
    numUsers = ratings.values().map(lambda r: r[0]).distinct().count()
    numMovies = ratings.values().map(lambda r: r[1]).distinct().count()

    print "Got %d ratings from %d users on %d movies." % (numRatings, numUsers, numMovies)

    # split ratings into train (60%), validation (20%), and test (20%) based on the 
    # last digit of the timestamp, add myRatings to train, and cache them

    # training, validation, test are all RDDs of (userId, movieId, rating)

    numPartitions = 4
    training = ratings.filter(lambda x: x[0] < 6) \
      .values() \
      .union(myRatingsRDD) \
      .repartition(numPartitions) \
      .cache()

    validation = ratings.filter(lambda x: x[0] >= 6 and x[0] < 8) \
      .values() \
      .repartition(numPartitions) \
      .cache()

    test = ratings.filter(lambda x: x[0] >= 8).values().cache()



    numTraining = training.count()
    numValidation = validation.count()
    numTest = test.count()

    print "Training: %d, validation: %d, test: %d" % (numTraining, numValidation, numTest)

    return training, test, validation, movies


def create_best_model(sc, training, test, validation):

    numTraining = training.count()
    numValidation = validation.count()
    numTest = test.count()

    ranks = [8, 12]
    lambdas = [0.1, 10.0]
    numIters = [10, 20]
    bestModel = None
    bestValidationRmse = float("inf")
    bestRank = 0
    bestLambda = -1.0
    bestNumIter = -1

    for rank, lmbda, numIter in itertools.product(ranks, lambdas, numIters):
        model = ALS.train(training, rank, numIter, lmbda)
        validationRmse = computeRmse(model, validation, numValidation)
        print "RMSE (validation) = %f for the model trained with " % validationRmse + \
              "rank = %d, lambda = %.1f, and numIter = %d." % (rank, lmbda, numIter)
        if (validationRmse < bestValidationRmse):
            bestModel = model
            bestValidationRmse = validationRmse
            bestRank = rank
            bestLambda = lmbda
            bestNumIter = numIter

    testRmse = computeRmse(bestModel, test, numTest)

    # evaluate the best model on the test set
    print "The best model was trained with rank = %d and lambda = %.1f, " % (bestRank, bestLambda) \
      + "and numIter = %d, and its RMSE on the test set is %f." % (bestNumIter, testRmse)

    # compare the best model with a naive baseline that always returns the mean rating
    meanRating = training.union(validation).map(lambda x: x[2]).mean()
    baselineRmse = sqrt(test.map(lambda x: (meanRating - x[2]) ** 2).reduce(add) / numTest)
    improvement = (baselineRmse - testRmse) / baselineRmse * 100
    print "The best model improves the baseline by %.2f" % (improvement) + "%."

    return bestModel

def build_recommendations(sc, myRatings, bestModel):
    myRatedMovieIds = set([x[1] for x in myRatings])
    candidates = sc.parallelize([m for m in movies if m not in myRatedMovieIds])
    predictions = bestModel.predictAll(candidates.map(lambda x: (0, x))).collect()
#    recommendations = sorted(predictions, key=lambda x: x[2], reverse=True)[:50]
    recommendations = sorted(predictions, key = lambda x: x.product)
    return recommendations

def print_top_recommendations(recommendations, movies):
    top_recommendations = sorted(recommendations, key=lambda x: x.rating,
            reverse=True)[:50]
    for i in xrange(len(top_recommendations)):
        print ("%2d: %s" % (i + 1, movies[top_recommendations[i][1]])).encode('ascii', 'ignore')

def recommendations_to_dd(recommendations):
    res = defaultdict(lambda: 0.0)
    for rec in recommendations:
        res[rec.product] = rec.rating
    return res

def set_ratings_in_dataset(sc, dataset, new_ratings):
    new_dataset = [x for x in dataset.toLocalIterator()]
    new_ratings_dict = {(x[0], x[1]):x[2] for x in new_dataset}
    for i in xrange(len(new_dataset)):
        if (new_dataset[i][0], new_dataset[i][1]) in new_ratings_dict:
            new_dataset[i] = (new_dataset[i][0], new_dataset[i][1],
                    new_ratings_dict[(new_dataset[i][0],
                        new_dataset[i][1])])
    new_dataset = sc.parallelize(new_dataset).cache()
    return new_dataset


def compute_local_influence(sc, myRatings, original_recommendations,
        bestModel, training_set, rank, lmbda, numIter, qii_iters = 5):
    res = defaultdict(lambda: 0.0)
    myMovies = get_users_movies(myRatings)
    old_recs = recommendations_to_dd(original_recommendations)
    for movie in myMovies:
        for i in xrange(qii_iters):
            new_rating = random.random*4.0 + 1.0
            new_ratings = set_users_rating(myRatings, movie, new_rating)
            print "New ratings:", new_ratings
            print "Building new data set"
            new_dataset = set_ratings_in_dataset(sc, training, new_ratings)
            print "Building model"
            new_model = ALS.train(new_dataset, rank, numIter, lmbda)
            print "Built, predicting"
            new_recommendations = build_recommendations(sc, new_ratings,
                    new_model)
            new_recs = recommendations_to_dd(new_recommendations)
            #print "New recommendations:", new_recommendations
            for mid in set(old_recs.keys()).union(set(new_recs.keys())):
                res[movie] += abs(old_recs[mid] - new_recs[mid])
                #print old_recs[mid], new_recs[mid]
            print "Local influence:", res
    return res


def get_users_movies(myRatings):
    return [x[1] for x in myRatings]

def set_users_rating(myRatings, movie_id, new_rating):
    new_ratings = copy.deepcopy(myRatings)
    for i in xrange(len(new_ratings)):
        if new_ratings[i][1] == movie_id:
            new_ratings[i] = (new_ratings[i][0], movie_id, new_rating)
            break
    return new_ratings

if __name__ == "__main__":
    if (len(sys.argv) != 3):
        print "Usage: /path/to/spark/bin/spark-submit --driver-memory 2g " + \
          "MovieLensALS.py movieLensDataDir personalRatingsFile"
        sys.exit(1)

    # set up environment
    conf = SparkConf() \
      .setAppName("MovieLensALS") \
      .set("spark.executor.memory", "2g")
    sc = SparkContext(conf=conf)

####################################### Fixes Stack Overflow issue when training ALS
    sc.setCheckpointDir('checkpoint/')
    ALS.checkpointInterval = 2
#######################################

    myRatings = loadRatings(sys.argv[2])
    myRatingsRDD = sc.parallelize(myRatings, 1)

    training, test, validation, movies = create_data_sets(sc, myRatingsRDD)


    # load personal ratings

    print "My ratings:"
    for i in xrange(len(myRatings)):
        print movies[myRatings[i][1]], ":", myRatings[i][2]

    bestModel = create_best_model(sc, training, test, validation)

    rank = 12
    lmbda = 0.1
    numIter = 20
#    bestModel = model = ALS.train(training, rank, numIter, lmbda)
#    model = ALS.train(training, rank, numIter, lmbda)


    # make personalized recommendations
    recommendations = build_recommendations(sc, myRatings, bestModel)
    print "Movies recommended for you:"
    print_top_recommendations(recommendations, movies)


    local_influence = compute_local_influence(sc, myRatings, recommendations,
            bestModel, training, rank, lmbda, numIter, qii_iters = 5)

    print "Local influence:"
    for mid, minf in sorted(local_influence.items(), key = lambda x: -x[1]):
        print movies[mid], ":", minf


    # clean up
    sc.stop()
