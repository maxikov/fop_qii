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

# Global variables

# Arguments to the ALS training function
rank = 12
lmbda = 0.1
numIter = 20

# Number of partitions created 
numPartitions = 4


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
    """
    DEPRACATED (no longer used)
    """
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
    """
    DEPRACATED (no longer used)
    """
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
        model = ALS.train(training, rank, numIter, lmbda, seed=7)
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

def build_recommendations(sc, myRatings, model):
    """
    Create recommendations for movies not in the current ratings set
    """
    #myRatedMovieIds = set([x[1] for x in myRatings])
    myRatedMovieIds = set([x[1] for x in myRatings.toLocalIterator()])
    candidates = sc.parallelize([m for m in movies if m not in myRatedMovieIds]).cache()
    predictions = model.predictAll(candidates.map(lambda x: (0, x))).collect()
    recommendations = sorted(predictions, key = lambda x: x.product)
    return recommendations

def print_top_recommendations(recommendations, movies):
    """
    Print the top 50 movie recommendations
    """
    top_recommendations = sorted(recommendations, key=lambda x: x.rating,
            reverse=True)[:50]
    for i in xrange(len(top_recommendations)):
        print ("%2d: %s" % (i + 1, movies[top_recommendations[i][1]])).encode('ascii', 'ignore')

def recommendations_to_dd(recommendations):
    """
    Convert recommendations to dictionary
    """
    res = defaultdict(lambda: 0.0)
    for rec in recommendations:
        res[rec.product] = rec.rating
    return res

def set_ratings_in_dataset(sc, dataset, new_ratings):
    """
    DEPRACATED (no longer used)
    THIS METHOD DOES NOT WORK
    """
    new_dataset = [x for x in dataset.toLocalIterator()]
    new_ratings_dict = {(x[0], x[1]):x[2] for x in new_dataset}
    for i in xrange(len(new_dataset)):
        if (new_dataset[i][0], new_dataset[i][1]) in new_ratings_dict:
            new_dataset[i] = (new_dataset[i][0], new_dataset[i][1],
                    new_ratings_dict[(new_dataset[i][0],
                        new_dataset[i][1])])
    new_dataset = sc.parallelize(new_dataset).cache()
###########################################
    # Test randomly iterating through ALL movies
#    myMovies = get_users_movies(myRatings)
#    new_ratings = myRatings
#    for i in xrange(len(myRatings)):
#	    temp = list(new_ratings[i])
#	    temp[1] = random.randint(1,3076)
#           new_ratings[i] = tuple(temp)
#           new_rating = random.randint(1,5) #*4.0 + 1.0
#           new_ratings = set_users_rating(new_ratings, movies[new_ratings[i][1]], new_rating)
#############################################
    return new_dataset

def compute_local_influence(sc, user_id, original_recommendations,
        ratings, rank, lmbda, numIter, qii_iters = 5, mode="exhaustive"):
    """
    Compute the QII metrics for each rating given by a user
    """
    print "Computing QII for user: ", user_id
    new_dataset, user_ratings = extract_ratings_by_uid(ratings, user_id)

    res = defaultdict(lambda: 0.0)
    myMovies = get_users_movies(user_ratings)
    old_recs = recommendations_to_dd(original_recommendations)
    for movie in myMovies:
        for i in xrange(qii_iters):
            if mode == "exhaustive":
	        print "xrange is: "+ str(i + 1.0)
                new_rating = i + 1.0
                if new_rating > 5:
                    break
            elif mode == "random":
                new_rating = random.random()*4.0 + 1.0
#TODOOOOOO
            new_ratings = dict()
            new_ratings[movie] = new_rating
            new_dataset = set_user_ratings(sc, new_dataset, user_ratings, new_ratings)
            """
            new_ratings = set_users_rating(myRatings, movie, new_rating)
            print "New ratings:", new_ratings
            newRatingsRDD = sc.parallelize(new_ratings, 1)
            print "Building new data set"
            new_dataset = new_ratings.values() \
              .union(newRatingsRDD) \
              .repartition(numPartitions) \
              .cache()
            """
            print "Building model"
            new_model = ALS.train(new_dataset, rank, numIter, lmbda, seed=7)
            print "Built, predicting"
            new_recommendations = build_recommendations(sc, new_dataset,
                    new_model)
            new_recs = recommendations_to_dd(new_recommendations)
            for mid in set(old_recs.keys()).union(set(new_recs.keys())):
                res[movie] += abs(old_recs[mid] - new_recs[mid])
            print "Local influence:", res
    res_normed = {k: v/float(qii_iters) for k, v in res.items()}
    return res_normed

def get_users_movies(myRatings):
    """
    Get all movies rated by a given user
    """
    #return [x[1] for x in myRatings]
    return [x[1] for x in myRatings.toLocalIterator()]

def set_users_rating(myRatings, movie_id, new_rating):
    """
    Set a user rating for a movie in a current set
    """
    new_ratings = copy.deepcopy(myRatings)
    for i in xrange(len(new_ratings)):
        if new_ratings[i][1] == movie_id:
            new_ratings[i] = (new_ratings[i][0], movie_id, new_rating)
            break
    return new_ratings

def compute_recommendations_and_qii(sc, dataset, user_id):
    """
    Computes the recommendations and qii metrics for a given dataset and user
    specified by ID
    """
    # TODO avoid retraining?
    model = ALS.train(dataset, rank, numIter, lmbda)

    print "Computing recommendations/QII for user: ", user_id
    myRatings = get_ratings_from_uid(dataset, user_id)
    print "User ratings: ", [x for x in myRatings.toLocalIterator()]

    # make personalized recommendations
    recommendations = build_recommendations(sc, myRatings, model)
    print "Movies recommended for you:"
    print_top_recommendations(recommendations, movies)

    local_influence = compute_local_influence(sc, user_id, recommendations,
            dataset, rank, lmbda, numIter)

    print "Local influence:"
    for mid, minf in sorted(local_influence.items(), key = lambda x: -x[1]):
        print movies[mid], ":", minf

    return recommendations, local_influence

def get_uid_from_ratings(myRatings):
    return myRatings.toLocalIterator().next()[0]

def perturb_user_ratings(sc, dataset, user_id):
    """
    Takes a data set and perturbs the ratings for single user
    specified by ID, to random values
    """
    new_dataset, user_ratings = extract_ratings_by_uid(dataset, user_id)
    set_user_ratings(sc, new_dataset, user_ratings)


def set_user_ratings(sc, new_dataset, user_ratings, new_ratings = dict()):
    """
    Takes a data set (missing user data) and perturbs the ratings for single user
    specified by ID, by default to random values or to specified
    values if provided
    """
    user_movies = get_users_movies(user_ratings)
    new_ratings_list = []
    for movie in user_movies:
        if not len(new_ratings):
            new_rating = random.random()*4.0 + 1.0
        elif movie in new_ratings:
            new_rating = new_ratings[movie]
        else:
            new_rating = user_ratings.filter(lambda x: x[1] == movie).toLocalIterator().next()[2]
        new_ratings_list.append((get_uid_from_ratings(user_ratings), movie, new_rating))

    new_ratings_rdd = sc.parallelize(new_ratings_list).cache()
    combined_dataset = new_dataset \
      .union(new_ratings_rdd) \
      .repartition(numPartitions) \
      .cache()

    return combined_dataset

def get_ratings_from_uid(dataset, user_id):
    """
    Returns the set of ratings from a given user specified by ID
    """
    user_ratings = dataset.filter(lambda x: x[0] == user_id) \
      .repartition(numPartitions) \
      .cache()

    return user_ratings

def extract_ratings_by_uid(dataset, user_id):
    """
    Removes the ratings from a given user in the dataset and
    returns those ratings along with the modified dataset
    """
    new_dataset = dataset.filter(lambda x: x[0] != user_id) \
      .repartition(numPartitions) \
      .cache()

    user_ratings = dataset.filter(lambda x: x[0] == user_id) \
      .repartition(numPartitions) \
      .cache()

    return new_dataset, user_ratings


def calculate_l1_distance(dict1, dict2):
    """
    Calcuate the L1 distance between two dictionaries
    """
    keys = set(dict1.keys()).union(dict2.keys())
    res = 0.0
    for key in keys:
        if key in dict1:
            d1 = dict1[key]
        else:
            d1 = 0
        if key in dict2:
            d2 = dict2[key]
        else:
            d2 = 0
        res += abs(d1 + d2)
    return res

def get_user_list(dataset):
    """
    Extract the full list of users from the dataset
    """
    res = dataset\
            .map(lambda x: x[0])\
            .collect()
    return list(set(res))


def compute_user_local_sensitivity(sc, dataset, user_id, num_iters_ls):
    """
    Computes the local sensitivitiy for a given user over a
    specific dataset

    TODO
    """
    recommendation_ls = []
    qii_ls = []

    for x in xrange(num_iters_ls):
        # Get a random user that is not the current user
        # code here for that, use get_user_list
        perturbed_datset = perturb_user_ratings(sc, dataset, other_user_id)
        recommendations, local_influence = compute_recommendations_qii(sc, dataset, user_id)
        # more TODO calculate_l1_distance for both

    return recommendations_l1_dist, local_influence_l1_dist

if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print "Usage: /path/to/spark/bin/spark-submit --driver-memory 2g " + \
          "MovieLensALS.py movieLensDataDir"
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

    movieLensHomeDir = sys.argv[1]
    #myRatings = loadRatings(sys.argv[2])
    #myRatingsRDD = sc.parallelize(myRatings, 1)

    ratings = sc.textFile(join(movieLensHomeDir, "ratings.dat")).map(parseRating)
    movies = dict(sc.textFile(join(movieLensHomeDir, "movies.dat")).map(parseMovie).collect())

    # create the initial training dataset with default ratings
    training = ratings.filter(lambda x: x[0] < 6) \
      .values() \
      .repartition(numPartitions) \
      .cache()


    # TODO specify a user ID
    # TODO specify the number of iterations
    # TODO decide on what local sensitivity metrics to use/print,
    # average or maxiumum, etc.
    # TODO call this function on many user IDs, not just one
    #compute_user_local_sensitivity(sc, training, user_id, num_iters_ls)

    # JUST FOR TESTING
    list_of_users = get_user_list(training)
    recommendations, local_influence = compute_recommendations_and_qii(sc, training, list_of_users[0])

    print recommendations, local_influence

    # clean up
    sc.stop()
