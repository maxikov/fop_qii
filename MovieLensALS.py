#!/usr/bin/env python

import logging
import sys
import itertools
import copy
import random
from math import sqrt
from operator import add
from os.path import join, isfile, dirname
from collections import defaultdict
import time
import argparse
import math
from prettytable import PrettyTable
import numpy
import itertools
import StringIO
import csv
import traceback

from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS
import pyspark.mllib.recommendation
from pyspark.mllib.classification import LabeledPoint
import pyspark.mllib.regression
import pyspark.mllib.tree
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.evaluation import RegressionMetrics,\
        BinaryClassificationMetrics

from pyspark.mllib.linalg import Matrices
from pyspark.mllib.linalg.distributed import BlockMatrix
from pyspark.sql import SQLContext

# Global variables

# Arguments to the ALS training function
rank = 12
lmbda = 0.1
numIter = 20

# Number of partitions created

qii_iters = 5
num_iters_ls = 5

max_movies_per_user = 0 #0 = no limit

recommendations_to_print = 0 # 0 = don't print

print_movie_names = False

perturb_specific_user = None

def loadCSV(fname, remove_first_line = True):
    """
    Probably won't work with very large data sets.
    """
    with open(fname, "r") as f:
        res = f.readlines()
        if remove_first_line:
            res = res[1:]
        return res

def parseTag(line, sep="::"):
    """
    Parses a tag record in MovieLens format.
    """
    #Multi-character delimiters aren't supported,
    #but this data set doesn't have %s anywhere.
    #Dirty hack, need to fix later
    if sep == "::":
        line = line.replace(sep, "%")
        sep = "%"
    s = StringIO.StringIO(line)
    r = csv.reader(s, delimiter=sep, quotechar='"')
    x = r.next()
    return (int(x[0]), int(x[1]), x[2])

def parseRating(line, sep="::"):
    """
    Parses a rating record in MovieLens format userId::movieId::rating::timestamp .
    """
    #Multi-character delimiters aren't supported,
    #but this data set doesn't have %s anywhere.
    #Dirty hack, need to fix later
    if sep == "::":
        line = line.replace(sep, "%")
        sep = "%"
    s = StringIO.StringIO(line)
    r = csv.reader(s, delimiter=sep, quotechar='"')
    fields = r.next()
    return long(fields[3]) % 10, (int(fields[0]), int(fields[1]), float(fields[2]))

def parseIMDBKeywords(line, sep="::"):
    """
    Parses movie genres in format
    movieId,title,genres,imdbId,localImdbID,tmdbId,imdb_genres,imdb_keywords
    """
    #Multi-character delimiters aren't supported,
    #but this data set doesn't have %s anywhere.
    #Dirty hack, need to fix later
    if sep == "::":
        line = line.replace(sep, "%")
        sep = "%"
    s = StringIO.StringIO(line)
    r = csv.reader(s, delimiter=sep, quotechar='"')
    fields = r.next()
    mid = int(fields[0])
    keywords = fields[6]
    keywords = keywords.split("|")
    return mid, set(keywords)

def parseGenre(line, sep="::"):
    """
    Parses movie genres in MovieLens format
    movieId::movieTitle::movieGenre1[|movieGenre2...]
    """
    #Multi-character delimiters aren't supported,
    #but this data set doesn't have %s anywhere.
    #Dirty hack, need to fix later
    if sep == "::":
        line = line.replace(sep, "%")
        sep = "%"
    s = StringIO.StringIO(line)
    r = csv.reader(s, delimiter=sep, quotechar='"')
    fields = r.next()
    mid = int(fields[0])
    genres = fields[2]
    genres = genres.split("|")
    return mid, set(genres)

def parseYear(line, sep="::"):
    """
    Parses movie years in MovieLens format
    movieId::movieTitle (movieYear)::movieGenre1[|movieGenre2...]
    """
    #Multi-character delimiters aren't supported,
    #but this data set doesn't have %s anywhere.
    #Dirty hack, need to fix later
    if sep == "::":
        line = line.replace(sep, "%")
        sep = "%"
    s = StringIO.StringIO(line)
    r = csv.reader(s, delimiter=sep, quotechar='"')
    fields = r.next()
    mid = int(fields[0])
    mtitle = fields[1]
    try:
        year = mtitle.split("(")[-1]
        year = year.split(")")[0]
        year = int(year)
        return mid, year
    except Exception as e: #Dirty hack, but this isn't even supposed to happen!
        print e
        traceback.print_exc()
        print "mid:", mid
        print "Setting the year to 2000, and continuing"
        return mid, 2000


def parseMovie(line, sep="::"):
    """
    Parses a movie record in MovieLens format movieId::movieTitle .
    """
    #Multi-character delimiters aren't supported,
    #but this data set doesn't have %s anywhere.
    #Dirty hack, need to fix later
    if sep == "::":
        line = line.replace(sep, "%")
        sep = "%"
    s = StringIO.StringIO(line)
    r = csv.reader(s, delimiter=sep, quotechar='"')
    fields = r.next()
    return int(fields[0]), fields[1]

def parseUser(line, sep="::"):
    """
    Parses user metadata from the 1m format:
    UserID::Gender::Age::Occupation::Zip-code

    All demographic information is provided voluntarily by the users and is
    not checked for accuracy.  Only users who have provided some demographic
    information are included in this data set.

    - Gender is denoted by a "M" for male and "F" for female
    - Age is chosen from the following ranges:

        *  1:  "Under 18"
        * 18:  "18-24"
        * 25:  "25-34"
        * 35:  "35-44"
        * 45:  "45-49"
        * 50:  "50-55"
        * 56:  "56+"

    - Occupation is chosen from the following choices:

        *  0:  "other" or not specified
        *  1:  "academic/educator"
        *  2:  "artist"
        *  3:  "clerical/admin"
        *  4:  "college/grad student"
        *  5:  "customer service"
        *  6:  "doctor/health care"
        *  7:  "executive/managerial"
        *  8:  "farmer"
        *  9:  "homemaker"
        * 10:  "K-12 student"
        * 11:  "lawyer"
        * 12:  "programmer"
        * 13:  "retired"
        * 14:  "sales/marketing"
        * 15:  "scientist"
        * 16:  "self-employed"
        * 17:  "technician/engineer"
        * 18:  "tradesman/craftsman"
        * 19:  "unemployed"
        * 20:  "writer"
    """
    #Multi-character delimiters aren't supported,
    #but this data set doesn't have %s anywhere.
    #Dirty hack, need to fix later
    if sep == "::":
        line = line.replace(sep, "%")
        sep = "%"
    s = StringIO.StringIO(line)
    r = csv.reader(s, delimiter=sep, quotechar='"')
    fields = r.next()
    fields[1] = 0 if fields[1] == "M" else 1
    fields[4] = fields[4].split("-")[0]
    fields = map(int, fields)
    return tuple(fields)


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




def build_recommendations(sc, myRatings, model):
    """
    Create recommendations for movies not in the current ratings set
    """
    #myRatedMovieIds = set([x[1] for x in myRatings])
    uid = get_uid_from_ratings(myRatings)
    #print "uid:", uid
    myRatedMovieIds = set([x[1] for x in myRatings.collect()])
    #print "myRatedMovieIds:", myRatedMovieIds
    candidates = sc.parallelize([m for m in movies if m not in myRatedMovieIds]).cache()
    #print candidates
    predictions = model.predictAll(candidates.map(lambda x: (uid, x))).collect()
    #print predictions
    recommendations = sorted(predictions, key = lambda x: x.product)
    return recommendations

def print_top_recommendations(recommendations, n=10, all_ratings=False,
        movie_qiis=None, movie_qiis_to_display=0):
    """
    Print the top n movie recommendations
    """
    top_recommendations = sorted(recommendations, key=lambda x: x.rating,
            reverse=True)
    if all_ratings:
        title = ["Rank", "Movie", "Rating"]
    elif movie_qiis is None:
        title = ["Rank", "Movie", "Estimated rating"]
        top_recommendations = top_recommendations[:n]
    else:
        title = ["Rank", "Movie", "Estimated rating"]
        for i in xrange(movie_qiis_to_display):
            title.append("#{} influence".format(i+1))
            title.append("#{} QII".format(i+1))
            top_recommendations = top_recommendations[:n]
    table = PrettyTable(title)
    for i in xrange(len(top_recommendations)):
        row = [
            i+1,
            movies[top_recommendations[i].product] if print_movie_names\
                    else top_recommendations[i].product,
            top_recommendations[i].rating,
        ]
        if movie_qiis is not None:
            qiis = movie_qiis[top_recommendations[i].product].items()
            qiis.sort(key=lambda x: -x[1])
            for j in xrange(movie_qiis_to_display):
                row.append(movies[qiis[j][0]])
                row.append(qiis[j][1])
        table.add_row(row)


    print table

def recommendations_to_dd(recommendations):
    """
    Convert recommendations to dictionary
    """
    res = defaultdict(lambda: 0.0)
    for rec in recommendations:
        res[rec.product] = rec.rating
    return res


def compute_local_influence(sc, user_id, original_recommendations,
        ratings, rank, lmbda, numIter, qii_iters, mode="exhaustive",
        per_movie=False):
    """
    Compute the QII metrics for each rating given by a user
    """
    print "Computing QII for user: ", user_id
    orig_dataset, user_ratings = extract_ratings_by_uid(ratings, user_id)

    new_dataset = None
    old_dataset = None
    if per_movie:
        res = defaultdict(lambda: defaultdict(lambda: 0.0))
    else:
        res = defaultdict(lambda: 0.0)
    myMovies = get_users_movies(user_ratings)
    old_recs = recommendations_to_dd(original_recommendations)
    for miter, movie in enumerate(myMovies):
        for i in xrange(qii_iters):
            if mode == "exhaustive":
                new_rating = i + 1.0
                if new_rating > 5:
                    break
            elif mode == "random":
                new_rating = random.random()*4.0 + 1.0
            print "Perturbing movie", movie, "(", miter + 1, "out of",\
                len(myMovies), ")"
            print "Perturbed rating:", new_rating
            new_ratings = dict()
            new_ratings[movie] = new_rating
            print "New ratings:", new_ratings
            new_dataset = set_user_ratings(sc, orig_dataset, user_ratings, new_ratings)
            print "Building model"
            new_model = ALS.train(new_dataset, rank, numIter, lmbda, seed=7)
            print "Built, predicting"
            new_recommendations = build_recommendations(sc, user_ratings,  #final chg
                    new_model)
            if recommendations_to_print > 0:
                print "New recommendations:"
                print_top_recommendations(new_recommendations,
                    recommendations_to_print)
            new_recs = recommendations_to_dd(new_recommendations)
            for mid in set(old_recs.keys()).union(set(new_recs.keys())):
                if per_movie:
                    res[mid][movie] += abs(old_recs[mid] - new_recs[mid])
                else:
                    res[movie] += abs(old_recs[mid] - new_recs[mid])
        if not per_movie:
            print "Local influence:", res[movie]
    if not per_movie:
        res_normed = {k: v/float(qii_iters) for k, v in res.items()}
        print "Final local influence:", res_normed
    else:
        res_normed = {k: {k1: v1/float(qii_iters) for k1, v1 in v.items()} for
            k, v in res.items()}
    return res_normed

def get_users_movies(myRatings):
    """
    Get all movies rated by a given user
    """
    #return [x[1] for x in myRatings]
    return list(myRatings.map(lambda x: x[1]).collect())

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

def compute_recommendations_and_qii(sc, dataset, user_id,
        dont_compute_qii=False, per_movie=False):
    """
    Computes the recommendations and qii metrics for a given dataset and user
    specified by ID
    """
    # TODO avoid retraining?
    print "Training the model, rank:", rank, "numIter:", numIter,\
            "lmbda:", lmbda
    start_recommend_time = time.time()
    model = ALS.train(dataset, rank, numIter, lmbda)

    print "Computing recommendations/QII for user: ", user_id
    myRatings = get_ratings_from_uid(dataset, user_id)
    #print "User ratings: ", list(myRatings.collect())

    # make personalized recommendations
    recommendations = build_recommendations(sc, myRatings, model)
    end_recommend_time = time.time()
    rec_time = end_recommend_time - start_recommend_time
    print "Time it took to create recommendations:", rec_time
    if dont_compute_qii:
        return recommendations, model
    if not per_movie and recommendations_to_print > 0:
        print "Movies recommended for you:"
        print_top_recommendations(recommendations, recommendations_to_print)

    local_influence = compute_local_influence(sc, user_id, recommendations,
            dataset, rank, lmbda, numIter, qii_iters, per_movie=per_movie)

    if not per_movie:
        print "Local influence:"
        t = PrettyTable(["Movie ID", "Local Influence"])
        for mid, minf in sorted(local_influence.items(), key = lambda x: -x[1]):
            if print_movie_names:
                t.add_row([movies[mid], minf])
            else:
                t.add_row([mid, minf])
        print t

    return recommendations, local_influence

def get_uid_from_ratings(myRatings):
    return list(myRatings.take(1))[0][0]

def perturb_user_ratings(sc, dataset, user_id):
    """
    Takes a data set and perturbs the ratings for single user
    specified by ID, to random values
    """
    new_dataset, user_ratings = extract_ratings_by_uid(dataset, user_id)
    combined_dataset = set_user_ratings(sc, new_dataset, user_ratings)
    return combined_dataset

def set_user_ratings(sc, new_dataset, user_ratings, new_ratings = None):
    """
    Takes a data set (missing user data) and perturbs the ratings for single user
    specified by ID, by default to random values or to specified
    values if provided
    """
    user_movies = get_users_movies(user_ratings)
    new_ratings_list = []
    for movie in user_movies:
        if not new_ratings:
            new_rating = random.random()*4.0 + 1.0
        elif movie in new_ratings:
            new_rating = new_ratings[movie]
        else:
            new_rating = user_ratings.filter(lambda x: x[1] ==
                movie).first()[2]
        new_ratings_list.append((get_uid_from_ratings(user_ratings), movie, new_rating))
    print "** New Ratings List: ", new_ratings_list

    new_ratings_rdd = sc.parallelize(new_ratings_list, 1).cache()  #chg
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

# debug
    print "Count of user ratings: ", user_ratings.count()

    return new_dataset, user_ratings


def calculate_l1_distance(dict1, dict2):
    """
    Calcuate the L1 distance between two dictionaries
    """
    res = 0.0
    for key in dict1.keys():
        d1 = dict1[key]
        d2 = dict2[key]
    res += abs(d1-d2)
    return res

def l1_norm(vec):
    """
    Calculate L1 norm of a dictionary
    """
    res = sum(abs(float(x)) for x in vec.values())
    return res

def get_user_list(dataset):
    """
    Extract the full list of users from the dataset
    """
    res = dataset\
            .map(lambda x: x[0])\
            .collect()
    return list(set(res))

def convert_recs_to_dict(rating_list):
    recdict = defaultdict( list )
    for _,k,v in rating_list:
        recdict[k] = v
    return recdict



def compute_user_local_sensitivity(sc, dataset, user_id, num_iters_ls):
    """
    Computes the local sensitivitiy for a given user over a
    specific dataset
    """

    res = defaultdict(lambda: 0.0)

    original_recs, original_qii = compute_recommendations_and_qii(sc, dataset,
            user_id)
    original_recs = recommendations_to_dd(original_recs)

    res["recommendee_user_id"] = user_id
    res["recommendee_recs_l1_norm"] = l1_norm(original_recs)
    res["recommendee_qii_l1_norm"] = l1_norm(original_qii)
    res["recommendee_recs_l0_norm"] = len(original_recs)
    res["recommendee_qii_l0_norm"] = len(original_qii)
    res["perturbations"] = []

    all_users = get_user_list(dataset)
    for x in xrange(num_iters_ls):
        if perturb_specific_user:
            other_user_id = perturb_specific_user
        else:
            other_user_id = random.choice(list(set(all_users) - {user_id}))
        print "Perturbing user", other_user_id, "(", x+1, "out of",\
            num_iters_ls, ")"
        perturbed_dataset = perturb_user_ratings(sc, dataset, other_user_id)
        start = time.time()
        recs, qii = compute_recommendations_and_qii(sc, perturbed_dataset, user_id)
        stop = time.time()
        recs = recommendations_to_dd(recs)
        rec_ls = calculate_l1_distance(original_recs, recs)
        qii_ls = calculate_l1_distance(original_qii, qii)

        report = {}
        report["perturbed_user_id"] = other_user_id
        report["perturbed_recs_l1_norm"] = l1_norm(recs)
        report["perturbed_qii_l1_norm"] = l1_norm(qii)
        report["perturbed_recs_l0_norm"] = len(recs)
        report["perturbed_qii_l0_norm"] = len(qii)
        report["recs_ls"] = rec_ls
        report["qii_ls"] = qii_ls
        report["recs_ls_norm"] = rec_ls/float((len(recs)*4))
        report["qii_ls_norm"] = qii_ls/float((len(qii)*4))
        print "Local sensitivity of recs: ", rec_ls/float((len(recs)*4))
        print "Local sensitivity of QII: ", qii_ls/float((len(qii)*4))
        report["computation_time"] = stop - start


        res["perturbations"].append(report)

    for per in res["perturbations"]:
        res["avg_recs_ls"] += float(per["recs_ls"])/len(res["perturbations"])
        res["max_recs_ls"] = max(res["max_recs_ls"], per["recs_ls"])
        res["avg_recs_ls_norm"] +=\
            float(per["recs_ls_norm"])/len(res["perturbations"])
        res["max_recs_ls_norm"] = max(res["max_recs_ls_norm"],
                per["recs_ls_norm"])
        res["avg_qii_ls"] += float(per["qii_ls"])/len(res["perturbations"])
        res["max_qii_ls"] = max(res["max_qii_ls"], per["qii_ls"])
        res["avg_qii_ls_norm"] +=\
            float(per["qii_ls_norm"])/len(res["perturbations"])
        res["max_qii_ls_norm"] = max(res["max_recs_qii_norm"],
                per["qii_ls_norm"])
    return dict(res)

def compute_multiuser_local_sensitivity(sc, dataset, num_iters_ls,
        num_users_ls):
    """
    Computes local sensitivity for a number of randomly chosen users.
    """
    res = []
    users_already_processed = set()
    all_users = list(get_user_list(dataset))
    for x in xrange(num_users_ls):
        while True:
            cur_user = random.choice(all_users)
            print "Trying user", cur_user
            if cur_user in users_already_processed:
                print "Oops, we've already processed this one"
                continue
            if max_movies_per_user == 0:
                break
            print "Looking at their ratings"
            u_ratings = get_ratings_from_uid(dataset, cur_user)
            u_ratings_list = u_ratings.collect()
            l = len(u_ratings_list)
            if l > max_movies_per_user:
                print "This user has too many movies: ",\
                        l, ">", max_movies_per_user
                users_already_processed.add(cur_user)
                continue
            else:
                print "This user with", l, "movies " +\
                        "rated is fine!"
                break
        print "Probing user", cur_user
        report = compute_user_local_sensitivity(sc, dataset, cur_user,
                num_iters_ls)
        users_already_processed.add(cur_user)
        res.append(report)
    return res

def users_with_most_ratings(training,listlength):
    userlist = sorted(training.countByKey().items(), key = lambda x: x[1], reverse=True)
    return userlist[0:listlength]

def correlate_genres(sc, genres, movies, ratings, rank, numIter, lmbda,
        invert_labels=False, no_threshold=False, classifier_model="logistic"):
        print "Bulding per-genre movie lists"
        start = time.time()
        gdict = dict(genres.collect())
        mrdd = sc.parallelize(movies.keys())
        all_genres = sorted(list(genres.map(lambda (_, x): x).fold(set(), lambda x, y:
            set(x).union(set(y)))))
        movies_by_genre = {}
        for cur_genre in all_genres:
            print "Processing {}".format(cur_genre)
            cur_movies = mrdd.map(lambda x: (x, 1 ^ invert_labels
                if cur_genre in gdict[x] else 0 ^ invert_labels))
            movies_by_genre[cur_genre] = dict(cur_movies.collect())
        print "Done in {} seconds".format(time.time() - start)

        print "Training model"
        start = time.time()
        model = ALS.train(ratings, rank, numIter, lmbda)
        print "Done in {} seconds".format(time.time() - start)
        features = dict(model.productFeatures().collect())

        print "Building a family of regresions"
        reg_models = {}
        start = time.time()
        avgbetter = 0.0
        for cur_genre, cur_movies in movies_by_genre.items():
            print "Processing {}".format(cur_genre)
            lr_data = [LabeledPoint(lbl, features[mid])
                    for (mid, lbl) in cur_movies.items()
                    if mid in features]
            lr_data = sc.parallelize(lr_data)
            n_pos = lr_data.filter(lambda x: x.label == 1).count()
            prate = float(n_pos)/float(lr_data.count())
            print "Percent of positives: {:3.1f}%".\
                    format(100*prate)
            if classifier_model == "logistic":
                lr_model = pyspark.\
                        mllib.\
                        classification.\
                        LogisticRegressionWithLBFGS.\
                        train(lr_data)
            elif classifier_model == "decision_tree":
                lr_model = pyspark.\
                        mllib.\
                        tree.\
                        DecisionTree.\
                        trainClassifier(lr_data,2,{})
            labels = lr_data.map(lambda x: x.label)
            if no_threshold:
                lr_model.clearThreshold()
                scores = lr_model.predict(lr_data.map(lambda x:
                    x.features))
                predobs = scores.zip(labels).map(
                    lambda(a, b): (float(a), float(b)))
                metrics = BinaryClassificationMetrics(predobs)
                auroc = metrics.areaUnderROC
                aupr = metrics.areaUnderPR
                better = (1.0 - prate)/(1.0 - aupr)
                reg_models[cur_genre] = {"auroc": auroc,
                            "auprc": aupr, "prate": prate, "model": lr_model, "better":
                            better}
                avgbetter += better
                print "Area under ROC: {:1.3f}, area under precision-recall curve: {:1.3f} ".\
                            format(auroc, aupr) +\
                            "(AuPRc for a random classifier: {:1.3f}, {:1.3f} times better)\n".\
                           format(prate, better)
            else:
                predictions = lr_model.predict(lr_data.map(lambda x:
                    x.features))
                predobs = predictions.zip(labels).map(
                    lambda(a, b): (float(a), float(b)))
                corrects = predobs.filter(
                        lambda (x, y): (x == y))
                fp_count = predobs.filter(lambda (x, y):
                        (x == 1) and (y == 0)).count()
                tp_count = corrects.filter(lambda (x, y): (x==1)).count()
                tn_count = corrects.filter(lambda (x, y): (x==0)).count()
                p_count = labels.filter(lambda x: (x==1)).count()
                n_count = labels.filter(lambda x: (x==0)).count()
                total_count = predobs.count()
                acc = float(tp_count + tn_count)/total_count
                print "Accuracy (tp+tn)/(p+n): {:3.1f}%".format(100*acc)
                if p_count > 0:
                    recall = float(tp_count)/p_count
                    print "Recall (sensitivity, tp rate, tp/p): {:3.1f}%".\
                        format(100*recall)
                else:
                    recall = 0
                    print "No positives in the data set, setting recall"+\
                            " (sensitivity, tp rate, tp/p) to 0"
                if n_count > 0:
                    specificity = float(tn_count)/n_count
                    print "Specificity (tn rate, tn/n): {:3.1f}%".\
                            format(100*specificity)
                else:
                    specificity = 0
                    print "No negatives in the data set, setting specificity"+\
                            " (tn rate, tn/n) to 0"
                if tp_count+fp_count > 0:
                    precision = float(tp_count)/(tp_count+fp_count)
                    print "Precision (positive predictive value, tp/(tp+fp)):" +\
                        " {:3.1f}%".format(100*precision)
                else:
                    precision = 0
                    print "No positives predicted by the classifier"+\
                            " (tp+fp <= 0), setting precision ("+\
                            "positive predictive value, tp/(tp+fp)) to 0"
                if tn_count+fp_count > 0:
                    fpr = float(fp_count)/(tn_count + fp_count)
                    print "False positive rate (fp/(tn+fp)): {:3.1f}%".\
                            format(100*fpr)
                else:
                    fpr = 0
                    print "No true negatives of false positives, setting"+\
                            "false positive rate (fp/(tn+fp)) to 0"
                print ""
                avgbetter += recall
                reg_models[cur_genre] = {"total_count": total_count,
                        "tp_count": tp_count, "fp_count": fp_count,
                        "p_count": p_count, "n_count": n_count,
                        "accuracy": acc, "recall": recall,
                        "specificity": specificity, "precision": precision,
                        "fpr": fpr, "model": lr_model}
        avgbetter = avgbetter/float(len(movies_by_genre))
        if no_threshold:
            print avgbetter, "times better than random on average"
        else:
            print "Average recall: {:3.1f}%".format(100*avgbetter)
        print "Done in {} seconds".format(time.time() - start)

        print "{} genres".format(len(reg_models))

        #Trying to bring it closer to diagonal
        reg_models_src = reg_models.items()
        if classifier_model == "logistic":
            reg_models_res = []
            for i in xrange(len(reg_models_src)):
                ind = min(
                        enumerate(
                                abs(i - (min if invert_labels else max)(
                                            enumerate(x["model"].weights),
                                            key = lambda y: y[1]
                                           )[0]
                                   ) for (gnr, x) in reg_models_src
                                 ), key = lambda y: y[1]
                         )[0]
                reg_models_res.append(reg_models_src[ind])
                del reg_models_src[ind]
        else:
            reg_models_res = reg_models_src
        return reg_models_res, avgbetter

def recommender_mean_error(model, data, power=1.0):
    predictions = model.predictAll(data.map(lambda x: (x[0], x[1])))
    predictionsAndRatings = predictions.map(lambda x: ((x[0], x[1]), x[2])) \
      .join(data.map(lambda x: ((x[0], x[1]), x[2]))) \
      .values()
    return mean_error(predictionsAndRatings, power)

def manual_predict_all(data, user_features, product_features):
    user_products = data.map(lambda x: (x[0], x[1]))
    user_features = dict(user_features.collect())
    product_features = dict(product_features.collect())
    res = user_products.map(lambda (user, product):
            (user, product, sum(map(lambda (x, y): x*y,
                zip(user_features[user],
                    product_features[product])))))
    return res

def manual_recommender_mean_error(user_features, product_features, data, power=1.0):
    predictions = manual_predict_all(data, user_features, product_features)
    predictionsAndRatings = predictions.map(lambda x: ((x[0], x[1]), x[2])) \
      .join(data.map(lambda x: ((x[0], x[1]), x[2]))) \
      .values()
    return mean_error(predictionsAndRatings, power)

def mean_error(predictionsAndRatings, power):
    return (predictionsAndRatings.map(lambda x: abs(x[0] - x[1]) **\
        power).reduce(add) / float(predictionsAndRatings.count())) ** (1.0/power)

def manual_diff_models(model_1, model_2, user_product_pairs, power=1.0):
    predictions_1 = manual_predict_all(user_product_pairs, *model_1)
    predictions_2 = manual_predict_all(user_product_pairs, *model_2)
    predictionsAndRatings = predictions_1.map(lambda x: ((x[0], x[1]), x[2])) \
      .join(predictions_2.map(lambda x: ((x[0], x[1]), x[2]))) \
      .values()
    return mean_error(predictionsAndRatings, power)

def set_list_value(lst, ind, val):
    new_lst = list(lst)
    new_lst[ind] = val
    return new_lst

def set_list_values(*args):
    return set_list_value(*args) #OOPS

def get_feature_distribution(features, f):
    res = features.map(lambda (_, arr): arr[f]).collect()
    return res

def perturb_feature(features, f):
    dist = get_feature_distribution(features, f)
    res = features.map(lambda (x, arr):
            (x, set_list_value(arr, f, random.choice(dist))))
    return res

def fast_feature_global_influence(user_features, product_features, f):
    #FIXME
    uarr = user_features.map(lambda (_, arr): arr[f])
    parr = product_features.map(lambda (_, arr): arr[f])
    ucount = user_features.count()
    pcount = product_features.count()
    um = Matrices.dense(ucount, 1, uarr.collect())
    um = BlockMatrix(sc.parallelize([((0,0),um)]), ucount, 1)
    pm = Matrices.dense(1, pcount, parr.collect())
    pm = BlockMatrix(sc.parallelize([((0,0), pm)]), 1, pcount)
    product = um.multiply(pm)
    res = product.rows.map(lambda x: sum(x)).sum()
    return res

def feature_global_influence(model, rank, user_product_pairs, power=1.0,
        compute_rmse = False, data = None, compute_fast_inf = False):
    original_user_features = model.userFeatures()
    original_product_features = model.productFeatures()
    original_model = (original_user_features, original_product_features)
    print "Computing the predictions of the original model"
    start = time.time()
    original_predictions = manual_predict_all(user_product_pairs,
            *original_model)
    print "Done in", time.time() - start, "seconds"
    res = {"feature_data": {}}
    if compute_rmse:
        print "Computing the mean error of the original model"
        start = time.time()
        error = recommender_mean_error(model, data, power)
        print "Done in", time.time() - start, "seconds"
        print "Mean error:", error
        res["original_rmse"] = error
    for f in xrange(rank):
        res["feature_data"][f] = {}
        if compute_fast_inf:
            print "Computing fast influence of feature", f, "out of", rank
            start = time.time()
            inf = fast_feature_global_influence(*original_model, f=f)
            print "\tDone in", time.time() - start, "seconds"
            print "\tInfluence:", inf
            res["feature_data"][f]["fast_influence"] = inf
        print "Perturbing feature", f, "out of", rank
        start = time.time()
        print "\tPerturbing user feature"
        perturbed_user_features = perturb_feature(original_user_features, f)
        print "\tDone in", time.time() - start, "seconds"
        print "\tPerturbing product feature"
        start = time.time()
        perturbed_product_features = perturb_feature(original_product_features,
                f)
        print "\tDone in", time.time() - start, "seconds"
        perturbed_model = (perturbed_user_features, perturbed_product_features)
        print "\tComputing the predictions of the perturbed model"
        start = time.time()
        perturbed_predictions = manual_predict_all(user_product_pairs,
            *perturbed_model)
        print "\tDone in", time.time() - start, "seconds"
        print "\tComparing models"
        start = time.time()
        predictionsAndRatings = original_predictions.map(lambda x: ((x[0], x[1]), x[2])) \
            .join(perturbed_predictions.map(lambda x: ((x[0], x[1]), x[2]))) \
            .values()
        diff = mean_error(predictionsAndRatings, power)
        print "\tDone in", time.time() - start, "seconds"
        print "\tAverage difference:", diff
        res["feature_data"][f]["model_diff"] = diff
        if compute_rmse:
            print "\tComputing the mean error of the original model"
            start = time.time()
            error = manual_recommender_mean_error(perturbed_user_features,
                    perturbed_product_features, data)
            print "\tDone in", time.time() - start, "seconds"
            print "\tMean error:", error
            res["feature_data"][f]["rmse"] = error
    return res


class AverageRatingRecommender:
    def __init__(self, logger=None):
        self.logger = logger

    def train(self, training):
        if self.logger is None:
            print "Training the average rating model"
        else:
            self.logger.debug("Training the average rating model")
        start = time.time()
        self.ratings = training\
                .groupBy(lambda x: x[1])\
                .map(lambda (mid, data):
                        (mid,
                        sum(x[2] for x in data)/float(len(data))
                        ))
        self.ratings = dict(self.ratings.collect())
        self.ratings = defaultdict(lambda: 0.0, self.ratings)
        if self.logger is None:
            print "Done in", time.time() - start, "seconds"
        else:
            self.logger.debug("Done in {} seconds".format(time.time() - start))

    def predict(self, user_movies):
        if self.logger is None:
            print "Making average rating recommendations"
        else:
            self.logger.debug("Making average rating predictions")
        start = time.time()
        res = user_movies\
                .map(lambda x: (x[0], x[1]))\
                .map(lambda (user, product):
                        (user,
                        product,
                        self.ratings[product]))
        if self.logger is None:
            print "Done in", time.time() - start, "seconds"
        else:
            self.logger.debug("Done in {} seconds".format(time.time() - start))
        return res

class RandomizedRecommender:
    def __init__(self, baseline_model, rank, logger=None):
        self.logger = logger
        self.baseline_model = baseline_model
        self.rank = rank

    def randomize(self):
        if self.logger is None:
            print "Creating randomized model"
        else:
            self.logger.debug("Creating randomized model")
        start = time.time()
        self.perturbed_features = self.baseline_model.productFeatures()
        for f in xrange(self.rank):
            if self.logger is None:
                print "Perturbing feature", f
            else:
                self.logger.debug("Perturbing feature {}".format(f))
            self.perturbed_features = perturb_feature(self.perturbed_features, f)
        if self.logger is None:
            print "Done in", time.time() - start, "seconds"
        else:
            self.logger.debug("Done in {} seconds".format(time.time() - start))

    def predict(self, user_movies):
        if self.logger is None:
            print "Making predictions of the randomized model"
        else:
            self.logger.debug("Making predictions of the randomized model")
        start = time.time()
        perturbed_predictions = manual_predict_all(user_movies,
                self.baseline_model.userFeatures(), self.perturbed_features)
        if self.logger is None:
            print "Done in", time.time() - start, "seconds"
        else:
            self.logger.debug("Done in {} seconds".format(time.time() - start))
        return perturbed_predictions

def train_regression_model(data, regression_model = "regression_tree",
    categorical_features = {}, max_bins = 32, max_depth = None, num_trees = 128,
    logger = None):
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
                    impurity = "variance",
                    maxDepth = max_depth,
                    maxBins = max_bins)

    elif regression_model == "random_forest":
        lr_model = pyspark.\
                mllib.\
                tree.\
                RandomForest.\
                trainRegressor(
                    data,
                    categoricalFeaturesInfo=categorical_features,
                    numTrees = num_trees,
                    maxDepth = max_depth,
                    maxBins = max_bins)
    elif regression_model == "linear":
        lr_model = LinearRegressionWithSGD.train(data)

    if logger is None:
        print "Done in {} seconds".format(time.time() - start)
    else:
        logger.debug("Done in {} seconds".format(time.time() - start))

    return lr_model


def build_meta_data_set(sc, sources, all_ids = None, logger = None):
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

def mean_relative_absolute_error(predobs):
    """
    Compute mean relative absolute error of regression.

    Parameters:
        predobs (mandatory) - RDD of (prediction, observation) pairs.

    Returns:
        MRAE (float)
    """
    res = predobs.\
        map(lambda (pred, obs):
            abs(pred-obs)/float(abs(obs if obs != 0 else 1))).\
                sum()/predobs.count()
    return res

def predict_internal_feature(features, indicators, f, regression_model,
        categorical_features, max_bins, logger = None):
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

    #logger.debug("Features for {} products loaded"\
            #        .format(features.count()))
    #logger.debug("Indicator vectors for {} products loaded"\
            #        .format(indicators.count()))

    start = time.time()

    joined = features.join(indicators)
    #logger.debug("{} items in 'joined'"\
            #        .format(joined.count()))
    data = joined.map(
            lambda (_id, (ftrs, inds)):
                LabeledPoint(ftrs[f], inds))
    #logger.debug("{} items in 'data'"\
            #        .format(data.count()))
    ids = joined.map(
            lambda (_id, _): _id)

    if logger is None:
        print "Done in {} seconds".format(time.time() - start)
    else:
        logger.debug("Done in {} seconds".format(time.time() - start))

    lr_model = train_regression_model(data,
        regression_model = regression_model,
        categorical_features = categorical_features,
        max_bins = nbins,
        logger = logger)

    observations = ids.zip(data.map(lambda x: float(x.label)))

    #logger.debug("{} items in 'observations'"\
            #        .format(observations.count()))

    predictions = ids.zip(
            lr_model\
            .predict(
                    data.map(lambda x: x.features)
                    )\
            .map(lambda x: float(x))
            )
    #logger.debug("{} items in 'predictions'"\
            #        .format(predictions.count()))

    return (lr_model, observations, predictions)


def evaluate_regression(predictions, observations, logger = None):
    if logger is None:
        print "Evaluating the model"
    else:
        logger.debug("Evaluating the model")
    start = time.time()
    predobs = predictions\
            .join(observations)\
            .values()
    metrics = RegressionMetrics(predobs)
    mrae = mean_relative_absolute_error(predobs)
    if logger is None:
        print "Done in {} seconds".format(time.time() - start)
        print "RMSE: {}, variance explained: {}, mean absolute error: {},".\
                    format(metrics.explainedVariance,\
                    metrics.rootMeanSquaredError,
                    metrics.meanAbsoluteError)
        print "MRAE: {}".format(mrae)
    else:
        logger.debug("Done in {} seconds".format(time.time() - start))
        logger.debug("RMSE: {}, variance explained: {}, mean absolute error: {},".\
                    format(metrics.explainedVariance,\
                    metrics.rootMeanSquaredError,
                    metrics.meanAbsoluteError))
        logger.debug("MRAE: {}".format(mrae))
    res = {"mre": metrics.meanAbsoluteError, "mrae": mrae}
    return res

def mean_feature_values(features, logger):
        logger.debug("Computing mean feature values")
        start = time.time()
        mean_p_feat_vals = features\
                .values()\
                .reduce(lambda x, y: (map(sum, zip(x, y))))
        nmovies = float(features.count())
        mpfv = {x: mean_p_feat_vals[x]/nmovies for x in xrange(len(mean_p_feat_vals))}
        logger.debug("Done in {} seconds".format(time.time() - start))
        logger.debug("Mean product feature values: {}".format(mpfv))
        return mpfv

def replace_feature_with_predicted(features, f, predictions, logger):
    logger.debug("Replacing original feature "+\
            "{} with predicted values".format(f))
    start = time.time()
    joined = features.join(predictions)
    replaced = joined.map(lambda (mid, (feats, pred)):
            (mid, set_list_values(feats, f, pred)))
    logger.debug("Done in {} seconds".format(time.time() - start))
    return replaced

def compare_baseline_to_replaced(baseline_predictions, uf, pf, logger, power):
                start = time.time()
                replaced_predictions = manual_predict_all(
                    training.map(lambda x: (x[0], x[1])),
                    uf,
                    pf)
                logger.debug("Done in {} seconds".format(time.time() - start))

                logger.debug("Computing replaced mean error relative to the "+\
                    "baseline model")
                start = time.time()
                predictionsAndRatings = replaced_predictions.map(lambda x: ((x[0], x[1]), x[2])) \
                   .join(baseline_predictions.map(lambda x: ((x[0], x[1]), x[2]))) \
                   .values()
                replaced_mean_error_baseline = mean_error(predictionsAndRatings, power)
                logger.debug("Done in {} seconds".format(time.time() - start))
                return replaced_mean_error_baseline

def internal_feature_predictor(sc, training, rank, numIter, lmbda,
        args, all_movies, metadata_sources, user_or_product_features, eval_regression,
        compare_with_replaced_feature, compare_with_randomized_feature, logger):

        results = {}
        power = 1.0

        if "average_rating" in args.metadata_sources:
            arc = AverageRatingRecommender(logger)
            arc.train(training)
            arc_ratings = sc.parallelize(arc.ratings.items())
            metadata_sources.append(
                    {"name": "average_rating",
                        "src_rdd": lambda: arc_ratings,
                        "loader": load_average_ratings})

        cur_mtdt_srcs = filter(lambda x: x["name"] in args.metadata_sources, metadata_sources)
        indicators, number_of_features, categorical_features =\
                build_meta_data_set(sc, cur_mtdt_srcs, all_movies, logger)

        if logger is None:
            print "Training ALS recommender"
        else:
            logger.debug("Training ALS recommender")
        start = time.time()
        model = ALS.train(training, rank=rank, iterations=numIter,
                lambda_=lmbda, nonnegative = args.non_negative)
        if logger is None:
            print "Done in {} seconds".format(time.time() - start)
        else:
            logger.debug("Done in {} seconds".format(time.time() - start))

        logger.debug("Fetching all products from the training set")
        start = time.time()
        training_set_products = set(training.map(lambda x: x[1]).collect())
        logger.debug("Done in {} seconds".format(time.time() - start))
        logger.debug("{} products collected"\
                        .format(len(training_set_products)))

        logger.debug("Fetching all products in model")
        start = time.time()
        model_products = set(model.productFeatures().keys().collect())
        logger.debug("Done in {} seconds".format(time.time() - start))
        logger.debug("{} products collected"\
                        .format(len(model_products)))
        logger.debug("{} products are missing"\
                        .format(len(training_set_products - model_products)))

        if compare_with_replaced_feature or compare_with_randomized_feature:
            logger.debug("Computing model predictions")
            start = time.time()
            baseline_predictions = model.predictAll(training.map(lambda x: (x[0], x[1])))
            logger.debug("Done in {} seconds".format(time.time() - start))
            logger.debug("Computing mean error")
            start = time.time()
            predictionsAndRatings = baseline_predictions.map(lambda x: ((x[0], x[1]), x[2])) \
                .join(training.map(lambda x: ((x[0], x[1]), x[2]))) \
                .values()
            baseline_mean_error = mean_error(predictionsAndRatings, power)
            baseline_rmse = mean_error(predictionsAndRatings, power=2.0)
            logger.debug("Done in {} seconds".format(time.time() - start))
            logger.debug("Mean error: {}, RMSE: {}".format(baseline_mean_error,
                baseline_rmse))
            results["baseline_mean_error"] = baseline_mean_error
            results["baseline_rmse"] = baseline_rmse

        features = model.productFeatures()
        other_features = model.userFeatures()
        if user_or_product_features == "user":
            features, other_features = other_features, features

        results["mean_feature_values"] = mean_feature_values(features, logger)

        results["features"] = {}

        for f in xrange(rank):
            lr_model, observations, predictions = predict_internal_feature(
                    features,
                    indicators,
                    f,
                    regression_model,
                    categorical_features,
                    nbins,
                    logger)
            results["features"][f] = {"model": lr_model}

            if eval_regression:
                reg_eval = evaluate_regression(predictions, observations, logger)
                results["features"][f]["regression_evaluation"] = reg_eval

            if compare_with_replaced_feature:
                logger.debug("Computing predictions of the model with replaced "+\
                    "feature {}".format(f))
                replaced_features = replace_feature_with_predicted(features, f,
                        predictions, logger)

                #logger.debug("Fetching all replaced product features")
                start = time.time()
                #replaced_products = set(replaced_features.keys().collect())
                #logger.debug("Done in {} seconds".format(time.time() - start))
                #logger.debug("{} products collected"\
                        #        .format(len(replaced_products)))
                #logger.debug("{} products are missing"\
                        #        .format(len(training_set_products - replaced_products)))

                if user_or_product_features == "product":
                    uf, pf = other_features, replaced_features
                else:
                    uf, pf = replaced_features, other_features

                replaced_mean_error_baseline = compare_baseline_to_replaced(
                        baseline_predictions, uf, pf, logger, power)

                logger.debug("Replaced mean error baseline: "+\
                    "{}".format(replaced_mean_error_baseline))
                results["features"][f]["replaced_mean_error_baseline"] =\
                    replaced_mean_error_baseline

            if compare_with_randomized_feature:
                logger.debug("Randomizing feature {}".format(f))
                start = time.time()
                replaced_features = perturb_feature(features, f)
                logger.debug("Done in {} seconds".format(time.time()-start))
                if user_or_product_features == "product":
                    uf, pf = other_features, replaced_features
                else:
                    uf, pf = replaced_features, other_features

                logger.debug("Computing predictions of the model with randomized"+\
                    " feature {}".format(f))
                randomized_mean_error_baseline = compare_baseline_to_replaced(
                        baseline_predictions, uf, pf, logger, power)

                logger.debug("Radnomized mean error baseline: "+\
                    "{}".format(randomized_mean_error_baseline))
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

def load_average_ratings(src_rdd):
    ratings = src_rdd.map(lambda (x, y): (x, [y]))
    nof = 1
    cfi = {}
    return (ratings, nof, cfi)

def load_years(src_rdd, sep=","):
    years = src_rdd\
            .map(lambda x: parseYear(x, sep=sep))\
            .map(lambda (x, y): (x, [y]))
    nof = 1
    cfi = {}
    return (years, nof, cfi)

def load_genres(src_rdd, sep=",", parser_function=parseGenre):
    print src_rdd.take(1)
    genres = src_rdd.map(lambda x: parser_function(x, sep=sep))
    print genres.take(1)
    all_genres = sorted(list(genres.map(lambda (_, x): x).fold(set(), lambda x, y:
        set(x).union(set(y)))))
    print all_genres
    indicators_genres = genres.map(
            lambda (mid, cur_genres): (
                mid, map(lambda g: int(g in cur_genres), all_genres)
                )
            )
    nof = len(all_genres)
    cfi = {x: 2 for x in xrange(nof)}
    return (indicators_genres, nof, cfi)

def load_tags(src_rdd, sep=","):
    tags = src_rdd.map(lambda x: parseTag(x, sep=sep))
    all_tags = set(tags.map(lambda x: x[2]).collect())
    all_tags = sorted(list(all_tags))
    tags = tags\
            .groupBy(lambda x: (x[1], x[2]))\
            .map(lambda x: x[0])\
            .groupBy(lambda x: x[0])\
            .map(lambda (mid, data): (mid, set(d[1] for d in data)))
    indicators = tags.map(
                lambda (mid, cur_tags): (
                    mid, map(lambda t: int(t in cur_tags), all_tags)
                     )
                )
    nof = len(all_tags)
    cfi = {x: 2 for x in xrange(nof)}
    return (indicators, nof, cfi)

def load_users(src_rdd, sep=","):
    users = src_rdd\
            .map(lambda x: parseUser(x, sep=sep))\
            .map(lambda x: (x[0], x[1:]))
    nof = 4
    cfi = {}
    return (users, nof, cfi)

def logger_init():
    #http://stackoverflow.com/questions/14058453/making-python-loggers-output-all-messages-to-stdout-in-addition-to-log
    #http://stackoverflow.com/questions/8269294/python-logging-only-log-from-script
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

if __name__ == "__main__":

    logger = logger_init()

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
    parser.add_argument("--qii-iters", action="store", default=5, type=int,
            help="Number of iterations for QII algorithm. 5 by default")
    parser.add_argument("--num-iters-ls", action="store", default=5, type=int,
            help="Number of iterations for local sensitvity algorithm. " +\
                    "5 by default")
    parser.add_argument("--data-path", action="store",
            default="datasets/ml-1m/", type=str, help="Path to MovieLens " +\
                    "home directory. datasets/ml-1m/ by default")
    parser.add_argument("--ofname", action="store", default="Output.txt",
            type=str, help="File to write the output. " +\
                    "Output.txt by default")
    parser.add_argument("--checkpoint-dir", action="store",
            default="checkpoint", type=str, help="Path to checkpoint " +\
                    "directory. checkpoint by default")
    parser.add_argument("--num-users-ls", action="store", default=5, type=int,
            help="Number of users for whom local sensitivity is computed. " +\
                    "5 by default")
    parser.add_argument("--specific-user", action="store", type=int,
            help="user-id to compute recommendations for a specific user")
    parser.add_argument("--max-movies-per-user", action="store", default=0,
            type=int, help="Maximum number of movie ratings allowed per " +\
                    "user. If a user has more movies rated, they're " +\
                    "skipped, until a user with fewer movies is found. " +\
                    "0 (default) means no limit")
    parser.add_argument("--prominent-raters", action="store", default=0,
            type=int, help="If set to anything other than 0 (default), " +\
                    "display a given number of users who had rated " +\
                    "the highest number of movies, and then exit.")
    parser.add_argument("--recommendations-to-print", action="store",
            default=10, type=int, help="How many movie recommendations "+\
                    "to display. 10 by default.")
    parser.add_argument("--print-movie-names", action="store_true", help=\
            "If set, movie names will be printed instead of movie IDs")
    parser.add_argument("--perturb-specific-user", action="store", type=int, help=\
            "If set, instead of sampling random users to perturb for local " +\
            "sensitivity, a particular UID gets perturbed. If set, " +\
            "--num-iters-ls gets automatically set to 1")
    parser.add_argument("--recommendations-only", action="store_true", help=\
            "If set, only recommendations for a specific user (must be set)" +\
            " will be displayed")
    parser.add_argument("--recommendations-and-per-movie-qii",
            action="store_true", help=\
            "If set, only recommendations and per movie qii "+\
            "for a specific user (must be set) are computed")
    parser.add_argument("--per-movie-qiis-displayed", action="store",
            default=3, type=int, help="The number of per movie qii values "+\
            "to display if --recommendations-and-per-movie-qii is set")
    parser.add_argument("--genres-correlator", action="store_true", help=\
            "Corralting genres. Make better description!#TODO")

    parser.add_argument("--gui", action="store_true", help=\
            "Enables GUI visualtisations")

    parser.add_argument("--years-correlator", action="store_true", help=\
            "Correlating years.")
    parser.add_argument("--iterate-rank", action="store_true", help=\
            "Iterate through models with different ranks")
    parser.add_argument("--iterate-from", action="store", type=int, default=5,\
            help="Iterate the value from (5 by default)")
    parser.add_argument("--iterate-to", action="store", type=int, default=40,\
            help="Iterate the vaue to (40 by default)")
    parser.add_argument("--iterate-step", action="store", type=int, default=5,\
            help="Step for iteration (5 by default)")
    parser.add_argument("--invert-labels", action="store_true", help=\
            "Invert true and false labels for genre correlator")
    parser.add_argument("--no-threshold", action="store_true", help=\
            "Turn off thresholds for regression models, "+\
            "look at scores, and do model-wide evaluation")
    parser.add_argument("--classifier-model", action="store", type=str,
            default="logistic", help="Model used in genres-correlator. "+\
                    "Possible values: logistic, "+\
                    "decision_tree (no_threshold not available), "+\
                    "logistic by default")
    parser.add_argument("--regression-model", action="store", type=str,
            default="linear", help="Model used in genres-regression, "+\
                    "Possible values: linear, "+\
                    "regression_tree, "+\
                    "random_forest, "+\
                    "linear by default")
    parser.add_argument("--nbins", action="store", type=int, default=32, help=\
            "Number of bins for a regression tree. 32 by default. "+\
            "Maximum depth is ceil(log(nbins, 2)).")
    parser.add_argument("--mean-error-experiments", action="store_true", help=\
            "Do experiments with mean error and feature substitution")
    parser.add_argument("--internal-feature-influence", action="store_true",
            help="Compute the global influence of internal features")
    parser.add_argument("--compute-mean-error", action="store_true", help=\
            "Also compute the influence of features on the mean error "+\
            "on the training set")
    parser.add_argument("--compute-fast-influence", action="store_true", help=\
            "Use the fast method for computing feature global influence")
    parser.add_argument("--sample-type", action="store", type=str,\
            default="training", help="What kind of sample of users and "+\
            "movies to use for global influence estimation. "+\
            "Possible values: training, random. If random is selected, "+\
            "--sample-size can be specified. training by default")
    parser.add_argument("--sample-size", action="store", type=int,\
            default=10000, help="Random sample size for feature global "+\
            "influence. 10000 by default")
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
    rank = args.rank
    lmbda = args.lmbda
    numIter = args.num_iter
    numPartitions = args.num_partitions
    qii_iters = args.qii_iters
    num_iters_ls = args.num_iters_ls
    movieLensHomeDir = args.data_path
    ofname = args.ofname
    checkpoint_dir = args.checkpoint_dir
    num_users_ls = args.num_users_ls
    specific_user = args.specific_user
    max_movies_per_user = args.max_movies_per_user
    prominent_raters = args.prominent_raters
    recommendations_to_print = args.recommendations_to_print
    print_movie_names = args.print_movie_names
    perturb_specific_user = args.perturb_specific_user
    if perturb_specific_user:
        num_iters_ls = 1
    recommendations_only = args.recommendations_only
    recommendations_and_per_movie_qii = args.recommendations_and_per_movie_qii
    per_movie_qiis_displayed = args.per_movie_qiis_displayed
    genres_correlator = args.genres_correlator
    gui = args.gui
    years_correlator = args.years_correlator
    iterate_rank = args.iterate_rank
    iterate_from = args.iterate_from
    iterate_to = args.iterate_to
    iterate_step = args.iterate_step
    invert_labels = args.invert_labels
    no_threshold = args.no_threshold
    classifier_model = args.classifier_model
    if classifier_model == "decision_tree":
        no_threshold = False
    regression_model = args.regression_model
    nbins = args.nbins
    mean_error_experiments = args.mean_error_experiments
    internal_feature_influence = args.internal_feature_influence
    compute_mean_error = args.compute_mean_error
    compute_fast_influence = args.compute_fast_influence
    sample_type = args.sample_type
    sample_size = args.sample_size
    regression_users = args.regression_users

    print "Rank: {}, lmbda: {}, numIter: {}, numPartitions: {}".format(
        rank, lmbda, numIter, numPartitions)
    print "qii_iters: {}, num_iters_ls: {}, movieLensHomeDir: {}".format(
        qii_iters, num_iters_ls, movieLensHomeDir)
    print "ofname: {}, checkpoint_dir: {}, num_users_ls:{}".format(
        ofname, checkpoint_dir, num_users_ls)
    print "specific_user: {}, max_movies_per_user: {}, prominent_raters:{}".\
            format(specific_user, max_movies_per_user, prominent_raters)
    print "perturb_specific_user: {}, recommendations_only:{}".\
            format(perturb_specific_user, recommendations_only)
    print "recommendations_and_per_movie_qii: {}".format(recommendations_and_per_movie_qii)
    print "per_movie_qiis_displayed: {}".format(per_movie_qiis_displayed)
    print "genres_correlator: {}, gui: {}, years_correlator: {}".format(
            genres_correlator, gui, years_correlator)
    print "iterate_rank: {}, iterate_from: {}, iterate_to: {}, iterate_step:{}".\
            format(iterate_rank, iterate_from, iterate_to, iterate_step)
    startconfig = time.time()
    print "invert_labels: {}, no_threshold: {}".format(invert_labels,
            no_threshold)
    print "classifier_model: {}".format(classifier_model)
    print "regression_model: {}".format(regression_model)
    print "mean_error_experiments: {}".format(mean_error_experiments)
    print "internal_feature_influence: {}".format(internal_feature_influence)
    print "compute_mean_error: {}".format(compute_mean_error)
    print "compute_fast_influence: {}".format(compute_fast_influence)
    print "sample_size: {}, sample_type: {}".format(sample_size, sample_type)
    print "nbins: {}".format(nbins)
    print "regression_users: {}".format(regression_users)
    print "predict_product_features: {}".format(args.predict_product_features)
    print "metadata_sources: {}".format(args.metadata_sources)

    if gui:
        import matplotlib.pyplot as plt


    # set up environment
    conf = SparkConf() \
      .setMaster("local[*]") \
      .setAppName("MovieLensALS") \
      .set("spark.executor.memory", "2g")
    sc = SparkContext(conf=conf)

######################################## Fixes Stack Overflow issue when training ALS
    sc.setCheckpointDir(checkpoint_dir)
    ALS.checkpointInterval = 2
#######################################

    if "ml-20m" in movieLensHomeDir:
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
            loadCSV(
                join(movieLensHomeDir, "ratings" + extension),
                remove_first_line = remove_first_line
                )
            )
    ratings = ratings_rdd.map(lambda x: parseRating(x,
         sep=sep))
    logger.debug("Done in {} seconds".format(time.time() - start))

    logger.debug("Loading movies")
    start = time.time()
    movies_rdd = sc.parallelize(
            loadCSV(
                join(movieLensHomeDir, "movies" + extension),
                remove_first_line = remove_first_line
                )
            )
    movies = dict(movies_rdd.map(lambda x: parseMovie(x,
        sep=sep)).collect())
    all_movies = set(movies.keys())
    logger.debug("Done in {} seconds".format(time.time() - start))
    logger.debug("{} movies loaded".format(len(all_movies)))


    metadata_sources = [
        {
            "name": "years",
            "src_rdd": (lambda: movies_rdd),
            "loader": (lambda x: load_years(x, sep))
        },
        {
            "name": "genres",
            "src_rdd": (lambda: movies_rdd),
            "loader": (lambda x: load_genres(x, sep))
        },
        {
            "name": "tags",
            "src_rdd": (
                lambda: sc.parallelize(
                    loadCSV(
                        join(movieLensHomeDir, "tags" + extension),
                        remove_first_line = remove_first_line
                    )
                )
            ),
            "loader": (lambda x: load_tags(x, sep))
        },
        {
            "name": "imdb_keywords",
            "src_rdd": (
                lambda: sc.parallelize(
                    loadCSV(
                        join(movieLensHomeDir, "ml-20m.imdb.small.csv"),
                        remove_first_line = True
                    )
                )
            ),
            "loader": (lambda x: load_genres(x, sep=",",
                parser_function=parseIMDBKeywords))
        }
    ]


    # create the initial training dataset with default ratings
    training = ratings\
      .filter(lambda x: (x[1][1] in all_movies) and (True or x[0] < 3))\
      .values() \
      .repartition(numPartitions) \
      .cache()

    logger.debug("{} records in the training set".format(training.count()))
    logger.debug("{} unique movies in the training set"\
            .format(len(set(training.map(lambda x: x[1]).collect()))))

    if prominent_raters > 0:
        UsersWithMostRatingslist =\
            users_with_most_ratings(training,prominent_raters)
        t = PrettyTable(["User ID", "Movies rated"])
        for uid, nm in UsersWithMostRatingslist:
            t.add_row([uid, nm])
        print t

    elif recommendations_only or recommendations_and_per_movie_qii:
        if not specific_user:
            print "Specific user must be set for this to work"
        ratings = [pyspark.mllib.recommendation.Rating(*x) for x in get_ratings_from_uid(
                        training, specific_user).collect()]
        print_top_recommendations(ratings, all_ratings=True)
        if recommendations_only:
            recommendations, model = compute_recommendations_and_qii(sc, training,
                    specific_user, dont_compute_qii=True)
            print_top_recommendations(recommendations, recommendations_to_print)
        elif recommendations_and_per_movie_qii:
            recommendations, qii = compute_recommendations_and_qii(sc, training,
                    specific_user, per_movie=True)
            print_top_recommendations(recommendations,
                    recommendations_to_print, movie_qiis = qii,
                    movie_qiis_to_display = per_movie_qiis_displayed)
    elif years_correlator:
        print "Loading years"
        start = time.time()
        years = sc.textFile(join(movieLensHomeDir, "movies.dat")).map(parseYear)
        print "Done in {} seconds".format(time.time() - start)
        print "Training model"
        start = time.time()
        model = ALS.train(training, rank, numIter, lmbda)
        print "Done in {} seconds".format(time.time() - start)
        print "Preparing features"
        start = time.time()
        features = model.productFeatures()
        data = features.join(years).map(
                lambda (mid, (ftrs, yr)):
                        LabeledPoint(yr, ftrs))
        print "Done in {} seconds".format(time.time() - start)
        if regression_model == "regression_tree":
            lr_model = pyspark.\
                    mllib.\
                    tree.\
                    DecisionTree.\
                    trainRegressor(
                            data,
                            categoricalFeaturesInfo={},
                            impurity = "variance",
                            maxDepth = int(math.ceil(math.log(nbins, 2))),
                            maxBins = nbins)
        elif regression_model == "linear":
            print "Building linear regression"
            start = time.time()
            lr_model = LinearRegressionWithSGD.train(data)
        print "Done in {} seconds".format(time.time() - start)
        observations = data.map(lambda x: x.label)
        predictions = lr_model.predict(data.map(lambda x:
                x.features))
        predobs = predictions.zip(observations).map(lambda (a, b): (float(a),
            float(b)))
        metrics = RegressionMetrics(predobs)
        print "RMSE: {}, variance explained: {}, mean absolute error: {}".\
                format(metrics.explainedVariance,\
                metrics.rootMeanSquaredError,
                metrics.meanAbsoluteError)
        if regression_model == "linear":
            print "Weights: {}".format(lr_model.weights)
        elif regression_model == "regression_tree":
            print lr_model.toDebugString()

    elif regression_users:
        args.metadata_sources = ["users"]
        logger.debug("Loading users")
        start = time.time()
        users_rdd = sc.parallelize(
            loadCSV(
                join(movieLensHomeDir, "users" + extension),
                remove_first_line = remove_first_line
                )
            )
        users = load_users(users_rdd, sep)
        all_users = set(users[0].keys().collect())
        logger.debug("Done in {} seconds".format(time.time() - start))
        metadata_sources = [
            {
                "name": "users",
                "src_rdd": (lambda: users),
                "loader": (lambda x: x)
            }]
        results = internal_feature_predictor(sc, training, rank, numIter, lmbda,
            args, all_users, metadata_sources,
            user_or_product_features="user", eval_regression = True,
            compare_with_replaced_feature = True,
            compare_with_randomized_feature = True, logger = logger)

        display_internal_feature_predictor(results, logger)

    elif args.predict_product_features:
        results = internal_feature_predictor(sc, training, rank, numIter, lmbda,
            args, all_movies, metadata_sources,
            user_or_product_features="product", eval_regression = True,
            compare_with_replaced_feature = True,
            compare_with_randomized_feature = True, logger = logger)

        display_internal_feature_predictor(results, logger)

    elif genres_correlator:
        print "Loading genres"
        start = time.time()
        if "ml-20m" in movieLensHomeDir:
            mv_file = sc.parallelize(loadCSV(join(movieLensHomeDir, "movies.csv")))
            genres = mv_file.map(lambda x: parseGenre(x, sep=","))
        else:
            genres = sc.textFile(join(movieLensHomeDir, "movies.dat")).map(parseGenre)
        print "Done in {} seconds".format(time.time() - start)

        if iterate_rank:
            results = []
            for rank in xrange(iterate_from, iterate_to+1, iterate_step):
                print "Processing rank", rank
                start = time.time()
                reg_models_res, avgbetter = correlate_genres(sc, genres, movies,
                        training, rank, numIter, lmbda, invert_labels,
                        no_threshold, classifier_model=classifier_model)
                reg_models_res = dict(reg_models_res)
                results.append({"rank": rank,
                                "reg_models_res": reg_models_res,
                                "avgbetter": avgbetter})
                print "Done in {} seconds".format(time.time() - start)
            genre_averages = defaultdict(lambda: 0.0)
            for datum in results:
                genre_averages["Average of all"] += datum["avgbetter"]
                for genre, d in datum["reg_models_res"].items():
                    if no_threshold:
                        genre_averages[genre] += d["better"]
                    else:
                        genre_averages[genre] += d["recall"]
            genre_averages = {k: v/float(len(results)) for k, v in
                    genre_averages.items()}
            avgall = genre_averages["Average of all"]
            del genre_averages["Average of all"]
            genre_averages_lst = genre_averages.items()
            genre_averages_lst.sort(key=lambda x: -x[1])
            genre_averages_lst = [("Average of all", avgall)] +\
                genre_averages_lst
            title = ["Genre"] + ["rank: {}".format(x["rank"]) for x in results]
            table = PrettyTable(title)
            for cur_genre, avg in genre_averages_lst:
                if no_threshold:
                    row = ["{} (AVG: {:1.3f})".format(cur_genre, avg)]
                    if cur_genre == "Average of all":
                        row += ["{:1.3f}".format(x["avgbetter"]) for x in results]
                    else:
                        row += ["{:3.1f}%".format(
                            x["reg_models_res"][cur_genre]\
                            ["better" if no_threshold else "recall"]*100)
                            for x in results]
                else:
                    row = ["{} (AVG: {:3.1f}%)".format(cur_genre, avg*100)]
                    if cur_genre == "Average of all":
                        row += ["{:3.1f}%".format(x["avgbetter"]*100) for x in results]
                    else:
                        row += ["{:3.1f}%".format(
                            x["reg_models_res"][cur_genre]\
                            ["better" if no_threshold else "recall"]*100)
                            for x in results]
                table.add_row(row)
            table.align["Genre"] = "r"
            print table

        else:
            reg_models_res, avgbetter = correlate_genres(sc, genres, movies,
                    training, rank, numIter, lmbda, invert_labels, no_threshold)

            for cur_genre, d in reg_models_res:
                row = (" "*3).join("{: 1.4f}".format(coeff)
                        for coeff in d["model"].weights)
                if no_threshold:
                    print "{:>12} (AuPRc: {:1.3f}, Prate: {:1.3f}, {:1.3f}x better) {}".\
                            format(cur_genre, d["auprc"], d["prate"], d["better"], row)
                else:
                    print "{:>12} (recall (tp/p): {:3.1f}%) {}".\
                            format(cur_genre, d["recall"]*100, row)
            if no_threshold:
                print "Average recall: {:3.1f}%".format(avgbetter*100)

        if gui:
            if iterate_rank:
                colors = ['k', 'r', 'g', 'b', 'y']
                styles = ['-', '--', '-.']
                markers = ["o", "^", "s"]
                csms = [(c, s, m) for m in markers for s in styles for c in colors]
                fig, ax = plt.subplots()
                ranks = [x["rank"] for x in results]
                for i in xrange(len(genre_averages_lst)):
                    color, style, marker = csms[i]
                    cur_genre, avg = genre_averages_lst[i]
                    if cur_genre == "Average of all":
                        avgs = [x["avgbetter"]*(1 if no_threshold else 100) for x in results]
                        lw = 2
                    else:
                        avgs = [x["reg_models_res"][cur_genre]\
                                ["better" if no_threshold else "recall"]*\
                                (1 if no_threshold else 100)
                                for x in results]
                        lw = 1
                    if no_threshold:
                        line_label = "{} (AVG: {:1.3f})".format(cur_genre, avg)
                    else:
                        line_label = "{} (AVG: {:3.1f}%)".format(cur_genre, avg*100)
                    ax.plot(ranks, avgs, color = color, linestyle=style,
                            label = line_label, marker=marker, lw=lw)
                legend = ax.legend(loc="center left", bbox_to_anchor=(1,0.5))
                ax.set_xticks(ranks)
                ax.set_xticklabels(ranks)
                ax.set_xlabel("Rank")
                if no_threshold:
                    ax.set_ylabel("Quality of {}".format(classifier_model))
                else:
                    if invert_labels:
                        ax.set_ylabel("Inverted recall (tn/n), %")
                    else:
                        ax.set_ylabel("Recall (tp/p), %)")
                ax.set_title("Performance of {} ".format(classifier_model) +\
                     ("with inverted labels " if invert_labels else "") +\
                     "from movie matrix to genres")
                plt.show()
            else:
                matrix = [list(x["model"].weights) for _, x in reg_models_res]
                matrix = numpy.array(matrix)
                fig, ax = plt.subplots()
                cax = ax.imshow(matrix, cmap='viridis', interpolation='nearest')
                ax.set_yticks(range(len(reg_models_res)))
                if no_threshold:
                    ax.set_yticklabels("{} ({:1.1f}x)".format(x,\
                       d["better"]) for x, d in reg_models_res)
                else:
                    ax.set_yticklabels("{} ({:3.1f}% recall)".format(x,\
                            d["recall"]*100) for x, d in reg_models_res)
                ax.set_ylabel("Genre")
                ax.set_xticks(range(len(reg_models_res[0][1]["model"].weights)))
                ax.set_xticklabels(range(len(reg_models_res[0][1]["model"].weights)))
                ax.set_xlabel("Product Features")
                ax.set_title("Coefficients for logistic regression"+\
                        (" with inverted labels" if invert_labels else "") +\
                        (" ({:1.3f} times better than random on average)"
                            if no_threshold
                            else "({:3.1f}% true positives)").\
                        format(avgbetter * (1 if no_threshold else 100)))
                cbar = fig.colorbar(cax)
                plt.show()

    elif internal_feature_influence:
        if sample_type == "training":
            user_product_pairs = training.map(lambda x: (x[0], x[1]))
        elif sample_type == "random":
            all_movies = list(set(training.map(lambda x: x[1]).collect()))
            all_users = list(set(training.map(lambda x: x[0]).collect()))
            pairs = [(random.choice(all_users), random.choice(all_movies)) for\
                    _ in xrange(sample_size)]
            user_product_pairs = sc.parallelize(pairs)
        print "Training model"
        start = time.time()
        model = ALS.train(training, rank, numIter, lmbda)
        print "Done in {} seconds".format(time.time() - start)
        infs = feature_global_influence(model, rank, user_product_pairs, 1.0,
                compute_mean_error, training, compute_fast_influence)
        if compute_mean_error:
            print "Mean error of the original model:", infs["original_rmse"]
        all_features = infs["feature_data"].keys()
        all_features.sort(key=lambda x: -infs["feature_data"][x]["model_diff"])
        title = ["Feature", "Global influence"]
        if compute_mean_error:
            title += ["Perturbed mean error", "delta mean error"]
        table = PrettyTable(title)
        for f in all_features:
            row = [f, infs["feature_data"][f]["model_diff"]]
            if compute_mean_error:
                row += [infs["feature_data"][f]["rmse"],
                        infs["feature_data"][f]["rmse"] -\
                        infs["original_rmse"]]
            table.add_row(row)
        print table
    else:
        endconfig = time.time()

        startfunction = time.time()

        if specific_user is not None:
            res = compute_user_local_sensitivity(sc, training, specific_user,
                    num_iters_ls)
            res = [res]
        else:
            res = compute_multiuser_local_sensitivity(sc, training, num_iters_ls,
                num_users_ls)

        endfunction = time.time()

        print("config time: " + str(endconfig - startconfig))
        print("function time: " + str(endfunction - startfunction))

        print "Result:", res
        out_file = open(ofname, "w")
        out_file.write("result: %s\n" % str(res))
        out_file.write("config time: \n" + str(endconfig - startconfig))
        out_file.write("function time: \n" + str(endfunction - startfunction))
        out_file.close()


    sc.stop()
