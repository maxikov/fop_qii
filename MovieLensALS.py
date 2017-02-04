#!/usr/bin/env python

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

from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS
import pyspark.mllib.recommendation
from pyspark.mllib.classification import LabeledPoint
import pyspark.mllib.regression
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.evaluation import RegressionMetrics,\
        BinaryClassificationMetrics

# Global variables

# Arguments to the ALS training function
rank = 12
lmbda = 0.1
numIter = 20

# Number of partitions created
numPartitions = 4
qii_iters = 5
num_iters_ls = 5

max_movies_per_user = 0 #0 = no limit

recommendations_to_print = 0 # 0 = don't print

print_movie_names = False

perturb_specific_user = None

def parseRating(line):
    """
    Parses a rating record in MovieLens format userId::movieId::rating::timestamp .
    """
    fields = line.strip().split("::")
    return long(fields[3]) % 10, (int(fields[0]), int(fields[1]), float(fields[2]))

def parseGenre(line):
    """
    Parses movie genres in MovieLens format
    movieId::movieTitle::movieGenre1[|movieGenre2...]
    """
    fields = line.strip().split("::")
    mid = int(fields[0])
    genres = fields[2]
    genres = genres.split("|")
    return mid, set(genres)

def parseYear(line):
    """
    Parses movie years in MovieLens format
    movieId::movieTitle (movieYear)::movieGenre1[|movieGenre2...]
    """
    fields = line.strip().split("::")
    mid = int(fields[0])
    mtitle = fields[1]
    year = mtitle.split("(")[-1]
    year = year.strip(")")
    year = int(year)
    return mid, year

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
                invert_labels=False, no_threshold=False):
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
            lr_model = pyspark.\
                    mllib.\
                    classification.\
                    LogisticRegressionWithLBFGS.\
                    train(lr_data)
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
        return reg_models_res, avgbetter


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=u"Usage: " +\
            "/path/to/spark/bin/spark-submit --driver-memory 2g " +\
            "MovieLensALS.py [arguments]")

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

    if gui:
        import matplotlib.pyplot as plt


    # set up environment
    conf = SparkConf() \
      .setAppName("MovieLensALS") \
      .set("spark.executor.memory", "2g")
    sc = SparkContext(conf=conf)

######################################## Fixes Stack Overflow issue when training ALS
    sc.setCheckpointDir(checkpoint_dir)
    ALS.checkpointInterval = 2
#######################################

    print "Loading ratings"
    start = time.time()
    ratings = sc.textFile(join(movieLensHomeDir, "ratings.dat")).map(parseRating)
    print "Done in {} seconds".format(time.time() - start)
    print "Loading movies"
    start = time.time()
    movies = dict(sc.textFile(join(movieLensHomeDir, "movies.dat")).map(parseMovie).collect())
    print "Done in {} seconds".format(time.time() - start)

    # create the initial training dataset with default ratings
    training = ratings.filter(lambda x: True or x[0] < 6)\
      .values() \
      .repartition(numPartitions) \
      .cache()

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
        print "Weights: {}".format(lr_model.weights)

    elif genres_correlator:
        print "Loading genres"
        start = time.time()
        genres = sc.textFile(join(movieLensHomeDir, "movies.dat")).map(parseGenre)
        print "Done in {} seconds".format(time.time() - start)

        if iterate_rank:
            results = []
            for rank in xrange(iterate_from, iterate_to+1, iterate_step):
                print "Processing rank", rank
                start = time.time()
                reg_models_res, avgbetter = correlate_genres(sc, genres, movies,
                        training, rank, numIter, lmbda, invert_labels,
                        no_threshold)
                reg_models_res = dict(reg_models_res)
                results.append({"rank": rank,
                                "reg_models_res": reg_models_res,
                                "avgbetter": avgbetter})
                print "Done in {} seconds".format(time.time() - start)
            genre_averages = defaultdict(lambda: 0.0)
            for datum in results:
                genre_averages["Average of all"] += datum["avgbetter"]
                for genre, d in datum["reg_models_res"].items():
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
                row = ["{} (AVG: {:3.1f}%)".format(cur_genre, avg*100)]
                if cur_genre == "Average of all":
                    row += ["{:3.1f}%".format(x["avgbetter"]*100) for x in results]
                else:
                    row += ["{:3.1f}%".format(
                        x["reg_models_res"][cur_genre]["recall"]*100)
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
                        avgs = [x["avgbetter"]*100 for x in results]
                        lw = 2
                    else:
                        avgs = [x["reg_models_res"][cur_genre]["recall"]*100
                                for x in results]
                        lw = 1
                    line_label = "{} (AVG: {:3.1f}%)".format(cur_genre, avg*100)
                    ax.plot(ranks, avgs, color = color, linestyle=style,
                            label = line_label, marker=marker, lw=lw)
                legend = ax.legend(loc="center left", bbox_to_anchor=(1,0.5))
                ax.set_xticks(ranks)
                ax.set_xticklabels(ranks)
                ax.set_xlabel("Rank")
                if no_threshold:
                    ax.set_ylabel("Quality of logistic regression")
                else:
                    ax.set_ylabel("Recall (tp/p), %)")
                ax.set_title("Performance of logistic regression " +\
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
