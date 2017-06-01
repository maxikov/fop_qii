#standard library
import argparse
import os.path
import random

#project files
import rating_explanation
import tree_qii

#pyspark library
from pyspark import SparkConf, SparkContext

#numpy library
import numpy.random

def generate_profiles(results, n_profiles):
    catfs = results["categorical_features"].keys()
    random.shuffle(catfs)
    profiles = {}
    for i in xrange(0, len(catfs), 2):
        pos, neg = catfs[i], catfs[i+1]
        lp = len(profiles)
        profiles[lp] = {"profile_id": lp,
                        "neg": neg,
                        "neg_name": results["feature_names"][neg],
                        "pos": pos,
                        "pos_name": results["feature_names"][pos],
                       }
        if len(profiles) >= n_profiles:
            break
    return profiles

def generate_users(profiles, n_users):
    users = {}
    for uid in xrange(n_users):
        users[uid] = random.choice(profiles.keys())
    return users

def generate_one_user_ratings(uid, user_profile, n_ratings, all_movies, indicators):
    ratings = []
    my_movies = list(all_movies)
    random.shuffle(my_movies)
    my_movies = my_movies[:n_ratings]
    for movie in my_movies:
        inds = indicators.lookup(movie)[0]
        rating = 3
        if inds[user_profile["pos"]] == 1:
            rating += 1
        if inds[user_profile["neg"]] == 1:
            rating -= 1
        ratings.append( (uid, movie, rating) )
    return ratings



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist-dir", action="store", type=str, help=\
                        "Path from which to import movie indicators and "+\
                        "feature names")
    parser.add_argument("--n-profiles", action="store", type=int, help=\
                        "Number of user preference profiles to generate")
    parser.add_argument("--n-users", action="store", type=int, help=\
                        "Number of users to generate")
    parser.add_argument("--mu", action="store", type=float, help=\
                        "mu of the lognormal distribution of the "+\
                        "number of each user's ratings")
    parser.add_argument("--sigma", action="store", type=float, help=\
                        "sigma of the lognormal distribution of the "+\
                        "number of each user's ratings")
    args = parser.parse_args()
    conf = SparkConf().setMaster("local[*]")
    sc = SparkContext(conf=conf)

    print "Loading results dict"
    results = rating_explanation.load_results_dict(args.persist_dir)
    print len(results["categorical_features"]), "categorical features found"
    profiles = generate_profiles(results, args.n_profiles)
    print len(profiles), "profiles generated"
    print profiles
    users = generate_users(profiles, args.n_users)
    print len(users), "users generated"
    print users
    (training_movies, test_movies, features_test, features_training,
             features_original_test, features_original_training,
             indicators_training, indicators_test) =\
        tree_qii.load_features_indicators(os.path.join(args.persist_dir,
                                              "features_training_test.pkl"), sc, 7)
    indicators = indicators_training.union(indicators_test).sortByKey().cache()
    all_movies = indicators.keys().collect()
    print len(all_movies), "movies in the data set"
    ratings = generate_one_user_ratings(1, profiles[users[1]], 20, all_movies, indicators)
    print ratings
    for r in ratings:
        print r

if __name__ == "__main__":
    main()
