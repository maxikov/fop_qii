#standard library
import argparse
import os
import os.path
import random
import time
import functools
import pickle

#project files
import rating_explanation
import tree_qii

#pyspark library
from pyspark import SparkConf, SparkContext

#numpy library
import numpy.random

def generate_profiles(results, n_profiles, semi_random=False,
        specific_features=None):
    if specific_features is not None:
        catfs = [f for (f,n) in results["feature_names"].items() if n in
                specific_features]
        print "Features requested:", sorted(specific_features)
        print "Features used:", sorted([results["feature_names"][x] for x in
            catfs])
        print "Features used:", catfs
        notfound = set(specific_features) - set(results["feature_names"].values())
        if len(notfound) > 0:
            print "Features", notfound, "not found"
    else:
        catfs = results["categorical_features"].keys()
    random.shuffle(catfs)
    profiles = {}
    profiles_semi_random = {}
    profiles_random = {}
    if semi_random:
        for i in xrange(0, len(catfs), 5):
            pos, neg = catfs[i], catfs[i+1]
            lp = len(profiles)
            profiles[lp] = {"profile_id": lp,
                            "neg": neg,
                            "neg_name": results["feature_names"][neg],
                            "pos": pos,
                            "pos_name": results["feature_names"][pos],
                           }
            if random.random > 0.5:
                pos, neg = catfs[i], catfs[i+2]
            else:
                pos, neg = catfs[i+2], catfs[i+1]
            profiles_semi_random[lp] = {"profile_id": lp,
                            "neg": neg,
                            "neg_name": results["feature_names"][neg],
                            "pos": pos,
                            "pos_name": results["feature_names"][pos],
                           }
            pos, neg = catfs[i+3], catfs[i+4]
            profiles_random[lp] = {"profile_id": lp,
                            "neg": neg,
                            "neg_name": results["feature_names"][neg],
                            "pos": pos,
                            "pos_name": results["feature_names"][pos],
                           }
            if len(profiles) >= n_profiles:
                break
        return profiles, profiles_semi_random, profiles_random
    else:
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
        return profiles, None, None


def generate_profile_movies(profiles, indicators):
    res = {}
    for upid, profile in profiles.items():
        filter_f = functools.partial(lambda profile, (mid, inds):
                                     inds[profile["neg"]] != 0 or\
                                     inds[profile["pos"]] != 0, profile)
        res[upid] = indicators.filter(filter_f).keys().collect()
    return res

def generate_users(profiles, n_users):
    users = {}
    for uid in xrange(n_users):
        users[uid] = random.choice(profiles.keys())
    return users

def generate_one_user_ratings(uid, user_profile, n_ratings, all_movies, indicators):
    my_movies = list(all_movies)
    random.shuffle(my_movies)
    my_movies = my_movies[:n_ratings]
    filter_f = functools.partial(lambda my_movies, (mid, _): mid in my_movies,
                                 my_movies)
    map_f = functools.partial(lambda user_profile, uid, (mid, inds):\
                                (uid, mid, 3 + inds[user_profile["pos"]] -\
                                inds[user_profile["neg"]],\
                                int(time.time())), user_profile, uid)
    ratings = indicators.filter(filter_f).map(map_f).collect()
    return ratings

def generate_all_user_ratings(users, user_profiles, mu, sigma, indicators):
    ratings = []
    items = sorted(users.items(), key=lambda x: x[0])
    profile_movies = generate_profile_movies(user_profiles, indicators)
    for uid, upid in items:
        profile = user_profiles[upid]
        n_ratings = int(numpy.random.lognormal(mu, sigma))
        print "creating {} ratings for user {} out of {}".format(n_ratings,
                                                                 uid,
                                                                 len(items))
        ratings += generate_one_user_ratings(uid, profile, n_ratings,
                                             profile_movies[upid], indicators)
    return ratings

def generate_random_user_ratings(n_users, mu, sigma, indicators):
    ratings = []
    all_movies = indicators.keys().collect()
    dist = [2 for _ in xrange(10)] + [3] + [4 for _ in xrange(10)]
    for uid in xrange(n_users):
        n_ratings = int(numpy.random.lognormal(mu, sigma))
        print "creating {} ratings for user {} out of {}".format(n_ratings,
                                                                 uid,
                                                                 n_users)
        my_movies = list(all_movies)
        random.shuffle(my_movies)
        my_movies = my_movies[:n_ratings]
        for movie in my_movies:
            ratings.append( (uid, movie, random.choice(dist), int(time.time())) )
    return ratings

def exact_histogram(ratings, values):
    hist = {v: 0 for v in values}
    for r in ratings:
        if r[2] in values:
            hist[r[2]] += 1
    return hist

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
    parser.add_argument("--random", action="store_true", help=\
                        "Generate profiles, but actually assign "+\
                        "ratings randomly")
    parser.add_argument("--odir", action="store", type=str, help=\
                        "Directory to save the generated data set")
    parser.add_argument("--semi-random", action="store_true", help=\
            "For each data set, also generate random and semi-random user"+\
            " profiles")
    parser.add_argument("--specific-features", type=str, nargs="*", help=\
            "Use specific metadata features instead of all available")
    args = parser.parse_args()
    conf = SparkConf().setMaster("local[*]")
    sc = SparkContext(conf=conf)

    if args.specific_features is not None:
        print "Using specific features:", args.specific_features
    print "Loading results dict"
    results = rating_explanation.load_results_dict(args.persist_dir)
    print len(results["categorical_features"]), "categorical features found"
    profiles, profiles_semi_random, profiles_random =\
        generate_profiles(results, args.n_profiles, args.semi_random,
                args.specific_features)
    print len(profiles), "profiles generated"
    users = generate_users(profiles, args.n_users)
    print len(users), "users generated"
    (training_movies, test_movies, features_test, features_training,
             features_original_test, features_original_training,
             indicators_training, indicators_test) =\
        tree_qii.load_features_indicators(os.path.join(args.persist_dir,
                                              "features_training_test.pkl"), sc, 7)
    print "indicators loaded"
    indicators = indicators_training.union(indicators_test).sortByKey().cache()
    print "indicators sorted"
    if args.random:
        ratings = generate_random_user_ratings(args.n_users, args.mu, args.sigma,
                                        indicators)
    else:
        ratings = generate_all_user_ratings(users, profiles, args.mu, args.sigma,
                                        indicators)
    print len(ratings), "ratings generated"

    values = [1, 2, 3, 4, 5]
    hist = exact_histogram(ratings, values)
    print ", ".join("{}: {}".format(v, hist[v]) for v in values)

    odir = args.odir
    try:
        os.mkdir(odir)
    except OSError as e:
        print e

    fname = os.path.join(odir, "args.pkl")
    ofile = open(fname, "wb")
    pickle.dump(args, ofile)
    ofile.close()

    fname = os.path.join(odir, "profiles.pkl")
    ofile = open(fname, "wb")
    pickle.dump((profiles, users), ofile)
    ofile.close()
    if args.semi_random:
        fname = os.path.join(odir, "profiles_semi_random.pkl")
        ofile = open(fname, "wb")
        pickle.dump((profiles_semi_random, users), ofile)
        ofile.close()
        fname = os.path.join(odir, "profiles_random.pkl")
        ofile = open(fname, "wb")
        pickle.dump((profiles_random, users), ofile)
        ofile.close()

    ratings = [("userId","movieId","rating","timestamp")] + ratings
    rating_strings = [",".join(map(str, x)) for x in ratings]
    str_res = "\n".join(rating_strings)
    fname = os.path.join(odir, "ratings.csv")
    ofile = open(fname, "w")
    ofile.write(str_res)
    ofile.close()

if __name__ == "__main__":
    main()
