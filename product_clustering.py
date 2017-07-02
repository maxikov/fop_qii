#standard library
import argparse
import os.path
import pickle
import random
import collections
import sys
import functools

#project files
import rating_explanation
import rating_qii
import tree_qii
import shadow_model_qii
import parsers_and_loaders
import common_utils
import explanation_correctness
import parsers_and_loaders

#pyspark libraryb
from pyspark import SparkConf, SparkContext
import pyspark
import pyspark.mllib.tree
from pyspark.mllib.classification import LabeledPoint
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.clustering import KMeans, GaussianMixture

#numpy library
import numpy as np

#sklearn library
import sklearn.cluster

class FitPredictWrapper(object):
    def __init__(self, sc, model, *args, **kwargs):
        self.model = model(*args, **kwargs)
        self.predictions = collections.defaultdict(lambda: 0.0)
        self.centroids = []
        self.res = None
        self.sc = sc

    def train(self, data, *args, **kwargs):
        X = np.array(data.collect())
        self.res = self.model.fit_predict(X, *args, **kwargs)
        n = X.shape[0]
        for i in xrange(n):
            self.predictions[tuple(map(float, X[i]))] = float(self.res[i])
        self.all_clusters = set(self.res.flatten())
        self.n_clusters = len(self.all_clusters)
        for i in xrange(self.n_clusters):
            self.centroids.append(X[self.res==i].mean(axis=0))
        return self

    def predict(self, data):
        X = data.collect()
        res = []
        for x in X:
            res.append(self.predictions[tuple(map(float, x))])
        res = self.sc.parallelize(res)
        return res

def dist(x, y):
    xa = np.array(x)
    ya = np.array(y)
    return np.linalg.norm(xa - ya)

def vsum(x, y):
    xa = np.array(x)
    ya = np.array(y)
    return xa + ya

def falmean(fal):
    rfun = lambda ((mid1, ftrs1), cls1), ((mid2, ftrs2), cls2):\
            vsum(ftrs1, ftrs2)
    res = fal.reduce(rfun)/float(fal.count())
    return res

def falvar(fal, mean=None):
    if mean is None:
        mean = falmean(fal)
    map_f = functools.partial(lambda mean, ((mid,ftrs),cls):
            dist(mean, ftrs)**2, mean)
    dists = fal.map(map_f)
    res = dists.reduce(lambda a, b: a+b)/float(dists.count())
    return res


def load_movies(sc, args, product_features=None):
    if args.movies_file is not None:
        if ".csv" in args.movies_file:
            msep = "," 
            movies_remove_first_line = True
        else:
            msep = "::"
            movies_remove_first_line = False
        movies_rdd = sc.parallelize(
            parsers_and_loaders.loadCSV(
                args.movies_file,
                remove_first_line=movies_remove_first_line
            )
            ).cache()
        movies_dict = dict(movies_rdd.map(lambda x: parsers_and_loaders.parseMovie(x,\
            sep=msep)).collect())
    else:
        movies = product_features.keys().collect()
        movies_dict = {x: str(x) for x in movies}
    return movies_dict

def test_train_split(data, test_ratio):
    n = data.count()
    randvec = [random.random() < test_ratio for _ in xrange(n)]
    data_ind = data.zipWithIndex()
    test_filter_f = functools.partial(lambda randvec, (_,\
        ind): randvec[ind], randvec)
    train_filter_f = functools.partial(lambda randvec, (_,\
        ind): not randvec[ind], randvec)
    training = data_ind.filter(train_filter_f).keys()
    test = data_ind.filter(test_filter_f).keys()
    return test, training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist-dir", action="store", type=str, help=\
                        "Path from which to loaad models and features to analyze")
    parser.add_argument("--movies-file", action="store", type=str, help=\
                        "CSV file with movie names")
    parser.add_argument("--n-clusters", action="store", type=int, default=10,
            help="Number of clusters to create")
    parser.add_argument("--max-depth", action="store", type=int, default=8,
                        help="Maximum depth of the decision tree")
    parser.add_argument("--impurity", action="store", type=str, default="gini",
                        help="Impurity function for tree")
    parser.add_argument("--user-profiles", action="store", type=str,
                        help="Path to the file with user profiles of a "+\
                        "synthetic data set (if applicable).")
    parser.add_argument("--cluster-model", action="store", type=str,
            default="kmeans", choices=["kmeans", "gmm", "spectral",
                "agglomerative", "birch"])
    parser.add_argument("--cluster-sample-size", action="store", type=int,
            default=10, help="Number of movies to display in each cluster")
    parser.add_argument("--test-ratio", action="store", type=float,
            default=0.3, help="Percent of the data set to use as a test set")
    args = parser.parse_args()
    conf = SparkConf().setMaster("local[*]")
    sc = SparkContext(conf=conf)

    print "Loading results dict"
    results = rating_explanation.load_results_dict(args.persist_dir)
    print "Loading indicators"
    (training_movies, test_movies, features_test, features_training,
             features_original_test, features_original_training,
             indicators_training, indicators_test) =\
        tree_qii.load_features_indicators(os.path.join(args.persist_dir,
                                              "features_training_test.pkl"),
                                              sc, 20)
    indicators = indicators_training.union(indicators_test).sortByKey().cache()
    products = indicators.keys().collect()
    products_set  = set(products)


    print "Loading ALS model"
    model = rating_qii.load_als_model(sc, os.path.join(args.persist_dir,
                                                       "als_model.pkl"))
    rank = model.rank

    product_features = model.productFeatures().filter(lambda (mid, _): mid in
            products_set).sortByKey().cache()

    print "Loading movies"
    movies_dict = load_movies(sc, args, product_features)
    print "Done,", len(movies_dict), "movies loaded"

    data_set = product_features.values()

    print "Training", args.cluster_model
    if args.cluster_model == "kmeans":
        kmeans_model = KMeans.train(data_set, args.n_clusters, maxIterations=100,
            initializationMode="random")
        centroids = list(map(np.array, kmeans_model.clusterCenters))
    elif args.cluster_model == "gmm":
        kmeans_model = GaussianMixture.train(data_set, args.n_clusters)
        centroids = [np.array(g.mu) for g in
                kmeans_model.gaussians]
    elif args.cluster_model == "spectral":
        kmeans_model = FitPredictWrapper(sc,
                sklearn.cluster.SpectralClustering, n_clusters=args.n_clusters)
        kmeans_model.train(data_set)
        centroids = kmeans_model.centroids
    elif args.cluster_model == "agglomerative":
        kmeans_model = FitPredictWrapper(sc,
                sklearn.cluster.AgglomerativeClustering, n_clusters=args.n_clusters)
        kmeans_model.train(data_set)
        centroids = kmeans_model.centroids
    elif args.cluster_model == "birch":
        kmeans_model = FitPredictWrapper(sc,
                sklearn.cluster.Birch, n_clusters=args.n_clusters)
        kmeans_model.train(data_set)
        centroids = kmeans_model.centroids

    print "Done"
    print "Centroids:", centroids
    print "Labeling clusters"
    clusters = kmeans_model.predict(data_set).map(float)
    features_and_labels = common_utils.safe_zip(product_features, clusters)
    cent_mean = np.mean(centroids, axis=0)
    cent_dists = [dist(cent_mean, cent)**2 for cent in centroids]
    cent_var = np.mean(cent_dists)
    print "Clusters:", clusters.countByValue()
    print "Centroid mean:", cent_mean
    print "Centroid variance:", cent_var
    cluster_data = []
    for cluster in xrange(args.n_clusters):
        filter_f = functools.partial(lambda cluster, ((mid, ftrs), cls): cls == cluster,
                cluster)
        cur_mvs = features_and_labels.filter(filter_f)
        centroid = centroids[cluster]
        key_f = functools.partial(lambda centroid, ((mid, ftrs), cls):
            dist(ftrs, centroid), centroid)
        sample = cur_mvs.takeOrdered(args.cluster_sample_size, key=key_f)
        text_sample = []
        for (i, ((mid, ftrs), cls)) in enumerate(sample):
            text_sample.append("\t{}: (dist: {}) {} (mid: {})".format(i, dist(ftrs, centroid),
                        movies_dict[mid], mid))
        cls_mean = centroid
        cls_var = falvar(cur_mvs, cls_mean)
        cluster_data.append((cluster, cls_var, text_sample))
    cluster_data.sort(key=lambda x: x[1])

    if args.user_profiles is not None:
        for (cluster, cls_var, text_sample) in cluster_data:
            print "Cluster {} (variance: {}):".format(cluster, cls_var)
            for s in text_sample:
                print s


    test_set, training_set = test_train_split(features_and_labels, args.test_ratio)
    test_features = test_set.map(lambda x: x[0][1])
    test_observations = test_set.map(lambda x: x[1])
    training_features = training_set.map(lambda x: x[0][1])
    training_observations = training_set.map(lambda x: x[1])
    all_features = features_and_labels.map(lambda x: x[0][1])

    latent_training_data = training_set.map(\
            lambda ((mid, ftrs), cls):\
                LabeledPoint(cls, ftrs))
    print "Training a latent tree"
    latent_tree = pyspark.\
                    mllib.\
                    tree.\
                    DecisionTree.\
                    trainClassifier(
                        latent_training_data,
                        numClasses=args.n_clusters,
                        categoricalFeaturesInfo={},
                        impurity=args.impurity,
                        maxDepth=args.max_depth)
    print latent_tree.toDebugString()
    print "Done, making predictions"
    tree_predictions  = latent_tree.predict(training_features).map(float)
    predobs = common_utils.safe_zip(tree_predictions, training_observations)
    acc = predobs.filter(lambda (x, y): x==y).count()\
            /float(predobs.count())
    print "Accuracy of latent tree on training set:", acc
    tree_predictions  = latent_tree.predict(test_features).map(float)
    predobs = common_utils.safe_zip(tree_predictions, test_observations)
    acc = predobs.filter(lambda (x, y): x==y).count()\
            /float(predobs.count())
    print "Accuracy of latent tree on test set:", acc

    print "Building meta training set"
    meta_data_set = common_utils.safe_zip(
            indicators, clusters)
    test_set, training_set = test_train_split(meta_data_set, args.test_ratio)
    test_features = test_set.map(lambda x: x[0][1])
    test_observations = test_set.map(lambda x: x[1])
    training_features = training_set.map(lambda x: x[0][1])
    training_observations = training_set.map(lambda x: x[1])
    all_features = features_and_labels.map(lambda x: x[0][1])

    meta_training_data = training_set.map(
                    lambda ((mid, inds), cls):
                    LabeledPoint(cls, inds))
    print "Training a meta tree"
    meta_tree = pyspark.\
                    mllib.\
                    tree.\
                    DecisionTree.\
                    trainClassifier(
                        meta_training_data,
                        numClasses=args.n_clusters,
                        categoricalFeaturesInfo=results["categorical_features"],
                        impurity=args.impurity,
                        maxDepth=args.max_depth)
    print common_utils.substitute_feature_names(
            meta_tree.toDebugString(),
            results["feature_names"])
    print "Done, making predictions"
    tree_predictions  = meta_tree.predict(training_features).map(float)
    predobs = common_utils.safe_zip(tree_predictions, training_observations)
    acc = predobs.filter(lambda (x, y): x==y).count()\
            /float(predobs.count())
    print "Accuracy of meta tree on training set:", acc
    tree_predictions  = meta_tree.predict(test_features).map(float)
    predobs = common_utils.safe_zip(tree_predictions, test_observations)
    acc = predobs.filter(lambda (x, y): x==y).count()\
            /float(predobs.count())
    print "Accuracy of meta tree on test set:", acc

    if args.user_profiles is None:
        used_features = tree_qii.get_used_features(meta_tree)
        print len(used_features), "features used"
        for (cluster, cls_var, text_sample) in cluster_data:
            filter_f = functools.partial(lambda cluster, ((mid, inds), cls):
                                cls == cluster, cluster)
            features = meta_data_set.filter(filter_f).keys()
            qiis = tree_qii.get_tree_qii(meta_tree, features, used_features)
            qiis_list = sorted(qiis.items(), key=lambda (f, q): -abs(q))
            qiis_list_names = [(results["feature_names"][f], q) for (f, q) in
                    qiis_list]
            print "Cluster {} (variance: {}):".format(cluster, cls_var)
            print "QIIs:", qiis_list_names
            for s in text_sample:
                print s
    else:
        used_features = tree_qii.get_used_features(meta_tree)
        print len(used_features), "features used"
        profiles, user_profiles = pickle.load(open(args.user_profiles,
                                                   "rb"))
        all_clusters = sorted(list(set(clusters.collect())))
        print len(all_clusters), "clusters found"
        all_corrs = []
        all_profiles = []
        ress = {}
        for cluster in all_clusters:
            print "Processing cluster", cluster
            filter_f = functools.partial(lambda cluster, ((mid, inds), cls):
                                cls == cluster, cluster)
            features = meta_data_set.filter(filter_f).keys()
            qiis = tree_qii.get_tree_qii(meta_tree, features, used_features)
            qiis_list = sorted(qiis.items(), key=lambda (f, q): -abs(q))
            qiis_list_names = [(results["feature_names"][f], q) for (f, q) in
                    qiis_list]
            print "QIIs:", qiis_list_names
            corrs = {pr: explanation_correctness.explanation_correctness(qiis,
                user_profile) for pr, user_profile in profiles.items()}
            corrs_list = sorted(corrs.items(), key=lambda x: -abs(x[1]))
            print "Correctness scores:", corrs_list
            all_corrs.append(corrs_list[0][1])
            all_profiles.append(corrs_list[0][0])
            top_profile, top_score = corrs_list[0]
            if top_profile not in ress:
                ress[top_profile] = {"cluster": cluster, "corr": top_score}
            else:
                if abs(ress[top_profile]["corr"]) < abs(top_score):
                        ress[top_profile] = {"cluster": cluster, "corr": top_score}
        print "All highers corrs:", all_corrs
        print "Their profiles:", all_profiles
        print len(set(all_profiles)), "profiles represented"
        print "Average:", float(sum(all_corrs))/len(all_corrs)
        print "Results:", ress
        print "Average corr:", sum(abs(x["corr"]) for x in
                ress.values())/float(len(ress))

        print "Making per-cluster trees"
        all_corrs = []
        all_profiles = []
        old_clusters = clusters
        ress = {}
        for cluster in all_clusters:
            print "Processing cluster", cluster
            clusters = old_clusters.map(functools.partial(lambda cluster, x:
                    1.0 if x == float(cluster) else 0.0, cluster))
            print "New clusters:", clusters.countByValue()
            print "Building meta training set"
            meta_data_set = common_utils.safe_zip(
                indicators, clusters)
            test_set, training_set = test_train_split(meta_data_set, args.test_ratio)
            test_features = test_set.map(lambda x: x[0][1])
            test_observations = test_set.map(lambda x: x[1])
            training_features = training_set.map(lambda x: x[0][1])
            training_observations = training_set.map(lambda x: x[1])
            all_features = features_and_labels.map(lambda x: x[0][1])

            meta_training_data = training_set.map(
                    lambda ((mid, inds), cls):
                    LabeledPoint(cls, inds))
            print "Training a meta tree"
            meta_tree = pyspark.\
                    mllib.\
                    tree.\
                    DecisionTree.\
                    trainClassifier(
                        meta_training_data,
                        numClasses=2,
                        categoricalFeaturesInfo=results["categorical_features"],
                        impurity=args.impurity,
                        maxDepth=args.max_depth)
            print common_utils.substitute_feature_names(
                meta_tree.toDebugString(),
                results["feature_names"])
            print "Done, making predictions"
            tree_predictions  = meta_tree.predict(training_features).map(float)
            predobs = common_utils.safe_zip(tree_predictions, training_observations)
            acc = predobs.filter(lambda (x, y): x==y).count()\
                    /float(predobs.count())
            print "Accuracy of cluster", cluster, "meta tree on training set:", acc
            evaluations = common_utils.evaluate_binary_classifier(
                tree_predictions.zipWithIndex().map(lambda (x,y):(y,x)),
                training_observations.zipWithIndex().map(lambda (x,y):(y,x)),
                None, no_threshold=False)
            print evaluations
            tree_predictions  = meta_tree.predict(test_features).map(float)
            predobs = common_utils.safe_zip(tree_predictions, test_observations)
            acc = predobs.filter(lambda (x, y): x==y).count()\
                    /float(predobs.count())
            print "Accuracy of cluster", cluster, "meta tree on test set:", acc
            evaluations = common_utils.evaluate_binary_classifier(
                tree_predictions.zipWithIndex().map(lambda (x,y):(y,x)),
                test_observations.zipWithIndex().map(lambda (x,y):(y,x)),
                None, no_threshold=False)
            print evaluations
            used_features = tree_qii.get_used_features(meta_tree)
            print len(used_features), "features used"
            if len(used_features) > 0:
                features = meta_data_set.keys()
                qiis = tree_qii.get_tree_qii(meta_tree, features, used_features)
                qiis_list = sorted(qiis.items(), key=lambda (f, q): -abs(q))
                qiis_list_names = [(results["feature_names"][f], q) for (f, q) in
                    qiis_list]
                print "QIIs:", qiis_list_names
                corrs = {pr: explanation_correctness.explanation_correctness(qiis,
                        user_profile) for pr, user_profile in profiles.items()}
                corrs_list = sorted(corrs.items(), key=lambda x: -abs(x[1]))
                print "Correctness scores:", corrs_list
                all_corrs.append(corrs_list[0][1])
                all_profiles.append(corrs_list[0][0])
                top_profile, top_score = corrs_list[0]
                if top_profile not in ress:
                        ress[top_profile] = {"cluster": cluster, "corr": top_score}
                else:
                        if abs(ress[top_profile]["corr"]) < abs(top_score):
                                ress[top_profile] = {"cluster": cluster, "corr": top_score}
        print "All highers corrs:", all_corrs
        print "Their profiles:", all_profiles
        print len(set(all_profiles)), "profiles represented"
        print "Average:", float(sum(all_corrs))/len(all_corrs)
        print "Results:", ress
        print "Average corr:", sum(abs(x["corr"]) for x in
                ress.values())/float(len(ress))


if __name__ == "__main__":
    main()
