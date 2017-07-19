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
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.classification import NaiveBayes, SVMWithSGD

#numpy library
import numpy as np

#sklearn library
import sklearn.cluster
from sklearn.neural_network import MLPClassifier

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

class ClassifierWrapper(object):
    def __init__(self, sc, model, args, n_classes, categorical_features={}):
        self.model = model
        self.args = args
        self.sc = sc
        self.n_classes = n_classes
        self.categorical_features = categorical_features

    def train(self, data):
        if self.model == "decision_tree":
            self._model = pyspark.\
            mllib.\
                tree.\
                DecisionTree.\
                trainClassifier(
                    data,
                    numClasses=self.n_classes,
                    categoricalFeaturesInfo=self.categorical_features,
                    impurity=self.args.impurity,
                    maxDepth=self.args.max_depth)
        elif self.model == "random_forest":
            self._model = RandomForest.trainClassifier(data,
                    numClasses=self.n_classes,
                    categoricalFeaturesInfo=self.categorical_features,
                    impurity=self.args.impurity,
                    maxDepth=self.args.max_depth,
                    numTrees=32)
        elif self.model == "naive_bayes":
            self._model = NaiveBayes.train(data)
        elif self.model == "svm":
            self._model = SVMWithSGD.train(data)
        elif self.model == "mlpc":
            X = np.array(data.map(lambda x: x.features).collect())
            Y = np.array(data.map(lambda x: x.label).collect())
            self.rank = X.shape[1]
            layers = (min(self.rank, 500), min(self.rank, 500))
            self._model = MLPClassifier(hidden_layer_sizes=layers,
                    max_iter=1000, activation="tanh", solver="lbfgs")
            self._model.fit(X, Y)
        return self

    def get_used_features(self):
        if self.model == "decision_tree":
            return tree_qii.get_used_features(self._model)
        elif self.model in ["mlpc", "random_forest", "naive_bayes", "svm"]:
            return range(self.rank)

    def predict(self, data):
        if self.model in ["decision_tree", "random_forest", "naive_bayes",
        "svm"]:
            return self._model.predict(data)
        elif self.model == "mlpc":
            X = np.array(data.collect())
            Y = self._model.predict(X)
            res = self.sc.parallelize(list(Y))
            return res

    def toDebugString(self, feature_names=None):
        if self.model == "decision_tree":
            if feature_names is not None:
                return common_utils.substitute_feature_names(
                        self._model.toDebugString(),
                        feature_names)
            else:
                return self._model.toDebugString()
        elif self.model in ["mlpc", "random_forest", "naive_bayes", "svm"]:
            return str(self._model)

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


#https://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/
def clusterdist(fal, centroid, cluster_id, normalize=True):
    filter_f = functools.partial(lambda clid, ((mid, ftrs), cls):
            cls == clid, cluster_id)
    fal = fal.filter(filter_f)
    nk = fal.count()
    map_f = functools.partial(lambda centr, ((mid, ftrs), cls):
            dist(centr, ftrs), centroid)
    dists = fal.map(map_f)
    Dk = dists.reduce(lambda a, b: a+b)
    if not normalize:
        Dk *= 2.0 * nk
    return Dk

def compactness(fal, centroids, n_clusters, log=True):
    Wk = 0.0
    for cluster_id in xrange(n_clusters):
        centroid = centroids[cluster_id]
        Dk = clusterdist(fal, centroid, cluster_id, normalize=True)
        Wk += Dk
    if log:
        Wk = np.log(Wk)
    return Wk

def bounding_box(fal):
    reduce_f_min = lambda ((mid1, ftrs1), cls1), ((mid2, ftrs2), cls2):\
        ((mid1, map(min, zip(ftrs1, ftrs2)) ), cls1)
    reduce_f_max = lambda ((mid1, ftrs1), cls1), ((mid2, ftrs2), cls2):\
        ((mid1, map(max, zip(ftrs1, ftrs2)) ), cls1)
    mins = fal.reduce(reduce_f_min)[0][1]
    maxs = fal.reduce(reduce_f_max)[0][1]
    return mins, maxs

def generate_data_set(sc, mins, maxs, N):
    rank = len(mins)
    res = None
    for i in xrange(rank):
        col = np.random.uniform(mins[i], maxs[i], N)
        col = col.reshape((N, 1))
        if res is None:
            res = col
        else:
            res = np.hstack((res, col))
    res_rdd = sc.parallelize([x for x in res])
    return res_rdd

def BWkb(sc, fal, B, n_clusters, cluster_model, cluster_args, cluster_kwargs):
    N = fal.count()
    print "Got", N, "data points, computing bounding box"
    mins, maxs = bounding_box(fal)
    print "Box:", mins, maxs
    Wkbs = []
    for b in xrange(B):
        print "Generating data set", b, "out of", B
        data_set = generate_data_set(sc, mins, maxs, N)
        print "Done"
        print "Running clustering, kwargs:", cluster_kwargs
        kmeans_model = cluster_model(*cluster_args, **cluster_kwargs)
        centroids = list(map(np.array, kmeans_model.clusterCenters))
        print "Done"
        print "Labeling clusters"
        clusters = kmeans_model.predict(data_set).map(float)
        product_features = data_set.zipWithIndex().map(lambda (a, b): (b, a))
        features_and_labels = common_utils.safe_zip(product_features, clusters)
        print "Computing compactness"
        wk = compactness(features_and_labels, centroids, n_clusters)
        print "Compactness:", wk
        Wkbs.append(wk)
    avg_comp = np.mean(Wkbs)
    print "Average compactness:", avg_comp
    comp_std = np.std(Wkbs)
    print "Standard deviation:", comp_std
    sk = np.sqrt(1+1.0/B) * comp_std
    print "sk:", sk
    return avg_comp, sk

def gap(sc, B, data_set, product_features, n_clusters, cluster_model,
        cluster_args, cluster_kwargs):
    print "Running clustering for", n_clusters, "clusters, kwargs:", cluster_kwargs
    kmeans_model = cluster_model(*cluster_args, **cluster_kwargs)
    centroids = list(map(np.array, kmeans_model.clusterCenters))
    print "Done"
    print "Labeling clusters"
    clusters = kmeans_model.predict(data_set).map(float)
    features_and_labels = common_utils.safe_zip(product_features, clusters)
    print "Computing compactness"
    wk = compactness(features_and_labels, centroids, n_clusters)
    print "Computing estimated compactness"
    avg_comp, sk = BWkb(sc, features_and_labels, B, n_clusters, cluster_model, cluster_args,
            cluster_kwargs)
    Gap = avg_comp - wk
    return Gap, sk

def choose_optimal_k(sc, B, data_set, product_features, max_clusters, cluster_model,
        cluster_args, cluster_kwargs):
    res = []
    for n_clusters in xrange(1, max_clusters+1):
        print "Trying", n_clusters
        cluster_args_ = list(cluster_args) + [n_clusters]
        Gap, sk = gap(sc, B, data_set, product_features, n_clusters, cluster_model,
                      cluster_args_, cluster_kwargs)
        res.append((Gap, sk))
        if n_clusters in [1,2]:
            continue
        else:
            Gap_k, sk_k = res[-2]
            kp1 = Gap - sk
            stat = Gap_k - kp1
            print "Gap_{}: {}, Gap_{} - s_{}: {}".format(n_clusters - 1, Gap_k,
                    n_clusters, n_clusters, kp1)
            print "Gap_{} - (Gap_{} - s_{}): {}".format(n_clusters - 1,
                    n_clusters, n_clusters, stat)
            if Gap_k >= kp1:
                print "Found optimal k:", n_clusters - 1
                return n_clusters - 1
    print "No change in gap statistic found, assuming k = 1"
    return 1


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
    parser.add_argument("--max-clusters", action="store", type=int,
            help="Override --n-clusters and use gap statistic to" +\
                    " determine the optimal k, no more than this.")
    parser.add_argument("--cluster-sample-size", action="store", type=int,
            default=10, help="Number of movies to display in each cluster")
    parser.add_argument("--test-ratio", action="store", type=float,
            default=0.3, help="Percent of the data set to use as a test set")
    parser.add_argument("--model", action="store", type=str,
            default="decision_tree", choices=["decision_tree", "mlpc",
            "random_forest", "naive_bayes", "svm"])
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
        def cluster_model(*args, **kwargs):
            return KMeans.train(*args, **kwargs)
        cluster_args = (data_set,)
        cluster_kwargs = {"maxIterations": 100, "initializationMode": "random"}
        if args.max_clusters is not None:
            B = 10
            k = choose_optimal_k(sc, B, data_set, product_features, args.max_clusters, cluster_model,
                                 cluster_args, cluster_kwargs)
            args.n_clusters = k
        cluster_args = list(cluster_args) + [args.n_clusters]
        kmeans_model = cluster_model(*cluster_args, **cluster_kwargs)
        centroids = list(map(np.array, kmeans_model.clusterCenters))
    elif args.cluster_model == "gmm":
        kmeans_model = GaussianMixture.train(data_set, args.n_clusters)
        centroids = [np.array(g.mu) for g in
                kmeans_model.gaussians]
    elif args.cluster_model == "spectral":
        kmeans_model = FitPredictWrapper(sc,
                sklearn.cluster.SpectralClustering, n_clusters=args.n_clusters,
                assign_label="discretize")
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
        key_f_least = functools.partial(lambda centroid, ((mid, ftrs), cls):
            -dist(ftrs, centroid), centroid)
        sample_least = cur_mvs.takeOrdered(args.cluster_sample_size,
                key=key_f_least)
        sample = sample + sample_least
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
    latent_tree = ClassifierWrapper(sc, args.model, args, args.n_clusters,
                 categorical_features={}).train(latent_training_data)

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
    meta_tree = ClassifierWrapper(sc, args.model, args, args.n_clusters,
                 categorical_features=results["categorical_features"]).train(meta_training_data)
    print meta_tree.toDebugString(results["feature_names"])
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
        used_features = meta_tree.get_used_features()
        print len(used_features), "features used"
        for (cluster, cls_var, text_sample) in cluster_data:
            print "Cluster {} (variance: {}):".format(cluster, cls_var)
            features = meta_data_set.keys()
            qiis = tree_qii.get_tree_qii(meta_tree, features, used_features,
                    classifier=True, class_of_interest=cluster)
            qiis_list = sorted(qiis.items(), key=lambda (f, q): -abs(q))
            qiis_list_names = [(results["feature_names"][f], q) for (f, q) in
                    qiis_list if q != 0]
            print "QIIs:", qiis_list_names
            for s in text_sample:
                print s
    else:
        used_features = meta_tree.get_used_features()
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
            features = meta_data_set.keys()
            qiis = tree_qii.get_tree_qii(meta_tree, features, used_features,
                    classifier=True, class_of_interest=cluster)
            qiis_list = sorted(qiis.items(), key=lambda (f, q): -abs(q))
            qiis_list_names = [(results["feature_names"][f], q) for (f, q) in
                    qiis_list if q != 0]
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
            meta_tree = ClassifierWrapper(sc, args.model, args, 2,
                 categorical_features=results["categorical_features"]).train(meta_training_data)
            print meta_tree.toDebugString(results["feature_names"])
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
            used_features = meta_tree.get_used_features()
            print len(used_features), "features used"
            if len(used_features) > 0:
                features = meta_data_set.keys()
                qiis = tree_qii.get_tree_qii(meta_tree, features,
                        used_features, classifier=True)
                qiis_list = sorted(qiis.items(), key=lambda (f, q): -abs(q))
                qiis_list_names = [(results["feature_names"][f], q) for (f, q) in
                    qiis_list if q!= 0]
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
