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

#pyspark libraryb
from pyspark import SparkConf, SparkContext
import pyspark
import pyspark.mllib.tree
from pyspark.mllib.classification import LabeledPoint
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.clustering import KMeans

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist-dir", action="store", type=str, help=\
                        "Path from which to loaad models and features to analyze")
    parser.add_argument("--n-clusters", action="store", type=int, default=10,
            help="Number of clusters to create")
    parser.add_argument("--max-depth", action="store", type=int, default=8,
                        help="Maximum depth of the decision tree")
    parser.add_argument("--impurity", action="store", type=str, default="gini",
                        help="Impurity function for tree")
    parser.add_argument("--user-profiles", action="store", type=str,
                        help="Path to the file with user profiles of a "+\
                        "synthetic data set (if applicable).")
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

    data_set = product_features.values()

    print "Training K-means"
    kmeans_model = KMeans.train(data_set, args.n_clusters, maxIterations=100,
            initializationMode="random")

    print "Done"
    print "Labeling clusters"
    clusters = kmeans_model.predict(data_set).map(float)
    features_and_labels = common_utils.safe_zip(product_features, clusters)

    latent_training_data = features_and_labels.map(\
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
    tree_predictions  = latent_tree.predict(data_set).map(float)
    predobs = common_utils.safe_zip(tree_predictions, clusters)
    acc = predobs.filter(lambda (x, y): x==y).count()\
            /float(predobs.count())
    print "Accuracy:", acc
    metrics = MulticlassMetrics(predobs)

    # Overall statistics
    precision = metrics.precision()
    recall = metrics.recall()
    f1Score = metrics.fMeasure()
    print("Summary Stats")
    print("Precision = %s" % precision)
    print("Recall = %s" % recall)
    print("F1 Score = %s" % f1Score)

    print "Building meta training set"
    meta_data_set = common_utils.safe_zip(
            indicators, clusters)
    meta_training_data = meta_data_set.map(
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
    tree_predictions  = meta_tree.predict(indicators.values()).map(float)
    predobs = common_utils.safe_zip(tree_predictions, clusters)
    acc = predobs.filter(lambda (x, y): x==y).count()\
            /float(predobs.count())
    print "Accuracy:", acc
    metrics = MulticlassMetrics(predobs)

    # Overall statistics
    precision = metrics.precision()
    recall = metrics.recall()
    f1Score = metrics.fMeasure()
    print("Summary Stats")
    print("Precision = %s" % precision)
    print("Recall = %s" % recall)
    print("F1 Score = %s" % f1Score)

    if args.user_profiles is not None:
        used_features = tree_qii.get_used_features(meta_tree)
        print len(used_features), "features used"
        profiles, user_profiles = pickle.load(open(args.user_profiles,
                                                   "rb"))
        all_clusters = sorted(list(set(clusters.collect())))
        print len(all_clusters), "clusters found"
        all_corrs = []
        all_profiles = []
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
        print "All highers corrs:", all_corrs
        print "Their profiles:", all_profiles
        print len(set(all_profiles)), "profiles represented"
        print "Average:", float(sum(all_corrs))/len(all_corrs)

        print "Making per-cluster trees"
        all_corrs = []
        all_profiles = []
        old_clusters = clusters
        for cluster in all_clusters:
            print "Processing cluster", cluster
            clusters = old_clusters.map(functools.partial(lambda cluster, x:
                    1.0 if x == float(cluster) else 0.0, cluster))
            print "New clusters:", clusters.countByValue()
            print "Building meta training set"
            meta_data_set = common_utils.safe_zip(
                indicators, clusters)
            meta_training_data = meta_data_set.map(
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
            tree_predictions  = meta_tree.predict(indicators.values()).map(float)
            print "Predictions:", tree_predictions.countByValue()
            predobs = common_utils.safe_zip(tree_predictions, clusters)
            acc = predobs.filter(lambda (x, y): x==y).count()\
                /float(predobs.count())
            print "Accuracy:", acc
            evaluations = common_utils.evaluate_binary_classifier(
                tree_predictions.zipWithIndex().map(lambda (x,y):(y,x)),
                clusters.zipWithIndex().map(lambda (x,y):(y,x)),
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
        print "All highers corrs:", all_corrs
        print "Their profiles:", all_profiles
        print len(set(all_profiles)), "profiles represented"
        print "Average:", float(sum(all_corrs))/len(all_corrs)


if __name__ == "__main__":
    main()
