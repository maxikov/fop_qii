#standard library
import argparse
import os.path
import pickle
import random
import collections
import sys

#project files
import rating_explanation
import rating_qii
import tree_qii
import shadow_model_qii
import parsers_and_loaders
import common_utils

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
                        impurity="gini",
                        maxDepth=8)
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
    meta_training_data = common_utils.safe_zip(
            indicators, clusters).map(
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
                        impurity="gini",
                        maxDepth=8)
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

if __name__ == "__main__":
    main()
