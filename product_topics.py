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

#Gensim library
import gensim
from gensim import corpora

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

def make_documents(movies_dict, indicators, feature_names,
        categorical_features):
    res = indicators.map(lambda (mid, inds):
            (mid, [feature_names[fid] for (fid, fval) in enumerate(inds) if fid in
                categorical_features and fval == 1])).collect()
    mids, docs = zip(*res)
    return mids, docs

def topicize_indicators(sc, movies_dict, indicators, feature_names,
        categorical_features, num_topics=15,
        num_words=10, passes=100, docs=None, mids=None):
    if mids is None or docs is None:
       mids, docs = make_documents(movies_dict, indicators,
                                   feature_names,
                                   categorical_features)
    dictionary = corpora.Dictionary(docs)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in docs]
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(doc_term_matrix, num_topics=num_topics, id2word = dictionary,
         passes=passes)
    feature_names = {t[0]:t[1] for t in
            ldamodel.print_topics(num_topics=num_topics,
        num_words=num_words)}
    categorical_features = {}
    topics = []
    for bow in doc_term_matrix:
        cur_topics_list = ldamodel.get_document_topics(bow)
        ctd = collections.defaultdict(float, dict(cur_topics_list))
        cur_topics = [ctd[tid] for tid in xrange(num_topics)]
        topics.append(cur_topics)
    indicators = sc.parallelize(zip(mids, topics))
    return indicators, feature_names, categorical_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist-dir", action="store", type=str, help=\
                        "Path from which to loaad models and features to analyze")
    parser.add_argument("--movies-file", action="store", type=str, help=\
                        "CSV file with movie names")
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
    print "Building documents"
    mids, docs = make_documents(movies_dict, indicators,
                                results['feature_names'],
                                results['categorical_features'])
    dictionary = corpora.Dictionary(docs)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in docs]
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(doc_term_matrix, num_topics=15, id2word = dictionary,
         passes=100)
    for t in ldamodel.print_topics(num_topics=15, num_words=10):
        print t

if __name__ == "__main__":
    main()
