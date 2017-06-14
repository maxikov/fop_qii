""" standard library, TODO: documentation"""

#standard library
import time
import pickle

#numpy library
import numpy

#project files
import common_utils
import functools

def load(fname, sc, num_partitions):
    ifile = open(fname, "rb")
    res = pickle.load(ifile)
    ifile.close()
    res.u_feats = sc.parallelize(res.u_feats)\
            .repartition(num_partitions)\
            .cache()
    res.p_feats = sc.parallelize(res.p_feats)\
            .repartition(num_partitions)\
            .cache()
    return res

def save(self, fname):
    uf, pf = self.u_feats, self.p_feats
    self.u_feats = self.u_feats.collect()
    self.p_feats = self.p_feats.collect()
    ofile = open(fname, "wb")
    pickle.dump(self, ofile)
    ofile.close()
    self.u_feats, self.p_feats = uf, pf
    return self

class CustomFeaturesRecommender(object):
    """ TODO documentation """
    def __init__(self, rank, userFeatures, productFeatures):
        self.rank = rank
        self.u_feats = userFeatures.sortByKey()
        self.p_feats = productFeatures.sortByKey()


    def userFeatures(self):
        return self.u_feats

    def productFeatures(self):
        return self.p_feats

    def predictAll(self, user_movies):
        start = time.time()
        res = common_utils.manual_predict_all(user_movies, self.u_feats,
                self.p_feats)
        return res

    def predict(self, user, product):
        _, uf = self.u_feats.lookup(user)[0]
        _, pf = self.p_feats.lookup(product)[0]
        res = sum(x*y for (x, y) in zip(uf, pf))
        return res
