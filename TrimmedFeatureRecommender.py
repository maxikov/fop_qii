""" standard library, TODO: documentation"""

#standard library
import time
import pickle

#numpy library
import numpy

#project files
import common_utils

def load(fname, sc, num_partitions):
    ifile = open(fname, "rb")
    res = pickle.load(file)
    ifile.close()
    res.u_feats = sc.parallelize(res.u_feats)\
            .repratition(num_partitions)\
            .cache()
    res.p_feats = sc.parallelize(res.p_feats)\
            .repratition(num_partitions)\
            .cache()
    return res

class TrimmedFeatureRecommender(object):
    """ TODO documentation """
    def __init__(self, rank, userFeatures, productFeatures, percentile, logger=None):
        self.logger = logger
        self.rank = rank
        self.u_feats = userFeatures
        self.p_feats = productFeatures
        self.percentile = percentile
        self.thresholds = {}

    def trim_feature(self, features, f):
        f_dist = common_utils.get_feature_distribution(features, f)
        f_dist = numpy.array(f_dist)
        top_threshold = numpy.percentile(f_dist, self.top_percentile)
        bottom_threshold = numpy.percentile(f_dist, self.bottom_percentile)
        self.thresholds[f] = (bottom_threshold, top_threshold)
        self.logger.debug("{}% of data are between {} and {}, thresholding the rest"\
                .format(self.percentile, bottom_threshold, top_threshold))
        features = features\
                .map(lambda (_id, cur_feats):\
                    (_id, common_utils.set_list_value(cur_feats, f, top_threshold if cur_feats[f] > top_threshold else
                        cur_feats[f]) ))\
                .map(lambda (_id, cur_feats):\
                    (_id, common_utils.set_list_value(cur_feats, f, bottom_threshold if cur_feats[f] < bottom_threshold else
                        cur_feats[f]) ))\
                .map(lambda (_id, cur_feats): (_id, map(float, cur_feats)))
        return features

    def train(self):
        self.logger.debug("Trimming feature distributions to leave "+\
                "{}% of data".format(self.percentile))
        start = time.time()
        self.bottom_percentile = (100-self.percentile)/2.0
        self.top_percentile = (100+self.percentile)/2.0
        for f in xrange(self.rank):
            self.logger.debug("Processing feature %d", f)
            self.u_feats = self.trim_feature(self.u_feats, f)
            self.p_feats = self.trim_feature(self.p_feats, f)
        self.logger.debug("Done in %f seconds", time.time() - start)
        return self

    def feature_threshold(self, feature):
        return self.thresholds[feature]

    def userFeatures(self):
        return self.u_feats

    def productFeatures(self):
        return self.p_feats

    def predictAll(self, user_movies):
        self.logger.debug("Making trimmed features predictions")
        start = time.time()
        res = common_utils.manual_predict_all(user_movies, self.u_feats,
                self.p_feats)
        self.logger.debug("Done in %f seconds", time.time() - start)
        return res

    def save(self, fname):
        uf, pf = self.u_feats, self.p_feats
        self.u_feats = self.u_feats.collect()
        self.p_feats = self.p_feats.collect()
        ofile = open(fname, "wb")
        pickle.dump(self, ofile)
        ofile.close()
        self.u_feats, self.p_feats = uf, pf
