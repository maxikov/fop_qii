""" TODO: documentation """

#standard library
import time
import functools
import random

#project files
from common_utils import perturb_feature, manual_predict_all, get_feature_distribution
import common_utils

class RandomizedRecommender(object):
    """ TODO: documentation """

    def __init__(self, sc, baseline_model, rank, perturbed_subset=None, logger=None):
        self.logger = logger
        self.baseline_model = baseline_model
        self.rank = rank
        self.perturbed_subset = perturbed_subset
        self.sc = sc

    def randomize(self):
        """ TODO: documentation """

        if self.logger is None:
            print "Creating randomized model"
        else:
            self.logger.debug("Creating randomized model")
        start = time.time()
        self.features = self.baseline_model.productFeatures()
        perturbed_subset = self.perturbed_subset
        if perturbed_subset is not None:
            filter_f = functools.partial(lambda perturbed_subset, x:
                    x[0] not in perturbed_subset, perturbed_subset)
            self.features_intact = self.features.filter(filter_f)
            filter_f = functools.partial(lambda perturbed_subset, x:
                    x[0] in perturbed_subset, perturbed_subset)
            self.features_perturbed = self.features.filter(filter_f)
            self.features = self.features_perturbed
        keys = self.features.keys().collect()
        data = None
        for f in xrange(self.rank):
            if self.logger is None:
                print "Perturbing feature", f
            else:
                self.logger.debug("Perturbing feature {}".format(f))
            dist = get_feature_distribution(self.features, f)
            random.shuffle(dist)
            ddist = {k: [v] for (k, v) in zip(keys, dist)}
            if data is None:
                data = ddist
            else:
                data = common_utils.dict_join(data, ddist, join_f=lambda x, y:
                        x+y)
        self.perturbed_features = self.sc.parallelize(data.items()).cache()
        if perturbed_subset is not None:
            self.perturbed_features =\
                self.perturbed_features.union(self.features_intact)
        if self.logger is None:
            print "Done in", time.time() - start, "seconds"
        else:
            self.logger.debug("Done in {} seconds".format(time.time() - start))

    def predictAll(self, user_movies):
        """ TODO: documentation """

        if self.logger is None:
            print "Making predictions of the randomized model"
        else:
            self.logger.debug("Making predictions of the randomized model")
        start = time.time()
        perturbed_predictions = manual_predict_all(user_movies,
                                                   self.baseline_model.userFeatures(),
                                                   self.perturbed_features)
        if self.logger is None:
            print "Done in", time.time() - start, "seconds"
        else:
            self.logger.debug("Done in {} seconds".format(time.time() - start))
        return perturbed_predictions
