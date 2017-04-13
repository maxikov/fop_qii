""" TODO: documentation """

#standard library
import time

#project files
from common_utils import perturb_feature, manual_predict_all

class RandomizedRecommender(object):
    """ TODO: documentation """

    def __init__(self, baseline_model, rank, perturbed_subset=None, logger=None):
        self.logger = logger
        self.baseline_model = baseline_model
        self.rank = rank
        self.perturbed_subset = perturbed_subset

    def randomize(self):
        """ TODO: documentation """

        if self.logger is None:
            print "Creating randomized model"
        else:
            self.logger.debug("Creating randomized model")
        start = time.time()
        self.perturbed_features = self.baseline_model.productFeatures()
        for f in xrange(self.rank):
            if self.logger is None:
                print "Perturbing feature", f
            else:
                self.logger.debug("Perturbing feature {}".format(f))
            self.perturbed_features = perturb_feature(self.perturbed_features,
                    f, self.perturbed_subset)
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
