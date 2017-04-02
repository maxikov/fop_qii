class RandomizedRecommender:
    def __init__(self, baseline_model, rank, logger=None):
        self.logger = logger
        self.baseline_model = baseline_model
        self.rank = rank

    def randomize(self):
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
            self.perturbed_features = perturb_feature(self.perturbed_features, f)
        if self.logger is None:
            print "Done in", time.time() - start, "seconds"
        else:
            self.logger.debug("Done in {} seconds".format(time.time() - start))

    def predict(self, user_movies):
        if self.logger is None:
            print "Making predictions of the randomized model"
        else:
            self.logger.debug("Making predictions of the randomized model")
        start = time.time()
        perturbed_predictions = manual_predict_all(user_movies,
                self.baseline_model.userFeatures(), self.perturbed_features)
        if self.logger is None:
            print "Done in", time.time() - start, "seconds"
        else:
            self.logger.debug("Done in {} seconds".format(time.time() - start))
        return perturbed_predictions
