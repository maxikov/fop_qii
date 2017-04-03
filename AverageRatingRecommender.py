#standard library
import time
from collections import defaultdict

class AverageRatingRecommender:
    def __init__(self, logger=None):
        self.logger = logger

    def train(self, training):
        self.logger.debug("Training the average rating model")
        start = time.time()
        self.ratings = training\
                .groupBy(lambda x: x[1])\
                .map(lambda (mid, data):
                     (mid, sum(x[2] for x in data)/float(len(data))))
        self.ratings = dict(self.ratings.collect())
        self.ratings = defaultdict(lambda: 0.0, self.ratings)
        self.logger.debug("Done in %f seconds", time.time() - start)

    def predict(self, user_movies):
        if self.logger is None:
            print "Making average rating recommendations"
        else:
            self.logger.debug("Making average rating predictions")
        start = time.time()
        res = user_movies\
                .map(lambda x: (x[0], x[1]))\
                .map(lambda (user, product):
                     (user,
                      product,
                      self.ratings[product]))
        self.logger.debug("Done in %f seconds", time.time() - start)
        return res
