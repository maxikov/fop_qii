# The API we would like for QII computations.

class Accuracy(object):
    def __init__(self):
        pass

class Full(Accuracy):
    def __init__(self):
        pass

class Samples(Accuracy):
    __fields__ = ['num_samples']

    def __init__(self, num_samples):
        self.num_samples = num_samples

class Confidence(Accuracy):
    __fields__ = ['confidence']

    def __init__(self, confidence):
        self.confidence = confidence

    def samples(self, dataset_size):
        return Samples(42) # how many samples we need to achieve
                           # self.confidence given dataset size
                           # dataset_size

class QII(object):
    fields = ['dataset', 'superfeatures', 'accuracy']
    def __init__(self, dataset, superfeatures, accuracy):
        # type: RDD[T] Map[Int,Int] Accuracy -> QII
        self.dataset = dataset
        self.superfeatures = superfeatures
        self.accuracy = accuracy

    def average_unary_individual(self, opts):
        """ something """

    def unary_individual(self, opts, index=0):
        """ something """

    # later:
    def discrim(self, opts):
        """ something """

    def banzhaf(self, opts):
        """ something """

    def shapley(self, opts):
        """ something """
