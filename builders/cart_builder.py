from classifiers.cart_classifier import cart_classifier


class cart_builder():
    def __init__(self):
        self.criterion = 'entropy'
        self.splitter = 'best'
        self.max_features = None
        self.min_samples_split = 2

    def set_criterion(self, criterion):
        if criterion:
            self.criterion = criterion
        return self

    def set_splitter(self, splitter):
        if splitter:
            self.splitter = splitter
        return self

    def set_max_features(self, max_f):
        if max_f:
            self.max_features = max_f
        return self

    def set_min_sample_split(self, min_sa_split):
        if min_sa_split:
            self.min_samples_split = min_sa_split
        return self

    def build(self):
        return cart_classifier(self.criterion, self.splitter, self.max_features, self.min_samples_split)

