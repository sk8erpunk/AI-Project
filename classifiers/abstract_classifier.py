from abc import ABCMeta, abstractmethod

import sklearn


class abstact_classifier(sklearn.base.BaseEstimator):

    __metaclass__ = ABCMeta

    @abstractmethod
    def predict(self, test_set):
        pass

    @abstractmethod
    def fit(self, train_data_set, train_classifications_set):
        pass
