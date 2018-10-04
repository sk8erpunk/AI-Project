from abc import ABCMeta, abstractmethod


class abstract_validator:
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_scores(self, classifier, X, y):
        return None


