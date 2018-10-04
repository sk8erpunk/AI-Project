from sklearn import cross_validation
from validators.abstract_validator import abstract_validator


class cross_validator(abstract_validator):
    def __init__(self, cv):
        self.cv = cv

    def get_scores(self, classifier, X, y):
        return cross_validation.cross_val_score(classifier, X, y, cv = self.cv, scoring='accuracy')