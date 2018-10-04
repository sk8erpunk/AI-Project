from sklearn.cross_validation import train_test_split
from validators.abstract_validator import abstract_validator
from sklearn.metrics import accuracy_score


class rand_split_validator(abstract_validator):
    def __init__(self, test_size):
        self.test_size = test_size

    def get_scores(self, classifier, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size)
        classifier.fit(X_train, y_train)
        classification_test = classifier.predict(X_test)
        return [accuracy_score(classification_test, y_test)]


