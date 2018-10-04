from sklearn.tree import DecisionTreeClassifier, export_graphviz
from classifiers.abstract_classifier import abstact_classifier


class cart_classifier(abstact_classifier):

    def __init__(self, criterion = 'entropy', splitter = 'best', max_features = None, min_samples_split = 2):
        self.classification_trained_tree = None
        self.criterion = criterion
        self.splitter = splitter
        self.max_features = max_features
        self.min_samples_split = min_samples_split

    def fit(self, train_data_set, train_classifications_set):
        classification_tree = DecisionTreeClassifier(criterion=self.criterion,
                                                    splitter=self.splitter,
                                                    max_features=self.max_features,
                                                    min_samples_split=self.min_samples_split,
                                                    random_state=13)
        self.classification_trained_tree = classification_tree.fit(train_data_set,train_classifications_set)

    def predict(self, test_set):
        y_test_tree_classification = self.classification_trained_tree.predict(test_set)  # classify test_set
        return y_test_tree_classification

    def draw_decision_tree(self, out_file):
        export_graphviz(self, out_file=out_file)
