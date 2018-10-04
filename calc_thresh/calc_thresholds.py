from r_scripts import *
from itertools import chain
from helpers.channel_helper_functions import channel_to_index
import numpy as np


# Calculates Thresholds
# Gets all human cells and a list of some features
def calculate_thresholds(humans_cells,features_names):
    # print "Converting features to indexes:"
    indexs_of_features = [channel_to_index(c) for c in features_names]
    # print indexs_of_features
    # print "Running R function..."
    r_func = robjects.globalenv['getmixtoolsRes']
    cutoff = r_func(humans_cells,indexs_of_features)
    thresholds = list(chain.from_iterable(np.asarray(cutoff)))
    # print "Calculate Thresholds:"
    # print thresholds
    # print "Creating Dictionary of Thresholds:"
    d = {}
    for feature,threshold in zip(features_names,thresholds):
        d[feature]=threshold
    # print d
    return d
