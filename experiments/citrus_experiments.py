import os
import matplotlib.pyplot as plt
import numpy as np
from main_functions import solve_through_cluster_manifold_classifier_validator
from main_functions import get_citrus_human_table
from validators.cross_validator import cross_validator
from sklearn import cross_validation
import pandas as pd
import csv


# Given Citrus information, run cross validation with different fold sizes 12.8.17
def citrus_experiment(classifier,labels,human_list,folds):
    print "=================================================="
    print "                Citrus Results"
    print "=================================================="
    citrus_dir = os.path.join('results','citrus_results')
    if not os.path.exists(citrus_dir):
        os.makedirs(citrus_dir)
    citrus_data = os.path.join('citrus_data','abundancesLeafs.csv')
    print "Citrus Cell Abundances: {0}".format('citrus_data/abundancesLeafs.csv')
    human_table = get_citrus_human_table(citrus_data)
    # save human freq table
    np.save(os.path.join(citrus_dir,'human_table.npy'), human_table)

    fold_scores_dict = {}
    for k in folds:
        rs = cross_validation.StratifiedKFold(y=labels, n_folds=k, shuffle=False)
        validator = cross_validator(rs)
        cross_res = solve_through_cluster_manifold_classifier_validator(human_list, labels=labels, classifier=classifier, validator=validator, results_dir=None,
                                                                    cluster=None, manifold=None, sampler=None, r_function=None, citrus_data=citrus_dir)
        fold_scores_dict[k] = np.average(cross_res)
    print "Citrus Results:"
    res = pd.DataFrame.from_dict(fold_scores_dict.items())
    res.columns = ['fold','accuracy']
    print res
    # save results
    with open(os.path.join(citrus_dir,'citrus_results.csv'),'wb') as f:
        w = csv.writer(f)
        w.writerow(fold_scores_dict.keys())
        w.writerow(fold_scores_dict.values())
    f.close()
    print "Saving under: {0}".format('results/citrus_results/citrus_results.csv')

    # figure settings
    plt.figure(figsize=(12,9))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False) # Remove the plot frame lines.
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.ylim(0, 1)
    plt.xlim(0, max(folds)+1)
    plt.yticks(np.arange(0, 1, 0.1), [str(int(x*100)) + "%" for x in np.arange(0, 1, 0.1)], fontsize=14)
    plt.xticks(np.arange(0, max(folds)+1),fontsize=14)

    #titles
    plt.xlabel('Folds',fontsize=16)
    plt.ylabel('Accuracy rate',fontsize=16)
    plt.title('Citrus Accuracy vs K-Fold',fontsize=20)

    # plots
    avgs_success = fold_scores_dict.values()
    plt.scatter(folds,avgs_success)
    plt.plot(folds,avgs_success,color='r')
    for a,b in zip(folds, avgs_success):
       plt.text(a, b, str(b*100)+"%", fontsize=14)
    plt.grid(True)

    # Save results
    print "Citrus graph: {0}".format('results/graphs_results/citrus_graph.png')
    plt.savefig('results/graphs_results/citrus_graph.png')

