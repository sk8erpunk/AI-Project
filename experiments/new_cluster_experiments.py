import os
import pickle

import numpy as np
from entities.human_being import get_all_cells_features
from sklearn import cross_validation
from entities.Tree import Tree
from main_functions import sample_once_per_size
from main_functions import solve_through_cluster_manifold_classifier_validator
from validators.cross_validator import cross_validator


# r_function_experiment
def new_cluster_experiments(classifier, labels, human_list, sample_size, min_entropy_arr, min_cell_arr, folds_size_arr):
    print "=================================================="
    print "        Running The Clustering Algorithm"
    print "=================================================="

    # create r results directory
    r_func_results = 'clustering_algorithm_results'
    results_directory = 'results'
    result_path = os.path.join(results_directory,r_func_results)
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    print "Sampling {0} cells from each patient...".format(sample_size)
    sampler_data = sample_once_per_size(human_list, sample_size)

    exp_index = 1  # Experiment number

    for minimal_entropy in min_entropy_arr:
        for minimal_cells in min_cell_arr:
                print "___________________________________________________"
                print "Running Experiment {0}:\n" \
                      "Sample Size: {1}\n" \
                      "Min Entropy: {2}\n" \
                      "Min Cells: {3}".format(exp_index, sample_size, minimal_entropy, minimal_cells)

                exp_index += 1
                r_function = dict()
                r_function['minimal_entropy'] = minimal_entropy  # VAR
                r_function['minimal_cells'] = minimal_cells  # VAR

                # prepare parameters
                humans_cells = get_all_cells_features(sampler_data)
                tree = Tree(humans_cells, r_function['minimal_entropy'], r_function['minimal_cells'])
                leaves = tree.get_leaves()
                # create freq_table
                human_table = [[x.getFreq(h_idx, h_cells_num) for x in leaves] for h_idx, h_cells_num in
                              zip(xrange(0, len(sampler_data)), [h.get_cells_number() for h in sampler_data])]
                human_table = np.array(tuple(human_table))

                # results directory
                fold_name = 'sample_size_' + str(sample_size) + '_min_ent_' + str(minimal_entropy) +\
                            '_min_cells_' + str(minimal_cells)
                exp_results = os.path.join(results_directory, r_func_results, fold_name)
                if not os.path.exists(exp_results):
                    os.makedirs(exp_results)

                # save exp files
                save_files(exp_results,human_table,leaves)

                print "Classifier Results:"
                # run classifier
                run_classifier(folds_size_arr,labels,human_list,sampler_data,classifier,r_function,exp_results)


def run_classifier(folds_size_arr,labels,human_list,sampler_data,classifier,r_function,exp_results):
    # Run Classifier
    print "Fold Size | Classifier Accuracy"
    for folds_number in folds_size_arr:
        rs = cross_validation.StratifiedKFold(y=labels, n_folds=folds_number,shuffle=False)
        validator = cross_validator(rs)

        # solve
        cross_res = solve_through_cluster_manifold_classifier_validator(human_list, labels, classifier,
                                                     validator, exp_results, cluster=None,
                                                    manifold=None, sampler=sampler_data,
                                                    r_function=r_function, citrus_data=None)

        print "{0}        | {1} ".format(folds_number, np.average(cross_res))
        # save cv_results
        cr_file_txt = str(folds_number)+'_cs_results.txt'
        with open(os.path.join(exp_results, cr_file_txt), 'w') as outfile:
            outfile.write(str(cross_res))

    print "Saving under: {0}".format(exp_results)


def save_files(results_dir,human_table,leaves):
    # save human freq table
    np.save(os.path.join(results_dir,'human_table.npy'), human_table)

    # save threshes of leaves
    thresh_dictionaries = [x.thresh for x in leaves]
    with open(os.path.join(results_dir,'new_threshes.pkl'), "w") as outfile:
        pickle.dump(thresh_dictionaries, outfile)

    # save entropies paths
    new_entropies_paths = np.array([x.entropies for x in leaves])
    np.save(os.path.join(results_dir,'new_entropies_path.npy'), new_entropies_paths )

    # save features paths
    new_features_paths = np.array([x.path for x in leaves])
    np.save(os.path.join(results_dir,'new_features_paths.npy'), new_features_paths)

    # create leave_dic [leaf_idx: cell_indices)]
    leaves_cell_idx = dict()
    for l,i in zip(leaves,range(0,len(leaves))):
        leaf_cell_indices = l.cells[:, [46,47]] # array of arrays - inner array has two values (cell idx,human idx)
        leaves_cell_idx[i] = leaf_cell_indices

    # save leaves dictionary
    with open(os.path.join(results_dir,'leaves_cell_idx.pkl'), "w") as outfile:
        pickle.dump(leaves_cell_idx, outfile)

    # create leaf idx with path dictionary [leaf_idx: (path,cells num)]
    leaves_cnum_path = dict()
    for l,p in zip(range(0,len(leaves)),new_features_paths):
        leaves_cnum_path[l] = (len(leaves_cell_idx[l]),p)

    # save leaves_idx_paths
    with open(os.path.join(results_dir,'leaves_cnum_path.pkl'), "w") as outfile:
        pickle.dump(leaves_cnum_path, outfile)
