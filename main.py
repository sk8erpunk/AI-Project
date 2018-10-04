from builders.cart_builder import cart_builder
from helpers.channel_helper_functions import get_channels_names_from_file, create_index_channel_dict
from experiments.citrus_experiments import citrus_experiment
from main_functions import get_tags_for
from main_functions import get_human_list
import os
import sys
import argparse


def define_parser():
    parser = argparse.ArgumentParser(prog="EXPERIMENTS", description='Process experiments parameters')
    parser.add_argument('experiment', help='experiment name')
    parser.add_argument('--sample_size', type=int, default=50000, help='an integer for sampling parameter')
    parser.add_argument('--min_cells', type=int, nargs='+',default=[20000, 10000], help='a list for minimal cells values')
    parser.add_argument('--min_entropy', type=float, nargs='+',default=[0.05, 0.1, 0.3, 0.5], help='a list for minimal entropy values')
    parser.add_argument('--k_fold', type=int, nargs='+',default=[2,4,5,8,10,20], help='a list for k fold values')
    args = parser.parse_args(sys.argv[1:])
    return args.experiment, args.sample_size, args.min_cells, args.min_entropy, args.k_fold


if __name__ == "__main__":

    # data directories
    data_dir = 'data'
    results_dir = 'results'
    graph_dir = 'graphs_results'

    # human_list is a list of human beings[cells (with new additional two columns),fcs_file,index]
    human_list, files_array = get_human_list(data_dir)
    labels = get_tags_for(files_array)


    # results directory
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # graphs results directory
    graph_dir_path = os.path.join(results_dir,graph_dir)
    if not os.path.exists(os.path.join(results_dir,graph_dir)):
        os.mkdir(graph_dir_path)

    # *******************************************************************************************************
    # ONE TIME RUN CODE, TO CREATE DICTIONARY OF CHANNELS
    # print "Create the index_channel_dictionary file"
    # channels = get_channels_names_from_file(data_dir)
    # create_index_channel_dict(channels)
    # *******************************************************************************************************

    # Create Classifier
    classifier = cart_builder().build()

    # Define parser and run experiments
    experiment, sample_size, min_cells_arr, min_entropy_arr, k_folds_arr = define_parser()
    if experiment == "cluster":
        # Clustering Algorithm Experiment
        from experiments.new_cluster_experiments import new_cluster_experiments
        new_cluster_experiments(classifier, labels, human_list, sample_size, min_entropy_arr, min_cells_arr, k_folds_arr)
    elif experiment == "citrus":
        # Citrus Experiment
        citrus_experiment(classifier, labels, human_list,k_folds_arr)
    else:
        print "Experiment does not exist! \n" \
              "Try: main.py cluster --sample_size [sample_size]" \
              " --min_cells min_cells [min_cells]" \
              " --min_entropy min_entropy [min_entropy]" \
              " --k_fold k_fold [k_fold]\n" \
              "Or try: main.py citrus --k_fold k_fold [k_fold]"
        exit(-1)