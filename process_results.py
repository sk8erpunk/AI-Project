import os
import pickle
import numpy as np
import pandas as pd
from graphviz import Graph


def create_graph_from_paths(paths, results_dir):
    dot = Graph(name='tree_graph', filename='tree_graph', format='png', directory=results_dir)
    node_id = -1

    # create root
    dot.node(name=str(node_id), label='root')
    node_id += 1

    prev_path = None
    prev_path_node_ids = []
    for path in paths:
        # find lowest common ancestor of curr path and prev path
        lowest_common_ancestor = -1
        if prev_path is not None:
            min_len = min(len(prev_path), len(path))
            for n_p, n_c, idx in zip(prev_path[:min_len], path[:min_len], range(min_len)):
                if n_p[0] != n_c[0] or n_p[1] != n_c[1]:
                    lowest_common_ancestor = idx-1
                    break
        # add path to tree
        prev_node_id = -1
        path_ids = []
        for i, n in enumerate(path):
            if i <= lowest_common_ancestor:
                prev_node_id = prev_path_node_ids[i]
                path_ids.append(prev_node_id)
                continue

            # add node and edges
            dot.node(name=str(node_id), label=str(n))
            path_ids.append(node_id)
            dot.edge(str(prev_node_id), str(node_id))
            prev_node_id = node_id
            node_id += 1

        # update prev path vars
        prev_path_node_ids = path_ids
        prev_path = path

    dot.render(filename='tree_graph',directory=results_dir)


def process_cs_txt(cs_txt):
    str_txt = str(cs_txt).replace("[", "")
    str_txt = str_txt.replace("]", "")
    str_txt = str_txt.split()
    arr = np.array(map(float, str_txt))
    return arr

if __name__ == '__main__':
    print "=================================================="
    print "  Process Results - Experiments Summary"
    print "=================================================="
    data_dir = 'data'
    results = 'results'
    clustering_algorithm_results = 'clustering_algorithm_results'
    exp_summary_dir = 'experiments_summary'
    exps_dir_path = os.path.join(results, clustering_algorithm_results)
    exps_summary_path = os.path.join(results,exp_summary_dir)
    dict_list = []
    dir_num = len(os.listdir(exps_dir_path))

    for directory,i in zip(os.listdir(exps_dir_path),range(dir_num)):
        dir_array = str(directory).split('_')
        k_folds_arr = []
        # get folds values via txt files
        for filename in os.listdir(os.path.join(exps_dir_path,directory)):
            if filename.endswith(".txt"):
                filename_array = str(filename).split('_')
                k_folds_arr.append(int(filename_array[0]))
        k_folds_arr.sort()
        # process cross validation results from the txt files
        for k in k_folds_arr:
            txt_file = open(os.path.join(exps_dir_path, directory, str(k)+'_cs_results.txt'))
            r_txt_file = txt_file.read()
            curr_results = process_cs_txt(r_txt_file)
            dict_list.append({'sample_size': int(dir_array[2]),
                            'min_entropy': float(dir_array[5]),
                            'min_cells': int(dir_array[8]),
                            'n_folds': str(k),
                            'results': (str(curr_results)).replace('\n',""),
                            'avg': np.average(curr_results)})

        print "Drawing tree illustration for {0}".format(directory)
        paths = np.load(os.path.join(exps_dir_path, directory, 'new_features_paths.npy'))
        create_graph_from_paths(paths=paths, results_dir=os.path.join(exps_dir_path, directory))
        print "Saving under: {0}".format('results/clustering_algorithm_results/'+str(directory)+'/tree_graph.png')
        print "___________________________________________________"

    print "Creating expirements summary table:"
    results_df = pd.DataFrame(dict_list, columns=['sample_size', 'min_entropy', 'min_cells', 'n_folds', 'results', 'avg'])
    # make exp summary dir
    if not os.path.exists(exps_summary_path):
        os.makedirs(exps_summary_path)
    print results_df
    print "Saving under: {0}".format('results/experiments_summary')
    # save csv table
    results_df.to_csv('{0}\{1}'.format(exps_summary_path,'results_table.csv'),',')
    print "___________________________________________________"

    print "Finding best experiments:"
    sf = results_df.groupby(['sample_size','min_entropy','min_cells'])['avg'].max() # find max per experiment
    max_avg_df = sf.reset_index()
    print "1.Best accuracy of each experiment:"
    print max_avg_df
    print "Saving under: {0}".format('results/experiments_summary/best_avgs_per_exp.csv')
    max_avg_df.to_csv('{0}\{1}'.format(exps_summary_path,'best_avgs_per_exp.csv'),',')  # save best avgs per exp
    best_avg = max_avg_df['avg'].max()
    best_row = results_df.loc[results_df['avg'] == best_avg].iloc[0]
    print "2.Best experiment:"
    print best_row
    print "Saving under: {0}".format('results/experiments_summary/best_experiment.csv')
    best_row.to_csv('{0}\{1}'.format(exps_summary_path,'best_experiment.csv'),',')  # save best exp
    min_entropy = best_row["min_entropy"]
    sample_size = best_row["sample_size"]
    min_cells = best_row["min_cells"]
    best_exp_fold = 'sample_size_'+str(sample_size)+'_min_ent_'+str(min_entropy)+"_min_cells_"+str(min_cells)
    best_dir_path = os.path.join(exps_dir_path,best_exp_fold)
    # save best exp to pkl
    best_row = best_row.to_dict()
    with open(os.path.join(exps_summary_path,'best_experiment.pkl'), "w") as outfile:
        pickle.dump(best_row, outfile)
