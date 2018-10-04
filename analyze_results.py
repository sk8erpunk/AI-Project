import pandas as pd
import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm
import csv
from entities.human_being import get_all_cells_features
from helpers.channel_helper_functions import get_channels_names_from_pickle
from main_functions import attach_leaf_index_to_cells
from main_functions import get_human_list


def find_cells(exp,exp_path):
    data_dir = 'data'
    if os.path.exists(os.path.join(exp_path, 'cells_with_leaves_table.csv')):
        return

    print "Tag sample cells with clusters indices (for experiment: {0})...".format(exp)
    print "This process may take a few minutes... please wait"
    all_humans, _ = get_human_list(data_dir)
    all_cells = get_all_cells_features(all_humans)          # combine all cells
    channel_dic_names = get_channels_names_from_pickle()    # load feature names
    # load files
    leaves_cell_idx_pkl = open(os.path.join(exp_path,'leaves_cell_idx.pkl'))
    leaves_cell_idx = pickle.load(leaves_cell_idx_pkl)
    #leaves_cnum_path_pkl= open(os.path.join(best_dir_path,'leaves_cnum_path.pkl'))
    #leaves_cnum_path = pickle.load(leaves_cnum_path_pkl)
    #entropies = np.load(os.path.join(best_dir_path, 'new_entropies_path.npy'))
    leaf_cells_table = attach_leaf_index_to_cells(leaves_cell_idx,all_cells)             # load cells given indices

    print "Saving under: {0}".format('results/clustering_algorithm_results/'+str(exp)+'/cells_with_leaves_table.csv')
    leaf_cells_table_df = pd.DataFrame(leaf_cells_table,columns=(channel_dic_names + ['Cidx','Hidx','Lidx']))
    leaf_cells_table_df.to_csv('{0}\{1}'.format(exp_path, 'cells_with_leaves_table.csv'))
    print "___________________________________________________"

if __name__ == "__main__":
    # Accepting parameter - exp cluster visualization
    exp = sys.argv[1]
    exps_path = os.path.join("results","our_results","clustering_algorithm_results")
    if str(exp) not in os.listdir(exps_path):
        print "Experiment does not exist!\n" \
        "Possible experiments:\n" \
        "{0}".format("\n".join(os.listdir(exps_path)))
        exit(-1)

    # parameters
    exp_params = exp.split('_')
    sample_size = int(exp_params[2])
    min_entropy = float(exp_params[5])
    min_cells = int(exp_params[8])

    print "=================================================="
    print "      Analyze Results - T-SNE visualization"
    print "=================================================="

    results = 'results'
    our_results = os.path.join('results','our_results')
    tsne_dir = 'tsne_results'
    clustering_algorithm_exps = 'clustering_algorithm_results'
    exps_path = os.path.join(our_results, clustering_algorithm_exps)

    # make new tsne result directory
    if not os.path.exists(os.path.join(results, tsne_dir)):
        os.mkdir(os.path.join(results, tsne_dir))

    exp_path = os.path.join(our_results,clustering_algorithm_exps,exp)
    find_cells(exp,exp_path)

    # new fold path
    new_fold_path = os.path.join(results, tsne_dir, exp)

    # make fold for exp
    if not os.path.exists(new_fold_path):
        os.mkdir(new_fold_path)

    # find csv cell leaves file
    csv_file = os.path.join(exps_path, exp, 'cells_with_leaves_table.csv')

    table_csv_file = 'mean_table.csv'
    table_pkl_file = 'mean_table.pkl'              # median/mean file name

    sizes_csv_file = 'cluster_sizes.csv'            # cluster file name
    sizes_pkl_file = 'cluster_sizes.pkl'

    features = ['CD3','CD8a','CD4','CD57', 'CD28', 'CD27', 'CD45RA', 'CD69', 'CD25', 'CD33', 'CD45RO', 'CD127', 'CD56','CD62L', 'CD45']

    # calculate cluster sizes file
    if not os.path.exists(os.path.join(new_fold_path, sizes_csv_file)):
        df_data = pd.read_csv(csv_file,sep = ',')
        cells_data = df_data[features+['Lidx']]
        cluster_sizes = cells_data.groupby(['Lidx']).size()
        cluster_sizes.to_csv('{0}\{1}'.format(new_fold_path, sizes_csv_file), ',')
        with open(os.path.join(new_fold_path, sizes_pkl_file), "w") as outfile:
            pickle.dump(cluster_sizes, outfile)
        print "Finding clusters sizes..."
        print "Saving under: {0}".format('results/tsne_results/'+str(exp)+'/cluster_sizes.csv')
        print "___________________________________________________"


    # calculate median/mean vectors table
    if not os.path.exists(os.path.join(new_fold_path, table_pkl_file)):
        df_data = pd.read_csv(csv_file,sep = ',')
        cells_data = df_data[features+['Lidx']]
        vectors_table = cells_data.groupby(['Lidx']).mean() # median/mean calculation
        vectors_table.to_csv('{0}\{1}'.format(new_fold_path, table_csv_file), ',')
        with open(os.path.join(new_fold_path, table_pkl_file), "w") as outfile:
            pickle.dump(vectors_table, outfile)
        print "Finding mean vectors..."
        print "Saving under: {0}".format('results/tsne_results/'+str(exp)+'/mean_table.csv')
        print "___________________________________________________"

    # load cluster num and vectors matrix
    cluster_sizes_pkl = open(os.path.join(new_fold_path, sizes_pkl_file))
    cluster_sizes = pickle.load(cluster_sizes_pkl).as_matrix()
    vectors_table_pkl = open(os.path.join(new_fold_path, table_pkl_file))
    df_vectors_table = pickle.load(vectors_table_pkl)

    # create range dictionary of each marker
    range_dict = dict()
    for marker in features:
         range_dict[marker] = (float(df_vectors_table[marker].min()), float(df_vectors_table[marker].max()))

    with open(os.path.join(new_fold_path,'markers_ranges.csv'),'wb') as f:
        w = csv.writer(f)
        w.writerows(range_dict.items())

    # Perform TSNE
    table_embedded_csv = 'mean_embedded.csv'

    if not os.path.exists(os.path.join(new_fold_path, sizes_csv_file)):
        df_data = pd.read_csv(csv_file,sep = ',')

    # preplexity - a guess about the number of close neighbors each point has. between 2 to 50
    # learning rate is from 10 to 1000 default 200
    # n_iter at least 250 default is 1000
    if min_cells == 100000:
        perplexity = 5
        learning_rate = 100
    else:
        perplexity = 30
        learning_rate = 240


    table_embedded = TSNE(n_components=2, perplexity=perplexity,learning_rate=learning_rate,n_iter=5000).fit_transform(df_vectors_table) # 158 x 2
    df_table_embedded = pd.DataFrame(table_embedded, columns=['x', 'y'])
    df_table_embedded['size'] = cluster_sizes  # add cluster sizes column
    df_table_embedded.to_csv('{0}\{1}'.format(new_fold_path, table_embedded_csv), ',')
    print "Performing T-SNE reduction..."
    print "Saving under: {0}".format('results/tsne_results/'+str(exp)+'/mean_embedded.csv')

    # columns
    x = df_table_embedded['x']
    y = df_table_embedded['y']
    sizes = df_table_embedded['size']
    print "___________________________________________________"
    print "Plot T-SNE visualization..."
    print "Saving under: {0}".format('results/tsne_results/'+str(exp)+'/Visualization_mean_'+str(perplexity)+'_'+str(learning_rate)+'.pdf')

    # plot T-SNE transformation
    with PdfPages(os.path.join(new_fold_path,'Visualization_mean_'+str(perplexity)+'_'+str(learning_rate)+'.pdf')) as pdf:
        for marker in features:

            # set plot params
            df_table_embedded[marker] = df_vectors_table[marker]            # add temp median/mean column of marker for coloring
            small_size_bins = [2000, 5000, 10000, 20000, 25000, 50000]            # ranges of cluster sizes
            big_size_bins = [50000, 60000, 70000, 80000, 90000, 100000]
            small_labels = ['2000', '5000', '10000', '20000', '25000','+50000']   # legend labels
            big_labels = ['50000', '60000', '70000', '80000', '90000', '+100000']

            if min_cells == 100000:
                size_bins = big_size_bins
                labels = big_labels
            if min_cells == 20000:
                size_bins = small_size_bins
                labels = small_labels

            factor_bins = [0.25, 0.5, 1, 2, 3, 4, 9]                            # scatter point sizes
            grouped = df_table_embedded.groupby(np.digitize(sizes, size_bins))  # group by bins
            group_indices = range(0,len(size_bins)+1)                           # 7 ranges : 0 group : size < 2k 1 group : 2k < size < 5k
            area_sizes = [50*i for i in factor_bins]                            # for scattering

            # figure settings
            plt.figure(figsize=(12,9))
            ax = plt.subplot(111)
            ax.spines["top"].set_visible(False) # Remove the plot frame lines.
            ax.spines["bottom"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            # scatter plot,  plot every group of sizes and color by marker range
            for i, (g_idx, group) in enumerate(grouped):
                plt.scatter(group['x'], group['y'], s=area_sizes[g_idx],
                            c=group[marker],cmap=cm.viridis,vmin=range_dict[marker][0],vmax=range_dict[marker][1])

            del df_table_embedded[marker]      # remove temp marker column
            plt.colorbar()

            # titles
            plt.xlabel('T-SNE x value',fontsize=16)
            plt.ylabel('T-SNE y value',fontsize=16)
            plt.title('T-SNE Visualization of Clusters - '+marker,fontsize=20)

            # custom legend
            leg_list =[]
            for size in area_sizes:
                leg = plt.scatter([],[], s=size, marker='o',color='black',alpha=0.5)
                leg_list.append(leg)

            ymin,ymax = plt.ylim()
            xmin,xmax = plt.xlim()

            plt.legend(leg_list,labels,scatterpoints=1,loc=9,fontsize=12,title='Population',bbox_to_anchor=(1.3, 0.4))

            #plt.show()
            # save pdf
            pdf.savefig()