import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np


def plotAllExp(df,gdp):
    folds = [2, 4, 5, 8, 10, 20]
    fold_dict = dict()
    for fold in folds:
        accuracies = df.loc[df['n_folds'] == fold]['avg']
        fold_dict[fold] = accuracies.as_matrix()

    plt.figure(figsize=(12,9))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False) # Remove the plot frame lines.
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.ylim(0, 1)
    plt.xlim(0, 21)
    plt.yticks(np.arange(0, 1, 0.1), [str(int(x*100)) + "%" for x in np.arange(0, 1, 0.1)], fontsize=14)
    plt.xticks(np.arange(0, 21),fontsize=14)
    # titles
    plt.xlabel('Folds',fontsize=16)
    plt.ylabel('Accuracy rate',fontsize=16)
    plt.title('Classifiers Accuracy vs K-Fold',fontsize=20)
    colors = ['k','b','r','g','y','c','m','crimson']
    labels = range(1,9)
    for i in range(0,8):
        exp_accuracies = []
        for fold in folds:
            exp_accuracies.append(fold_dict[fold][i])
        plt.scatter(folds,exp_accuracies)
        plt.plot(folds,exp_accuracies,color=colors[i],label=labels[i])
    plt.grid(True)
    plt.legend(title='Experiments',loc=9 ,bbox_to_anchor=(1.05, 0.7))
    path = 'results/graphs_results/all_exp_graph.png'
    plt.savefig(path)
    print "Saving under: {0}".format(path)


def plotAllExp2(df,gdp):
    folds = [2, 4, 5, 8, 10, 20]
    fold_dict = dict()
    for fold in folds:
        accuracies = df.loc[df['n_folds'] == fold]['avg']
        fold_dict[fold] = accuracies.as_matrix()
    Exps = ['Min Cells 100000\nMin Entropy 0.05','Min Cells 20000\nMin Entropy 0.05',
            'Min Cells 100000\nMin Entropy 0.1','Min Cells 20000\nMin Entropy 0.1',
            'Min Cells 100000\nMin Entropy 0.3','Min Cells 20000\nMin Entropy 0.3',
            'Min Cells 100000\nMin Entropy 0.5','Min Cells 20000\nMin Entropy 0.5']
    colors = ['k','b','r','g','y','c','m','crimson']
    labels = range(1,9)
    for i in range(0,8):
        plt.figure(figsize=(12,9))
        ax = plt.subplot(111)
        ax.spines["top"].set_visible(False) # Remove the plot frame lines.
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        plt.ylim(0, 1)
        plt.xlim(0, 21)
        plt.yticks(np.arange(0, 1, 0.1), [str(int(x*100)) + "%" for x in np.arange(0, 1, 0.1)], fontsize=14)
        plt.xticks(np.arange(0, 21),fontsize=14)
        # titles
        plt.xlabel('Folds',fontsize=16)
        plt.ylabel('Accuracy rate',fontsize=16)
        plt.title('Classifier Accuracy vs K-Fold',fontsize=20)
        exp_accuracies = []
        for fold in folds:
            exp_accuracies.append(fold_dict[fold][i])
        plt.scatter(folds,exp_accuracies)
        plt.plot(folds,exp_accuracies,color=colors[i],label =Exps[i],linewidth=2)
        for a,b in zip(folds,exp_accuracies):
            plt.text(a, b, str(b*100)+"%", fontsize=14)
        plt.grid(True)
        plt.legend(title='Experiment' ,fontsize=14)
        path = 'results/graphs_results/exp_graph_'+str(i+1)+'.png'
        plt.savefig(path)
        plt.clf()


def plotFoldAvgs(df,gdp):
    folds = [2, 4, 5, 8, 10, 20]
    plt.figure(figsize=(12,9))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False) # Remove the plot frame lines.
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.ylim(0, 1)
    plt.xlim(0, 21)
    plt.yticks(np.arange(0, 1, 0.1), [str(int(x*100)) + "%" for x in np.arange(0, 1, 0.1)], fontsize=14)
    plt.xticks(np.arange(0, 21),fontsize=14)
    # titles
    plt.xlabel('Folds',fontsize=16)
    plt.ylabel('Accuracy Average',fontsize=16)
    plt.title('Average Accuracy vs K-Fold',fontsize=20)
    colors = ['blue','red']
    labels = ['20000','100000']

    for i,mc in enumerate([20000,100000]):
        fold_dict = dict()
        for fold in folds:
            sf = df.loc[df['min_cells'] == mc]
            accuracies = sf.loc[df['n_folds'] == fold]['avg']
            fold_dict[fold] = round(np.average(accuracies.as_matrix()),2)
        avg_accuracies = fold_dict.values()
        plt.scatter(folds,avg_accuracies)
        plt.plot(folds,avg_accuracies,color=colors[i],label=labels[i])
        for a,b in zip(folds,avg_accuracies):
            plt.text(a, b, str(b*100)+"%", fontsize=14)
    plt.grid(True)

    plt.legend(title='Minimal Cells',loc=9 ,bbox_to_anchor=(1.04, 0.5))
    path = 'results/graphs_results/avgs_fold_graph.png'
    plt.savefig(path)
    print "Saving under: {0}".format(path)


def plotEntAvgs(df,gdp):
    entropies = df['min_entropy'].drop_duplicates()
    entropies = entropies.as_matrix()

    plt.figure(figsize=(12,9))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False) # Remove the plot frame lines.
    #ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    #ax.spines["left"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.ylim(0, 1)
    plt.xlim(0, 0.6)
    plt.yticks(np.arange(0, 1, 0.1), [str(int(x*100)) + "%" for x in np.arange(0, 1, 0.1)], fontsize=14)
    plt.xticks(np.arange(0, 0.6, 0.05),fontsize=14)

    # titles
    plt.xlabel('Minimal Entropy',fontsize=16)
    plt.ylabel('Accuracy Average',fontsize=16)
    plt.title('Average Accuracy vs Minimal Entropy',fontsize=20)
    colors = ['blue','red']
    labels = ['20000','100000']

    for i,mc in enumerate([20000,100000]):
        ent_dict = dict()
        avg_accuracies = []
        for ent in entropies:
            sf = df.loc[df['min_cells'] == mc]
            accuracies = sf.loc[df['min_entropy'] == ent]['avg']
            ent_dict[ent] = round(np.average(accuracies.as_matrix()),2)
            avg_accuracies.append(ent_dict[ent])
        plt.scatter(entropies,avg_accuracies)
        plt.plot(entropies,avg_accuracies,color=colors[i],label=labels[i])
        for a,b in zip(entropies,avg_accuracies):
            plt.text(a, b, str(b*100)+"%", fontsize=14)
    plt.grid(True)

    plt.legend(title='Minimal Cells',loc=9 ,bbox_to_anchor=(1.04, 0.5))
    path = 'results/graphs_results/avgs_entropy_graph.png'
    plt.savefig(path)
    print "Saving under: {0}".format(path)

def plotEntropies(df,gdp):
    entropies = df['min_entropy'].drop_duplicates()
    entropies = entropies.as_matrix()

    # figure settings
    plt.figure(figsize=(12,9))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False) # Remove the plot frame lines.
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.ylim(0, 1)
    plt.xlim(0, 0.6)
    plt.yticks(np.arange(0, 1, 0.1), [str(int(x*100)) + "%" for x in np.arange(0, 1, 0.1)], fontsize=14)
    plt.xticks(np.arange(0, 0.6, 0.05),fontsize=14)

    # titles
    plt.xlabel('Minimal Entropy',fontsize=16)
    plt.ylabel('Accuracy rate',fontsize=16)
    plt.title('Best Accuracy vs Minimal Entropy',fontsize=20)
    colors = ['blue','red']
    labels = ['20000','100000']

    for i,mc in enumerate([20000,100000]):
        sf = df.loc[df['min_cells'] == mc]
        accuracies = sf['avg']
        avg_accuracies = accuracies.as_matrix()
        plt.scatter(entropies,avg_accuracies)
        plt.plot(entropies,avg_accuracies,color=colors[i],label=labels[i])
        for a,b in zip(entropies,avg_accuracies):
            plt.text(a, b, str(b*100)+"%", fontsize=14)
    plt.grid(True)

    plt.legend(title='Minimal Cells',loc=9 ,bbox_to_anchor=(1.04, 0.5))
    path = 'results/graphs_results/entropy_graphs.png'
    plt.savefig(path)
    print "Saving under: {0}".format(path)

def plotEntWithFold(df,gdp,fold):
    entropies = df['min_entropy'].drop_duplicates()
    entropies = entropies.as_matrix()

    plt.figure(figsize=(12,9))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False) # Remove the plot frame lines.
    #ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    #ax.spines["left"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.ylim(0, 1)
    plt.xlim(0, 0.6)
    plt.yticks(np.arange(0, 1, 0.1), [str(int(x*100)) + "%" for x in np.arange(0, 1, 0.1)], fontsize=14)
    plt.xticks(np.arange(0, 0.6, 0.05),fontsize=14)

    # titles
    plt.xlabel('Minimal Entropy',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy vs Minimal Entropy \n Fold = {0}'.format(fold),fontsize=20)
    colors = ['blue','red']
    labels = ['20000','100000']

    for i,mc in enumerate([20000,100000]):
        ent_dict = dict()
        avg_accuracies = []
        for ent in entropies:
            sf = df.loc[df['min_cells'] == mc]
            sf = sf.loc[df['n_folds'] == fold]
            accuracies = sf.loc[df['min_entropy'] == ent]['avg']
            ent_dict[ent] = accuracies.as_matrix()[0]
            avg_accuracies.append(ent_dict[ent])
        plt.scatter(entropies,avg_accuracies)
        plt.plot(entropies,avg_accuracies,color=colors[i],label=labels[i])

        for a,b in zip(entropies,avg_accuracies):
            plt.text(a, b, str(b*100)+"%", fontsize=14)
    plt.grid(True)

    plt.legend(title='Minimal Cells',loc=9 ,bbox_to_anchor=(1.04, 0.5))
    path = 'results/graphs_results/entropy_graph_fold_{0}.png'.format(fold)
    plt.savefig(path)
    print "Saving under: {0}".format(path)


# if __name__ == '__main__':
#
#     # make new graphs result directory
#     results_dir = 'results'
#     graph_dir = 'graphs_results'
#     best_exps_csv = 'best_avgs_per_exp.csv'
#     all_exps = 'results_table.csv'
#
#     # paths for files
#     files_dir = os.path.join(results_dir,'our_results','experiments_summary')
#     best_exps_path = os.path.join(files_dir,best_exps_csv)
#     all_exps_path = os.path.join(files_dir,all_exps)
#     graph_dir_path = os.path.join(results_dir,graph_dir)
#
#     # plot all experiments
#     df1 = pd.read_csv(all_exps_path,sep=',')
#     plotAllExp(df1,graph_dir_path)
#
#     # plot entropies and accuracies with given fold
#     plotEntWithFold(df1,graph_dir_path,20)


