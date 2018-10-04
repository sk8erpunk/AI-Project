import os
import pickle
import numpy as np
from FlowCytometryTools import FCMeasurement
from entities.human_being import get_all_cells_features
from entities.human_being import human_being
from samplers import simple_sampler_by_size


def solve_through_cluster_manifold_classifier_validator(human_list, labels, classifier, validator, results_dir=None,
                         cluster=None, manifold=None, sampler=None, r_function=None, citrus_data=None):
    human_table = None

    # if sampler:
    #     human_list = sampler
    # if cluster:
    #     human_table = cluster.fit_predict(human_list)
    # if manifold:
    #     human_table = manifold.fit_transform(human_list)
    if r_function:
        human_table = np.load(os.path.join(results_dir, 'human_table.npy'))
    if citrus_data:
        human_table = np.load(os.path.join(citrus_data, 'human_table.npy'))

    return validator.get_scores(classifier, human_table, labels)  # calc scores


# gets a sample size per human
# samples data from all humans once and saves it
# returns sampler data
def sample_once_per_size(human_list, sample_size):

    sample_dir = 'sample_dir'
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    sample_size_dir = os.path.join(sample_dir, 'sample_size_'+ str(sample_size))  # sample directory

    if os.path.exists(os.path.join(sample_size_dir,'random_sample.pkl')):            # check if such sample exists
        sample_file = open(os.path.join(sample_size_dir,'random_sample.pkl'))        # use existed sample
        sampler_data = pickle.load(sample_file)
    else:                                                                       # else make sample directory
        os.makedirs(sample_size_dir)
        sampler = simple_sampler_by_size.simple_sampler_by_size(sample_size)    # sample obj
        sampler_data = sampler.sample_from_humans(human_list)                   # sample and save
        with open(os.path.join(sample_size_dir,'random_sample.pkl'), "w") as outfile:
            pickle.dump(sampler_data, outfile)

    return sampler_data


def get_tags_for(files_array):
    # create classification dict: key is the file name, value is 1 for responder and 0 for non-responder
    classification_dict = {}
    classification_reader = open(os.path.join("classifications", "AnnotationFull.csv"))
    classification_reader.next()
    for line in classification_reader:
        line = line.split(",")
        file_name = (line[7]).strip("\n\"\'\r")
        if line[4] == '"Responder"':
            classification_dict[file_name] = 1
        else:
            classification_dict[file_name] = 0
    classification_reader.close()
    return [classification_dict[x] for x in files_array]


def get_citrus_human_table(abundances_csv_file):
    # create classification dict: key is the file name, value is the abundances data
    classification_dict = {}
    files_array = []
    classification_reader = open(abundances_csv_file)
    classification_reader.next()
    for line in classification_reader:
        line = line.split(",")
        file_name = (line[0]).strip("\n\"\'\r")
        files_array.append(file_name)
        line_data = line[1:98]
        line_data = [str.replace("\"","") for str in line_data]
        classification_dict[file_name] = map(float, line_data)
    classification_reader.close()
    citrus_matrix = np.array([classification_dict[x] for x in files_array])

    return citrus_matrix


# returns human beings list
# human being cells list has 48 columns (cells has indexes and human index)
def get_human_list(data_dir):
    humans_data = []
    files_array = []
    for fcs_file, file_idx in zip(os.listdir(data_dir),range(len(os.listdir(data_dir)))):
        files_array.append(fcs_file)
        file_name_path = os.path.join(data_dir, fcs_file)
        human = FCMeasurement(ID='Test Sample', datafile=file_name_path)
        human_data = human.data.values
        human_data = np.delete(human_data, [0,1], 1) # cut the first two features (time and length)
        human_data = np.divide(human_data, 5)
        human_data = np.arcsinh(human_data)
        human_cells_with_index = add_indices_to_cells(human_data,file_idx) # adds two columns (cell index, human index)
        humans_data.append(human_being(human_cells_with_index, fcs_file, file_idx))
    return humans_data, files_array


# adds two columns, index for each human cell and human index column
def add_indices_to_cells(human_cells,human_index):
    n,m = human_cells.shape # n cells
    cell_indices = np.reshape(range(0,human_cells.__len__()) ,(n,1))   # column of [[0],[1], ... , [n-1]]
    human_indices = np.reshape([human_index]*n,(n,1))                  # column of [[idx],[idx], ... , [idx]]
    cells_indexed = np.hstack((human_cells,cell_indices))              # union column to table
    cells_with_human_index = np.hstack((cells_indexed,human_indices))  # last two columns are human index and cell index
    return cells_with_human_index


# load cells from data dir
def attach_leaf_index_to_cells(leaves_dic,all_cells):
    sample_cells_table = []
    all_cell_idx = all_cells[:,[46,47]]  # all cell indicies
    all_cell_idx_map = np.char.array(all_cell_idx[:,0])+'-'+np.char.array(all_cell_idx[:,1])
    # get original cells by indices from leaves
    for k in leaves_dic:
        cell_idx = leaves_dic[k]   # cells indicies in leaves
        cell_idx_map = np.char.array(cell_idx[:,0])+'-'+np.char.array(cell_idx[:,1]) # map to one array
        m_vec = np.in1d(all_cell_idx_map,cell_idx_map) # vector mask
        sample_cells = all_cells[m_vec]
        lc_num = len(leaves_dic[k])
        leaf_idx = np.reshape([k]*lc_num,(lc_num,1))    # leaf index column
        sample_cells = np.hstack((sample_cells,leaf_idx)) # attach leaf index column to leave cells
        sample_cells_table.append(sample_cells)
    return np.concatenate(tuple([c for c in sample_cells_table]))


# since now we saved the sample, this is a faster version.
def attach_leaf_index_to_cells_given_sample(leaves_dic,sample_size):
    sample_dir = 'samples_dir'
    sample_size_dir = 'sample_size_' + str(sample_size)
    sample_pkl_file = open(os.path.join(sample_dir,sample_size_dir,'random_sample.pkl'))
    all_cells = get_all_cells_features(pickle.load(sample_pkl_file))
    sample_cells_table = []
    all_cell_idx = all_cells[:,[46,47]]  # all cell indicies
    all_cell_idx_map = np.char.array(all_cell_idx[:,0])+'-'+np.char.array(all_cell_idx[:,1])
    # get original cells by indices from leaves
    for k in leaves_dic:
        cell_idx = leaves_dic[k]   # cells indices in leaf k
        cell_idx_map = np.char.array(cell_idx[:,0])+'-'+np.char.array(cell_idx[:,1]) # map to one array
        m_vec = np.in1d(all_cell_idx_map,cell_idx_map) # vector mask
        sample_cells = all_cells[m_vec]                # which cells are in k leaf
        lc_num = len(leaves_dic[k])
        leaf_idx = np.reshape([k]*lc_num,(lc_num,1))    # leaf index column
        sample_cells = np.hstack((sample_cells,leaf_idx)) # attach leaf index column to leave cells
        sample_cells_table.append(sample_cells)
    return np.concatenate(tuple([c for c in sample_cells_table]))