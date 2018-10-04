import numpy as np


class human_being:
    def __init__(self, cells_features, fcs_file, file_idx):
        self.cells_features = cells_features
        self.fcs_file = fcs_file
        self.human_idx = file_idx
        
    def get_cell(self, index):
        if index >= self.get_cells_number():
            return np.array([])
        return self.cells_features[index, :]
    
    def get_feature(self, index):
        if index >= self.get_features_number():
            return np.array([])
        return self.cells_features[:, index]

    def get_cells_number(self):
        return self.cells_features.shape[0]
        
    def get_features_number(self):
        return self.cells_features.shape[1]


def get_max_features_number(humans):
    features_numbers = []
    for human in humans:
        features_numbers.append(human.get_features_number())
    return max(features_numbers)


def get_all_cells_features(humans):
    return np.concatenate(tuple([h.cells_features for h in humans]))


def get_all_cells_features_with_idx(humans):
    return np.concatenate(tuple([np.append(h.cells_features,np.full((h.cells_features.shape[0],1),h.human_idx,dtype=np.int),axis=1) for h in humans]))


if __name__ == "__main__":
    arr = np.zeros((2,3))
    arr = arr +1
    print(arr)