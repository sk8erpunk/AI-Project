import numpy as np

from entities.human_being import human_being


class simple_sampler_by_size:
    def __init__(self, sample_size):
        self.sample_size = sample_size

    def sample_from_humans(self, human_list):
        humans_data = []
        for human in human_list:
            shuffled_cells = human.cells_features
            sample_size = self.sample_size
            np.random.shuffle(shuffled_cells)
            humans_data.append(human_being(shuffled_cells[0:sample_size], fcs_file=human.fcs_file, file_idx=human.human_idx))
        return humans_data
