import numpy as np

from entities.human_being import human_being


class simple_sampler_by_percentage:
    def __init__(self, sample_percentage):
        self.sample_percentage = sample_percentage

    def sample_from_humans(self, human_list):
        humans_data = []
        for human in human_list:
            shuffled_cells = human.cells_features
            sample_size = int(round(human.get_cells_number()*self.sample_percentage))
            np.random.shuffle(shuffled_cells)
            humans_data.append(human_being(shuffled_cells[0:sample_size], fcs_file=human.fcs_file, file_idx=human.human_idx))
        return humans_data
