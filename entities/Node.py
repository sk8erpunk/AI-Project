import pickle
import copy
from helpers.channel_helper_functions import channel_to_index

POSITIVE = True     # larger than thresh
NEGATIVE = False    # smaller or equal to thresh

class Node:
    '''
    thresh: dict of tuples: {name: (thresh, entropy)}
    path: list of tuples: [(name, isPositive)]
    '''
    def __init__(self, cells, marker=None, is_positive=None, thresh=None, parent=None, entropy=None):
        self.parent = parent
        self.cells = cells
        self.thresh = thresh
        self.path = []
        self.sons = []
        self.entropies = []

        if parent:
            assert not(is_positive is None or marker is None)
            self.path = list(parent.path)
            self.path.append((marker, is_positive))
            self.entropies = list(parent.entropies)
            self.entropies.append((marker,entropy))

            # update pre_knowledge markers list
            if parent.pre_knowledge_markers:
                updated_pre_knowledge_markers = copy.deepcopy(parent.pre_knowledge_markers)
                updated_pre_knowledge_markers.remove(marker)
                self.pre_knowledge_markers = updated_pre_knowledge_markers
            else:
                self.pre_knowledge_markers = []
        else:
            # initial pre knowledge markers list for root
            self.pre_knowledge_markers = ['CD3','CD8a','CD4']

    # returns percentage
    def getFreq(self, human_ind, human_total_cells_num):
        if self.parent is None:
            return 100
        assert not(self.path is None)
        cur_human = self.cells[self.cells[:, -1] == human_ind, :]
        size = len(cur_human)
        return (float(size)/float(human_total_cells_num))*100

    # returns 0 or 1 !!!
    def _isInCluster(self, cell):
        for pos in self.path:
            i = self._getMarkerIndexByName(pos[0])
            if not ((cell[i] >= self.thresh[pos[0]][0]) == pos[1]):
                return 0
        return 1

    def _getMarkerIndexByName(self, name):
        return channel_to_index(name) - 1  # (-1): index in table starts from 0
