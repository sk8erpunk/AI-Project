import copy
import math

import scipy as scp
from scipy import stats

from calc_thresh.calc_thresholds import calculate_thresholds
from entities.Node import Node
from helpers.channel_helper_functions import get_channels_names_from_pickle


class Tree:
    def __init__(self,cells, minimal_entropy, minimal_cells):
        self.minimal_entropy = minimal_entropy
        self.minimal_cells = minimal_cells
        self.root = Node(cells=cells)
        self.leaves = []
        self._recursive_init(self.root)

    def split(self,node):

        # stop if less than minimal cells condition
        if node.cells.shape[0] < self.minimal_cells:
            node.sons.extend([None, None])
            return

        original_markers_list = get_channels_names_from_pickle()    # 46 markers
        markers_list = ['CD57', 'CD28', 'CD27', 'CD45RA', 'CD69', 'CD25', 'CD33', 'CD45RO', 'CD127', 'CD56','CD62L', 'CD45']
        pre_knowledge_markers = ['CD3','CD8a','CD4']
        markers_threshes = []

        # first use all pre_knowledge_markers by given order
        if node.pre_knowledge_markers:
            pre_marker_threshes = calculate_thresholds(node.cells,node.pre_knowledge_markers)
            pre_marker_threshes = {k: v for k,v in pre_marker_threshes.iteritems() if v == v} # filter nan values
            if len(pre_marker_threshes) == 0:      # if it cant choose pre_knowledge
                node.pre_knowledge_markers = []
            else:
                new_markers_list = [node.pre_knowledge_markers[0]]
                markers_threshes = pre_marker_threshes

        if not node.pre_knowledge_markers:      # start splitting by markers list
            visited = [x[0] for x in node.path]
            new_markers_list = [x for x in markers_list if x not in visited]

            if not new_markers_list:
                node.sons.extend([None, None])
                return

        if not markers_threshes:
            # calculates threshes of non pre knowledge markers
            markers_threshes = calculate_thresholds(node.cells, new_markers_list)  # threshes is a dictionary. threshes[name]=thresh_val

        # entropy calculation
        threshes = {}
        entropies = {}

        for marker in new_markers_list:
            curr_thresh = markers_threshes[marker]
            if math.isnan(curr_thresh):        # if nan (no thresh is available for this marker) not relevant for pre knowledge markers
                entropies[marker] = -1
                threshes[marker] = None
                continue

            # following code finds max entropy of a marker
            curr_marker_idx = original_markers_list.index(marker)
            cells_plus_num = (node.cells[(curr_thresh <= node.cells[:,curr_marker_idx])]).shape[0]
            cells_minus_num = (node.cells[(curr_thresh > node.cells[:,curr_marker_idx])]).shape[0]
            total_cells_num = cells_plus_num + cells_minus_num
            assert total_cells_num > 0
            pk1 = float(cells_plus_num)/total_cells_num
            pk2 = float(cells_minus_num)/total_cells_num
            assert pk1 + pk2 == 1
            entropies[marker] = scp.stats.entropy([pk1,pk2])
            threshes[marker] = curr_thresh

        # choose marker by entropy
        marker_chosen = max(entropies, key=entropies.get)
        max_entropy = entropies[marker_chosen]

        # check entropy thresh and ignore if it is a pre knowledge marker
        if max_entropy < self.minimal_entropy and (marker_chosen not in pre_knowledge_markers):
            # print "Reached minimal entropy. stop splitting {0}".format(max_entropy)
            node.sons.extend([None, None])
            return

        thresh_chosen = markers_threshes[marker_chosen]
        thresh_chosen_idx = original_markers_list.index(marker_chosen)

        # calculate cells number of by chosen marker
        cells_plus = node.cells[(thresh_chosen <= node.cells[:, thresh_chosen_idx])]
        cells_minus = node.cells[(thresh_chosen > node.cells[:, thresh_chosen_idx])]

        # create thresh dictionary for sons of curr node
        if node.thresh:
            thresh_dict_for_sons = copy.deepcopy(node.thresh)
        else:
            thresh_dict_for_sons = {}

        # update thresh dictionary for sons with the new marker chosen thresh
        thresh_dict_for_sons[marker_chosen] = threshes[marker_chosen]

        # build children nodes
        node_plus = Node(parent=node, cells=cells_plus, thresh=thresh_dict_for_sons, marker=marker_chosen, is_positive=True,entropy=max_entropy)
        node_minus = Node(parent=node, cells=cells_minus, thresh=thresh_dict_for_sons, marker=marker_chosen, is_positive=False,entropy=max_entropy)

        # append sons
        node.sons.append(node_plus)
        node.sons.append(node_minus)

        node.cells = None

    def _recursive_init(self,node):
        # print("splits")
        self.split(node)
        # print("split is over")
        if node.sons[0] is not None:
            self._recursive_init(node.sons[0])
        if node.sons[1] is not None:
            self._recursive_init(node.sons[1])
        if node.sons[0] is None and node.sons[1] is None:
            self.leaves.append(node)

    def get_leaves(self):
        return list(self.leaves)

