# -*- coding: utf-8 -*-
import networkx as nx
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import norm as spnorm

import random
import math
from graph_kernel.WL_subtree_kernel import compute_mle_wl_kernel
#from numba import jit
from graph.dataset import load

def load_npz_edges(file_name):
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    dict_of_lists = {}
    with np.load(file_name) as loader:
        loader = dict(loader)
        num_nodes = loader['adj_shape'][0]
        indices = loader['adj_indices']
        indptr = loader['adj_indptr']
        for i in range(num_nodes):
            if len(indices[indptr[i]:indptr[i+1]]) > 0:
                dict_of_lists[i] = indices[indptr[i]:indptr[i+1]].tolist()

    return dict_of_lists


def load_npz(file_name):
    """Load a SparseGraph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    sparse_graph : gust.SparseGraph
        Graph in sparse matrix format.

    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name,allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                              loader['adj_indptr']), shape=loader['adj_shape'])
        #import pdb; pdb.set_trace()
        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                                   loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            attr_matrix = None

        labels = loader.get('labels')

    return adj_matrix, attr_matrix, labels


def largest_connected_components(adj, n_components=1):
    """Select the largest connected components in the graph.返回最大连通子图

    Parameters
    ----------
    sparse_graph : gust.SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.

    Returns
    -------
    sparse_graph : gust.SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.

    """
    _, component_indices = connected_components ( adj ) #Return the length-N array of each node's label in the connected components.
    component_sizes = np.bincount(component_indices) #Count number of occurrences of each value in array of non-negative ints.
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending 最大连通子图中的节点存成list
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep


    ]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


def train_val_test_split_tabular(*arrays, train_size=0.5, val_size=0.3, test_size=0.2, stratify=None, random_state=None):
    """
    Split the arrays or matrices into random train, validation and test subsets.

    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
            Allowed inputs are lists, numpy arrays or scipy-sparse matrices.
    train_size : float, default 0.5
        Proportion of the dataset included in the train split.
    val_size : float, default 0.3
        Proportion of the dataset included in the validation split.
    test_size : float, default 0.2
        Proportion of the dataset included in the test split.
    stratify : array-like or None, default None
        If not None, data is split in a stratified fashion, using this as the class labels. #是否为标签均衡的数据
    random_state : int or None, default None
        Random_state is the seed used by the random number generator;

    Returns
    -------
    splitting : list, length=3 * len(arrays)
        List containing train-validation-test split of inputs.

    """
    if len(set(array.shape[0] for array in arrays)) != 1:
        raise ValueError("Arrays must have equal first dimension.")
    idx = np.arange(arrays[0].shape[0])
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=random_state,
                                                   train_size=(train_size + val_size),
                                                   test_size=test_size,
                                                   stratify=stratify)
    if stratify is not None:
        stratify = stratify[idx_train_and_val]
    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=random_state,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)
    result = []
    for X in arrays:
        result.append(X[idx_train])
        result.append(X[idx_val])
        result.append(X[idx_test])
    return result

def preprocess_graph(adj):
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = adj_.sum(1).A1 #sum_{j}(A_{ij})
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5)) #D^(-1/2)
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).T.dot(degree_mat_inv_sqrt).tocsr() #求A^hat并将其转化为Compressed Sparse Row format
    #import pdb; pdb.set_trace()
    #adj_normalized = adj_
    return adj_normalized

'''
def cal_scores(A):
    eig_vals, eig_vecl = linalg.eig(A.todense(), left = True, right = False)
    eig_vals, eig_vecr = linalg.eig(A.todense(), left = False, right = True)
    eig_idx = eig_vals.argmax()
    eig_l = eig_vecl[eig_idx]
    eig_r = eig_vecr[eig_idx]
    eig_l[eig_l < 0 ] = -eig_l[eig_l < 0 ]
    eig_r[eig_r < 0 ] = -eig_r[eig_r < 0 ]
    print ("The largest eigenvalue of A is {}".format(eig_vals.max()))
'''

'''
def cal_scores(A):
    eig_vals, eig_vec = linalg.eigh(A.todense())
    eig_idx = eig_vals.argmax()
    eig = eig_vec[eig_idx]
    eig[eig < 0 ] = -eig[eig < 0 ]
    print ("The largest eigenvalue of A is {}".format(eig_vals.max()))
    scores = (eig.reshape(eig.shape[0],1) * eig).flatten()
    #import pdb; pdb.set_trace()
    return scores
'''



def k_edgedel(A, scores, dict_of_lists, k):
    N = A.shape[0]
    idxes = np.argsort(-scores)
    edge_pert = []
    for p in idxes:
        x = p // N
        y = p % N
        if x == y:
            continue
        if x not in dict_of_lists.keys() or y not in dict_of_lists.keys():
            continue
        if not x in dict_of_lists[y] or not y in dict_of_lists[x]:
            continue
        edge = np.array([x, y])
        if np.isin(edge[::-1], edge_pert).all():
           continue
        print ("The best edge for deletion is ({0} {1}), with score {2}".format(x, y, scores[p]))
        edge_pert.append(edge)
        #import pdb; pdb.set_trace()
        if len(edge_pert) >= k:
            break
    for edge in edge_pert:
        #import pdb; pdb.set_trace()
        A[tuple(edge)] = A[tuple(edge[::-1])] = 1 - A[tuple(edge)]
    adj = preprocess_graph(A)
    return adj, edge_pert

def randomly_add_edges(adj, k, num_node):
    num_nodes = adj.shape[0]
    adj_out = adj.copy()
    ## find the position which need to be add
    adj_orig_dense = adj.todense()
    flag_adj = np.triu(np.ones([num_nodes, num_nodes]), k=1) - np.triu(adj_orig_dense, k=1)
    flag_adj = flag_adj[:num_node, :num_node]
    idx_list = np.argwhere(flag_adj == 1)
    if len(idx_list) == 0:
        # print("there is a full graph!!")
        return adj, 0
    selected_idx_of_idx_list = np.random.choice(len(idx_list),size = min(max(k,1), len(idx_list)), replace = False)
    selected_idx = idx_list[selected_idx_of_idx_list]
    adj_out[selected_idx[:,0],selected_idx[:,1]] = 1
    adj_out[selected_idx[:, 1], selected_idx[:, 0]] = 1
    return adj_out, min(max(k,1), len(idx_list))

def add_edges_between_labels(adj,k, y_train,seed = 152):
    "add edges"
    np.random.seed(seed)
    num_nodes = adj.shape[0]
    adj_out = adj.copy()
    adj_orig_dense = adj.todense()
    flag_adj = np.triu(np.ones([num_nodes, num_nodes]), k = 1) - np.triu(adj_orig_dense,k=1)
    idx_list = np.argwhere(flag_adj == 1)
    different_labels_edges = []
    for idxes in idx_list:
        if ((not (y_train[idxes[0], :] == y_train[idxes[1],:]).all()) and
            y_train[idxes[0],:].sum() !=0 and y_train[idxes[1],:].sum() !=0 ):
            different_labels_edges.append(idxes)
    different_labels_edges = np.array(different_labels_edges)
    selected_idx_of = np.random.choice(len(different_labels_edges), size = k)
    selected_edges = different_labels_edges[selected_idx_of]
    adj_out[selected_edges[:,0], selected_edges[:,1]] = 1
    adj_out[selected_edges[:,1], selected_edges[:,0]] = 1
    add_idxes = selected_edges[:,0] * adj_orig_dense.shape[0] + \
                selected_edges[:,1]
    add_idxes_other = selected_edges[:,1] * adj_orig_dense.shape[0] + \
                selected_edges[:,0]
    add_idxes = np.append(add_idxes, add_idxes_other)
    return adj_out,  add_idxes


def randomly_delete_edges(adj, k):
    num_nodes = adj.shape[0]
    adj_out = adj.copy()
    ## find the position which need to be add
    adj_orig_dense = adj.todense()
    flag_adj = np.triu(adj_orig_dense, k=1)
    idx_list = np.argwhere(flag_adj == 1)
    selected_idx_of_idx_list = np.random.choice(len(idx_list), size=k, replace = False)
    selected_idx = idx_list[selected_idx_of_idx_list]
    adj_out[selected_idx[:, 0], selected_idx[:, 1]] = 0
    adj_out[selected_idx[:, 1], selected_idx[:, 0]] = 0

    return adj_out

def randomly_flip_features(features, k,seed):
    np.random.seed(seed)
    num_node = features.shape[0]
    num_features = features.shape[1]
    features_lil = features.tolil()
    flip_node_idx_select = np.random.choice(num_node, size = max(num_node, 400), replace = False)   ## select 100 node
    flip_node_idx = np.random.choice(flip_node_idx_select, size=k,replace = True)
    flip_fea_idx_select = np.random.choice(num_features, size = 10, replace = False)   ## select 2 features
    flip_fea_idx = np.random.choice(flip_fea_idx_select, size=k)
    ### this is the matrix one
    for i in range(len(flip_node_idx)):
        if features[flip_node_idx[i], flip_fea_idx[i]] == 1:
            features_lil[flip_node_idx[i], flip_fea_idx[i]] = 0
        else:
           features_lil[flip_node_idx[i], flip_fea_idx[i]] = 1
    return features_lil.tocsr()

def flip_features_fix_attr(features, k,seed, fixed_list, row_sum):
    np.random.seed(seed)
    num_node = features.shape[0]
    num_features = features.shape[1]
    features_lil = features.tolil()
    #flip_node_idx = np.random.choice(num_node, size = min(k, num_node), replace = False)   ## select 100 node
    flip_node_idx = np.argsort(row_sum)
    if k !=0:
        flip_node_idx = flip_node_idx[-1*min(int(k), num_node):]
    else:
        flip_node_idx = []
    #flip_fea_idx = np.random.choice(flip_fea_idx_select, size=k)
    ### this is the matrix one
    for i in range(len(flip_node_idx)):
        for j in fixed_list:
            features_lil[flip_node_idx[i], j] = 1 - features_lil[flip_node_idx[i],j]
        # if features[flip_node_idx[i], fixed_list] == 1:
        #     features_lil[flip_node_idx[i], flip_fea_idx[i]] = 0
        # else:
        #    features_lil[flip_node_idx[i], flip_fea_idx[i]] = 1
    return features_lil.tocsr()

def denoise_ratio(add_idxes, delete_idxes):
    """
    check the ratio between the add edges and delted edges
    """
    add_set = set(add_idxes)
    del_set = set(delete_idxes)
    union = add_set.intersection(del_set)
    num = len(union)
    ratio = num / len(add_set)
    return ratio, num

def get_noised_indexes(x_tilde, adj, num_node):
    noised_list = []
    clean_list = []
    node_comm = np.argmax(x_tilde, axis = 1)
    clusters = x_tilde.shape[1]
    for i in range(clusters):
        nodes_idx = np.argwhere(node_comm == i)[:,0]
        other_nodes_idx = np.argwhere(node_comm != i)[:,0]
        for start_idx in nodes_idx:
            for end_idx in other_nodes_idx:
                if (adj[start_idx, end_idx] == 1) and (start_idx < num_node) and (end_idx < num_node):   ## remove the padding
                    idx = start_idx * adj.shape[0] + end_idx
                   #other_idx = end_idx * adj.shape[0] + start_idx
                    noised_list.append(idx)
    adj_flatten = np.reshape(adj.todense(), [-1])
    adj_flatten = np.squeeze(np.asarray(adj_flatten))
    adj_all_indexes = np.argwhere(adj_flatten == 1)
    clean_list = set(adj_all_indexes[:,0]).difference(set(noised_list))
    return noised_list, list(clean_list)

def load_data_subgraphs(dataset_name, train_ratio):
    structure_input, diff, feature_input, ally, num_nodes_all = load(dataset_name)
    ## train test split
    indexes = np.arange(len(structure_input))
    np.random.shuffle(indexes)
    train_indexes = indexes[:int(train_ratio* len(structure_input))]
    test_indexes = indexes[int((train_ratio)*len(structure_input)):]
    train_structure_input = structure_input[train_indexes]
    train_feature_input = feature_input[train_indexes]
    train_y = ally[train_indexes]
    train_num_nodes_all = np.array(num_nodes_all)[train_indexes]
    test_structure_input =structure_input[test_indexes]
    test_feature_input = feature_input[test_indexes]
    test_y = ally[test_indexes]
    test_num_nodes_all = np.array(num_nodes_all)[test_indexes]

    return train_structure_input, train_feature_input, train_y, train_num_nodes_all, test_structure_input, test_feature_input, test_y, test_num_nodes_all


def PSNR(clean_adj, modified_adj):
    # mse = spnorm(clean_adj - modified_adj, ord = 1) / (clean_adj.shape[0]*(clean_adj.shape[1] - 1))
    # mse = max(spnorm(clean_adj - modified_adj, ord = 1),1.0) / (clean_adj.shape[0]*(clean_adj.shape[1] - 1))
    mse = max(spnorm(clean_adj - modified_adj, ord = 1),0.1) / (clean_adj.shape[0]*(clean_adj.shape[1] - 1))
    PSNR = 10 * np.log(1/ mse)
    return PSNR

def PSNR_with_features(clean_adj, clean_fea, modified_adj, modified_fea):
    mse = spnorm(clean_adj - modified_adj, ord = 1) / (clean_adj.shape[0]*(clean_adj.shape[1] - 1))
    mse = mse + sp.linalg.norm(clean_fea - modified_fea, ord = 1) / (clean_fea.shape[0]* modified_fea.shape[1])
    PSNR = 10 * np.log(1/ mse)
    return PSNR

def WL_no_label(clean_adj, modified_adj):
    G_clean = nx.from_scipy_sparse_matrix(clean_adj)
    G_modified = nx.from_scipy_sparse_matrix(modified_adj)
    WL_clean = compute_mle_wl_kernel([G_clean, G_modified], 20)
    print(WL_clean)
    return float(WL_clean[0,1]/(math.sqrt(WL_clean[0,0])* math.sqrt(WL_clean[1,1])))

def WL(clean_adj, modified_adj, y_label):
    G_clean = nx.from_scipy_sparse_matrix(clean_adj)
    G_modified = nx.from_scipy_sparse_matrix(modified_adj)
    #### set labels  #######
    label_dict = {}
    label_dict_modified = {}
    for idx, label in enumerate(y_label):
        lbl = np.argwhere(label == 1)
        if lbl.shape[0] < 1:
            lbl = 0
        else:
            lbl = int(np.squeeze(lbl))
        label_dict.update({idx:{"node_label":lbl}})
        label_dict_modified.update({idx:{"node_label":lbl}})
    nx.set_node_attributes(G_clean, label_dict)
    nx.set_node_attributes(G_modified, label_dict_modified)
  #####              ############
    WL_clean = compute_mle_wl_kernel([G_clean, G_modified], 20)

    print(WL_clean)
    return float(WL_clean[0,1]/(math.sqrt(WL_clean[0,0])* math.sqrt(WL_clean[1,1])))
if __name__ == "__main__":
    graph1 = nx.gnp_random_graph(75, 0.2)
    graph2 = nx.random_regular_graph(2, 75)
