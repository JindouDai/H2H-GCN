"""Data utils functions for pre-processing and data loading."""
import os
import pickle as pkl
import sys
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from utils.pre_utils import *

def convert_hgnn_adj(adj):
    hgnn_adj = [[i] for i in range(adj.shape[0])]
    hgnn_weight = [[1] for i in range(adj.shape[0])]
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i,j] == 1:
                hgnn_adj[i].append(j)
                hgnn_weight[i].append(1)

    max_len = max([len(i) for i in hgnn_adj])
    normalize_weight(hgnn_adj, hgnn_weight)
 
    hgnn_adj = pad_sequence(hgnn_adj, max_len)
    hgnn_weight = pad_sequence(hgnn_weight, max_len)
    hgnn_adj = np.array(hgnn_adj)
    hgnn_weight = np.array(hgnn_weight)
    return torch.from_numpy(hgnn_adj).cuda(), torch.from_numpy(hgnn_weight).cuda().float()


def load_data(args, datapath):
    if args.task == 'nc':
        data = load_data_nc(args, args.dataset, args.use_feats, datapath, args.split_seed)

    else:
        data = load_data_lp(args, args.dataset, args.use_feats, datapath)
        adj = data['adj_train']
        if args.task == 'lp':
            adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false, hgnn_adj, hgnn_weight = mask_edges(
                    adj, args.val_prop, args.test_prop, args.split_seed
            )
            data['adj_train'] = adj_train
            data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false 
            data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
            data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false
            data['hgnn_adj'] = hgnn_adj
            data['hgnn_weight'] = hgnn_weight
    data['adj_train_norm'], data['features'] = process(
            data['adj_train'], data['features'], args.normalize_adj, args.normalize_feats
    )
    return data

# ############### FEATURES PROCESSING ####################################

def process(adj, features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features

def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def augment(adj, features, normalize_feats=True):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features

# ############### DATA SPLITS #####################################################

def mask_edges(adj, val_prop, test_prop, seed):
    np.random.seed(seed)  
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y))) 
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero() 
    neg_edges = np.array(list(zip(x, y))) 
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges) 
    n_val = int(m_pos * val_prop) 
    n_test = int(m_pos * test_prop) 
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]     
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    tmp_a, tmp_b = adj_train.nonzero()
    hgnn_adj = [[i] for i in range(adj.shape[0])]
    hgnn_weight = [[1] for i in range(adj.shape[0])]
    indptr_tmp = adj_train.indptr
    indices_tmp = adj_train.indices
    data_tmp = adj_train.data
    flag = 0
    for i in range(len(indptr_tmp)-1):
        items = indptr_tmp[i+1] - indptr_tmp[i]
        for j in range(items):
            hgnn_adj[i].append(indices_tmp[flag])
            hgnn_weight[i].append(1)
            flag += 1

    max_len = max([len(i) for i in hgnn_adj])
    normalize_weight(hgnn_adj, hgnn_weight)
    hgnn_adj = pad_sequence(hgnn_adj, max_len)
    hgnn_weight = pad_sequence(hgnn_weight, max_len)
    hgnn_adj = np.array(hgnn_adj)
    hgnn_weight = np.array(hgnn_weight)

    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
            test_edges_false), torch.from_numpy(hgnn_adj).cuda(), torch.from_numpy(hgnn_weight).cuda().float()

def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]
    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg

def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()

# ############### LINK PREDICTION DATA LOADERS ####################################

def load_data_lp(args, dataset, use_feats, data_path):
    if dataset in ['disease_lp']:
        adj, features = load_synthetic_data(dataset, use_feats, data_path)[:2]
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
    data = {'adj_train': adj, 'features': features, }
    return data

# ############### NODE CLASSIFICATION DATA LOADERS ####################################

def load_data_nc(args, dataset, use_feats, data_path, split_seed):
    if dataset in ['disease_nc']:
        adj, features, labels = load_synthetic_data(dataset, use_feats, data_path)
        val_prop, test_prop = 0.10, 0.60
        hgnn_adj, hgnn_weight = convert_hgnn_adj(adj.todense())
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
    idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)

    labels = torch.LongTensor(labels)

    data = {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test, 'hgnn_adj': hgnn_adj, 'hgnn_weight': hgnn_weight}
    return data

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_synthetic_data(dataset_str, use_feats, data_path):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0])
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    return sp.csr_matrix(adj), features, labels

