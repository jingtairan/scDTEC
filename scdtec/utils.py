#!/usr/bin/env python

import numpy as np
import pandas as pd
import scipy
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, scale
from sklearn.metrics import classification_report, confusion_matrix, adjusted_rand_score, normalized_mutual_info_score

import scipy.sparse as sp
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
import dgl
from sklearn import metrics
from munkres import Munkres

EOS = 1e-10

# ============== Data Processing ==============
# =============================================

def read_labels(ref, return_enc=False):
    """
    Read labels and encode to 0, 1 .. k with class names 
    """
    # if isinstance(ref, str):
    ref = pd.read_csv(ref, sep='\t', index_col=0, header=None)

    encode = LabelEncoder()
    ref = encode.fit_transform(ref.values.squeeze())
    classes = encode.classes_
    if return_enc:
        return ref, classes, encode
    else:
        return ref, classes

def gene_filter_(data, X=6):
    """
    Gene filter in SC3:
        Removes genes/transcripts that are either (1) or (2)
        (1). rare genes/transcripts:
            Expressed (expression value > 2) in less than X% of cells 
        (2). ubiquitous genes/transcripts:
            Expressed (expression value > 0) in at least (100 - X)% of cells
    Input data:
        data is an array of shape (p,n)
    """
    total_cells = data.shape[1]
    count_1 = data[data > 1].count(axis=1)
    count_2 = data[data > 0].count(axis=1)

    genelist_1 = count_1[count_1 > 0.01*X * total_cells].index
    genelist_2 = count_2[count_2 < 0.01*(100-X) * total_cells].index
    genelist = set(genelist_1) & set(genelist_2)
    data = data.loc[genelist]
    return data

def sort_by_mad(data, axis=0):
    """
    Sort genes by mad to select input features
    """
    genes = data.mad(axis=axis).sort_values(ascending=False).index
    if axis==0:
        data = data.loc[:, genes]
    else:
        data = data.loc[genes]
    return data


# =========== scATAC Preprocessing =============
# ==============================================
def peak_filter(data, x=10, n_reads=2):
    count = data[data >= n_reads].count(1)
    index = count[count >= x].index
    data = data.loc[index]
    return data

def cell_filter(data):
    thresh = data.shape[0]/50
    # thresh = min(min_peaks, data.shape[0]/50)
    data = data.loc[:, data.sum(0) > thresh]
    return data

def sample_filter(data, x=10, n_reads=2):
    data = peak_filter(data, x, n_reads)
    data = cell_filter(data)
    return data

# =================== Other ====================
# ==============================================

from scipy.sparse.linalg import eigsh
def estimate_k(data):
    """
    Estimate number of groups k:
        based on random matrix theory (RTM), borrowed from SC3
        input data is (p,n) matrix, p is feature, n is sample
    """
    p, n = data.shape

    x = scale(data, with_mean=False)
    muTW = (np.sqrt(n-1) + np.sqrt(p)) ** 2
    sigmaTW = (np.sqrt(n-1) + np.sqrt(p)) * (1/np.sqrt(n-1) + 1/np.sqrt(p)) ** (1/3)
    sigmaHatNaive = x.T.dot(x)

    bd = np.sqrt(p) * sigmaTW + muTW
    evals, _ = eigsh(sigmaHatNaive)

    k = 0
    for i in range(len(evals)):
        if evals[i] > bd:
            k += 1
    return k

# def estimate_k(data):
#     return min(data.shape[0]/100, 30)

def get_decoder_weight(model_file):
    state_dict = torch.load(model_file)
    weight = state_dict['decoder.reconstruction.weight'].data.cpu().numpy()
    return weight

def peak_selection(weight, weight_index, kind='both', cutoff=2.5):
    """
    Select represented peaks of each components of each peaks, 
    correlations between peaks and features are quantified by decoder weight,
    weight is a Gaussian distribution, 
    filter peaks with weights more than cutoff=2.5 standard deviations from the mean.

    Input:
        weight: weight of decoder
        weight_index: generated peak/gene index. 
        kind: both for both side, pos for positive side and neg for negative side.
        cutoff: cutoff of standard deviations from mean.
    """
    std = weight.std(0)
    mean = weight.mean(0)
    specific_peaks = []
    for i in range(10):
        w = weight[:,i]
        if kind == 'both':
            index = np.where(np.abs(w-mean[i]) > cutoff*std[i])[0]
        if kind == 'pos':
            index = np.where(w-mean[i] > cutoff*std[i])[0]
        if kind == 'neg':
            index = np.where(mean[i]-w > cutoff*std[i])[0]
        specific_peaks.append(weight_index[index])
    return specific_peaks
    

def pairwise_pearson(A, B):
    from scipy.stats import pearsonr
    corrs = []
    for i in range(A.shape[0]):
        if A.shape == B.shape:
            corr = pearsonr(A.iloc[i], B.iloc[i])[0]
        else:
            corr = pearsonr(A.iloc[i], B)[0]
        corrs.append(corr)
    return corrs

# ================= Metrics ===================
# =============================================

def reassign_cluster_with_ref(Y_pred, Y):
    """
    Reassign cluster to reference labels
    Inputs:
        Y_pred: predict y classes
        Y: true y classes
    Return:
        f1_score: clustering f1 score
        y_pred: reassignment index predict y classes
        indices: classes assignment
    """
    def reassign_cluster(y_pred, index):
        y_ = np.zeros_like(y_pred)
        for i, j in index:
            y_[np.where(y_pred==i)] = j
        return y_
    from aaaaaa import linear_assignment
#     print(Y_pred.size, Y.size)
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)

    return reassign_cluster(Y_pred, ind)


def cluster_report(ref, pred, classes=None):
    """
    Print Cluster Report
    """
    pred = reassign_cluster_with_ref(pred, ref)
    cm = confusion_matrix(ref, pred)
    #print('\n## Confusion matrix ##\n')
    #print(cm)
    #print('\n## Cluster Report ##')
#     print(classification_report(ref, pred, target_names=classes))
    #print("Adjusted Rand Index score: {:.4f}".format(adjusted_rand_score(ref, pred)))
    #print("`Normalized Mutual Info score: {:.4f}".format(normalized_mutual_info_score(ref, pred)))
    return cm, adjusted_rand_score(ref, pred), normalized_mutual_info_score(ref, pred)

def binarization(imputed, raw):
    return scipy.sparse.csr_matrix((imputed.T > raw.mean(1).T).T & (imputed>raw.mean(0))).astype(np.int8)


def torch_sparse_eye(num_nodes):
    indices = torch.arange(num_nodes).repeat(2, 1)
    values = torch.ones(num_nodes)
    return torch.sparse.FloatTensor(indices, values)


def normalize(adj, mode, sparse=False):
    if not sparse:
        if mode == "sym":
            inv_sqrt_degree = 1./ (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
        elif mode == "row":
            inv_degree = 1./ (adj.sum(dim=1, keepdim=False) + EOS)
            return inv_degree[:, None] * adj
        else:
            exit("wrong norm mode")
    else:
        adj = adj.coalesce()
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()))
            D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]

        elif mode == "row":
            aa = torch.sparse.sum(adj, dim=1)
            bb = aa.values()
            inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EOS)
            D_value = inv_degree[adj.indices()[0]]
        else:
            exit("wrong norm mode")
        new_values = adj.values() * D_value

        return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size())


def torch_sparse_to_dgl_graph(torch_sparse_mx):
    torch_sparse_mx = torch_sparse_mx.coalesce()
    indices = torch_sparse_mx.indices()
    values = torch_sparse_mx.values()
    rows_, cols_ = indices[0,:], indices[1,:]
    dgl_graph = dgl.graph((rows_, cols_), num_nodes=torch_sparse_mx.shape[0], device='cuda')
    dgl_graph.edata['w'] = values.detach().cuda()
    return dgl_graph


def nearest_neighbors_pre_elu(X, k, metric, i):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    adj = adj * i - i
    return adj


def knn_fast(X, k, b):
    X = F.normalize(X, dim=1, p=2)
    index = 0
    values = torch.zeros(X.shape[0] * (k + 1)).cuda()
    rows = torch.zeros(X.shape[0] * (k + 1)).cuda()
    cols = torch.zeros(X.shape[0] * (k + 1)).cuda()
    norm_row = torch.zeros(X.shape[0]).cuda()
    norm_col = torch.zeros(X.shape[0]).cuda()
    while index < X.shape[0]:
        if (index + b) > (X.shape[0]):
            end = X.shape[0]
        else:
            end = index + b
        sub_tensor = X[index:index + b]
        similarities = torch.mm(sub_tensor, X.t())
        vals, inds = similarities.topk(k=k + 1, dim=-1)
        values[index * (k + 1):(end) * (k + 1)] = vals.view(-1)
        cols[index * (k + 1):(end) * (k + 1)] = inds.view(-1)
        rows[index * (k + 1):(end) * (k + 1)] = torch.arange(index, end).view(-1, 1).reshape(1, k + 1).view(-1)
        norm_row[index: end] = torch.sum(vals, dim=1)
        norm_col.index_add_(-1, inds.view(-1), vals.view(-1))
        index += b
    norm = norm_row + norm_col
    rows = rows.long()
    cols = cols.long()
    values *= (torch.pow(norm[rows], -0.5) * torch.pow(norm[cols], -0.5))
    return rows, cols, values


def apply_non_linearity(tensor, non_linearity, i):
    if non_linearity == 'elu':
        return F.elu(tensor * i - i) + 1
    elif non_linearity == 'relu':
        return F.relu(tensor)
    elif non_linearity == 'none':
        return tensor
    else:
        raise NameError('We dont support thr non-linearity yet')


def cal_similarity_graph(node_embeddings):
    similarity_graph = torch.mm(node_embeddings, node_embeddings.t())
    return similarity_graph


def top_k(raw_graph, K):
    values, indices = raw_graph.topk(k=int(K), dim=-1)
    assert torch.max(indices) < raw_graph.shape[1]
    mask = torch.zeros(raw_graph.shape).cuda()
    mask[torch.arange(raw_graph.shape[0]).view(-1, 1), indices] = 1.

    mask.requires_gard = False
    sparse_graph = raw_graph * mask
    return sparse_graph


def get_feat_mask(features, mask_rate):
    feat_node = features.shape[1]
    mask = torch.zeros(features.shape)
    samples = np.random.choice(feat_node, size=int(feat_node * mask_rate), replace=False)
    mask[:, samples] = 1
    return mask.cuda(), samples


def symmetrize(adj):  # only for non-sparse
    return (adj + adj.T) / 2


def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list


def dgl_graph_to_torch_sparse(dgl_graph):
    values = dgl_graph.edata['w'].cpu().detach()
    rows_, cols_ = dgl_graph.edges()
    indices = torch.cat((torch.unsqueeze(rows_, 0), torch.unsqueeze(cols_, 0)), 0).cpu()
    torch_sparse_mx = torch.sparse.FloatTensor(indices, values)
    return torch_sparse_mx
