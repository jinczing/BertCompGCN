import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import re
import csv
import dgl
import torch


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

    # training nodes are training docs, no initial features
    # print("x: ", x)
    # test nodes are training docs, no initial features
    # print("tx: ", tx)
    # both labeled and unlabeled training instances are training docs and words
    # print("allx: ", allx)
    # training labels are training doc labels
    # print("y: ", y)
    # test labels are test doc labels
    # print("ty: ", ty)
    # ally are labels for labels for allx, some will not have labels, i.e., all 0
    # print("ally: \n")
    # for i in ally:
    # if(sum(i) == 0):
    # print(i)
    # graph edge weight is the word co-occurence or doc word frequency
    # no need to build map, directly build csr_matrix
    # print('graph : ', graph)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    # print(len(labels))

    idx_test = test_idx_range.tolist()
    # print(idx_test)
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def load_multi_relations_doc(data_paths, logger):
    '''

    doc2doc
        * docs
            -Similarity
            -Keyword Inclusion
            -Named Entity Inclusion
            -Keyword TF-IDFs
            -Named Entity TF-IDFs
            -Ground Truth Relation
    '''
    data_dict = {}
    g = dgl.DGLGraph()
    word_features = []
    edge_type = []
    edge_weight = []
    

    bert_similarities = torch.load(data_paths['bert_similarities'])

    with open(data_paths['key_inclusions'], 'rb') as f:
        key_inclusions = pkl.load(f)
    with open(data_paths['ne_inclusions'], 'rb') as f:
        ne_inclusions = pkl.load(f)
    with open(data_paths['matching_table']) as f:
        matching_table = list(csv.reader(f))[1:]

    with open(data_paths['key_tf_idfs'], 'rb') as f:
        key_tf_idfs = pkl.load(f)
    with open(data_paths['ne_tf_idfs'], 'rb') as f:
        ne_tf_idfs = pkl.load(f)


    doc_num = key_inclusions.shape[0]
    edge_type_num = 0

    g.add_nodes(doc_num)
    doc_mask = [True]*doc_num
    test_num = 421
    test_mask = [False]*(g.num_nodes()-test_num) + [True]*test_num



    # doc to doc relations
    rfs = np.zeros((doc_num, doc_num, 4))

    # data_dict[('doc', 'sim', 'doc')] = []
    doc_title_to_ids = []
    for i, bert_similarity in enumerate(bert_similarities):
        doc_title_to_ids.append(bert_similarity[0])
        ids = np.where(bert_similarity[3]>0.5)[0]
        for id in ids:
            edge_type.append(edge_type_num)
            edge_weight.append(bert_similarity[3][id])
            rfs[i, bert_similarity[2][id], 0] = bert_similarity[3][id]
        g.add_edges(i, bert_similarity[2][ids])
    edge_type_num += 1

    # data_dict[('doc', 'key_inclusion', 'doc')] = []
    # ids = list(np.where(key_inclusions>0.5))
    # logger.info(f'key inclusions {ids[0].shape[0]}')
    # for i, j in zip(ids[0], ids[1]):
    #     if i == j: continue
    #     edge_type.append(7)
    #     edge_weight.append(key_inclusions[i][j])
    #     rfs[i, j, 1] = key_inclusions[i][j]
    # mask = np.where(ids[0]!=ids[1])[0]
    # ids[0] = ids[0][mask]
    # ids[1] = ids[1][mask]
    # g.add_edges(ids[0]+cum, ids[1]+cum)
    # print(g.num_edges(), len(edge_type))

    # data_dict[('doc', 'ne_inclusion', 'doc')] = []
    # ids = list(np.where(ne_inclusions>0.5))
    # logger.info(f'ne inclusions {ids[0].shape[0]}')
    # for i, j in zip(ids[0], ids[1]):
    #     if i == j: continue
    #     edge_type.append(8)
    #     edge_weight.append(ne_inclusions[i][j])
    #     rfs[i, j, 2] = ne_inclusions[i][j]
    # mask = np.where(ids[0]!=ids[1])[0]
    # ids[0] = ids[0][mask]
    # ids[1] = ids[1][mask]
    # g.add_edges(ids[0]+cum, ids[1]+cum)
    # print(g.num_edges(), len(edge_type))


    # data_dict[('doc', 'gt', 'doc')] = []
    src, dest = [], []
    for match in matching_table:
        src.append(doc_title_to_ids.index(match[0]))
        dest.append(doc_title_to_ids.index(match[1]))
        edge_type.append(edge_type_num)
        edge_weight.append(1)
    g.add_edges(src, dest)
    gt_edge_id = edge_type_num
    edge_type_num += 1
    

    key_tf_idfs/=np.linalg.norm(key_tf_idfs, axis=-1, keepdims=True)
    key_sim = np.matmul(key_tf_idfs, key_tf_idfs.transpose(1, 0)) # doc_num*doc_num

    ne_tf_idfs/=np.linalg.norm(ne_tf_idfs, axis=-1, keepdims=True)
    ne_sim = np.matmul(ne_tf_idfs, ne_tf_idfs.transpose(1, 0))

    key_num = 0
    for i in range(doc_num):
        ids = np.where(key_sim[i]>0.5)[0]
        key_num += ids.shape[0]
        for id in ids:
            if i == id: continue
            edge_type.append(edge_type_num)
            edge_weight.append(key_sim[i][id])
        ids = ids[ids!=i]
        g.add_edges(i, ids)
    edge_type_num += 1

    ne_num = 0
    for i in range(doc_num):
        ids = np.where(ne_sim[i]>0.5)[0]
        ne_num += ids.shape[0]
        for id in ids:
            if i == id: continue
            edge_type.append(edge_type_num)
            edge_weight.append(ne_sim[i][id])
        ids = ids[ids!=i]
        g.add_edges(i, ids)
    edge_type_num += 1

    print('key ne', key_num, ne_num)

    g.add_edges(g.edges()[1], g.edges()[0])
    edge_type += list(map(lambda x:x+edge_type_num, edge_type))
    edge_weight *= 2

    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    norm = in_deg ** -0.5
    norm[torch.isinf(norm)] = 0
    g.ndata['xxx'] = norm
    g.apply_edges(lambda edges: {'xxx': edges.dst['xxx'] * edges.src['xxx']})
    norm = g.edata.pop('xxx').squeeze()

    return g, edge_type, edge_weight, norm, np.array([0]), 0, doc_num, doc_mask, test_mask, rfs, edge_type_num, gt_edge_id

def load_multi_relations_corpus(data_paths, logger):
    '''
    word2word:
        * keywords
            -PIM
            -Similarity
        * named entities
            -PIM
            -Similarity
        * word segments
            -PIM
            -Similarity
    doc2doc
        * docs
            -Similarity
            -Keyword Inclusion
            -Named Entity Inclusion
            -Word Segment Inclusion
            -Ground Truth Relation
    word2doc
        * keywords
            -TF-IDF
        * named entities
            -TF-IDF
        * word segments
            -TF-IDF
    '''
    data_dict = {}
    g = dgl.DGLGraph()
    word_features = []
    edge_type = []
    edge_weight = []

    with open(data_paths['key_pims'], 'rb') as f:
        key_pims = pkl.load(f)
    with open(data_paths['key_sims'], 'rb') as f:
        key_sims = pkl.load(f)
    with open(data_paths['ne_pims'], 'rb') as f:
        ne_pims = pkl.load(f)
    with open(data_paths['ne_sims'], 'rb') as f:
        ne_sims = pkl.load(f)
    with open(data_paths['ws_pims'], 'rb') as f:
        ws_pims = pkl.load(f)
    with open(data_paths['ws_sims'], 'rb') as f:
        ws_sims = pkl.load(f)
    

    bert_similarities = torch.load(data_paths['bert_similarities'])
    # bert_features = torch.load(data_paths['bert_features'])

    with open(data_paths['key_inclusions'], 'rb') as f:
        key_inclusions = pkl.load(f)
    with open(data_paths['ne_inclusions'], 'rb') as f:
        ne_inclusions = pkl.load(f)
    with open(data_paths['ws_inclusions'], 'rb') as f:
        ws_inclusions = pkl.load(f)
    with open(data_paths['matching_table']) as f:
        matching_table = list(csv.reader(f))[1:]

    with open(data_paths['key_tf_idfs'], 'rb') as f:
        key_tf_idfs = pkl.load(f)
    with open(data_paths['ne_tf_idfs'], 'rb') as f:
        ne_tf_idfs = pkl.load(f)
    with open(data_paths['ws_tf_idfs'], 'rb') as f:
        ws_tf_idfs = pkl.load(f)


    key_num = key_pims.shape[0]
    ne_num = ne_pims.shape[0]
    ws_num = 0 # ws_pims.shape[0]
    doc_num = key_inclusions.shape[0]
    edge_type_num = 0

    g.add_nodes(key_num+ne_num+ws_num+doc_num)
    doc_mask = [False]*(key_num+ne_num+ws_num) + [True]*doc_num
    test_num = 421
    test_mask = [False]*(g.num_nodes()-test_num) + [True]*test_num

    # word features
    with open(data_paths['key_berts'], 'rb') as f:
        key_berts = pkl.load(f)
        word_features.append(key_berts)
    with open(data_paths['ne_berts'], 'rb') as f:
        ne_berts = pkl.load(f)
        word_features.append(ne_berts)
    # with open(data_paths['ws_berts'], 'rb') as f:
    #     ws_berts = pkl.load(f)
    #     word_features.append(ws_berts)


    # word to word 
    # data_dict[('key', 'pim', 'key')] = []
    ids = list(np.where(key_pims>5))
    for i, j in zip(ids[0], ids[1]):
        if i == j: continue
        edge_type.append(edge_type_num)
        edge_weight.append(key_pims[i][j])
    mask = np.where(ids[0]!=ids[1])[0]
    ids[0] = ids[0][mask]
    ids[1] = ids[1][mask]
    g.add_edges(ids[0], ids[1])
    edge_type_num += 1

    # data_dict[('key', 'sim', 'key')] = []
    ids = list(np.where(key_sims>0.7))
    for i, j in zip(ids[0], ids[1]):
        if i == j: continue
        edge_type.append(edge_type_num)
        edge_weight.append(key_sims[i][j])
    mask = np.where(ids[0]!=ids[1])[0]
    ids[0] = ids[0][mask]
    ids[1] = ids[1][mask]
    g.add_edges(ids[0], ids[1])
    edge_type_num += 1

    # data_dict[('ne', 'pim', 'ne')] = []
    cum = key_num
    ids = list(np.where(ne_pims>5))
    for i, j in zip(ids[0], ids[1]):
        if i == j: continue
        edge_type.append(edge_type_num)
        edge_weight.append(ne_pims[i][j])
    mask = np.where(ids[0]!=ids[1])[0]
    ids[0] = ids[0][mask]
    ids[1] = ids[1][mask]
    g.add_edges(ids[0]+cum, ids[1]+cum)
    edge_type_num += 1

    # data_dict[('ne', 'sim', 'ne')] = []
    ids = list(np.where(ne_sims>0.7))
    for i, j in zip(ids[0], ids[1]):
        if i == j: continue
        edge_type.append(edge_type_num)
        edge_weight.append(ne_sims[i][j])
    mask = np.where(ids[0]!=ids[1])[0]
    ids[0] = ids[0][mask]
    ids[1] = ids[1][mask]
    g.add_edges(ids[0]+cum, ids[1]+cum)
    edge_type_num += 1



    # data_dict[('ws', 'pim', 'ws')] = []
    cum += ne_num
    # ids = np.where(ws_pims>0.5)
    # logger.info(f'ws pims {ids[0].shape[0]}')
    # logger.info(f'info word to word ws len {ids[0].shape[0]}')
    # for i, j in zip(ids[0], ids[1]):
    #     if i == j: continue
    #     edge_type.append(4)
    #     edge_weight.append(ws_pims[i][j])
    # g.add_edges(ids[0], ids[1])

    # # data_dict[('ws', 'sim', 'ws')] = []
    # ids = np.where(ws_sims>0.5)
    # logger.info(f'ws sims {ids[0].shape[0]}')
    # logger.info(f'info word to word ws len {ids[0].shape[0]}')
    # for i, j in zip(ids[0], ids[1]):
    #     if i == j: continue
    #     edge_type.append(5)
    #     edge_weight.append(ws_sims[i][j])
    # g.add_edges(ids[0], ids[1])
    # logger.info('info word to word')



    # doc to doc relations
    rfs = np.zeros((doc_num, doc_num, 4))

    cum += ws_num
    # data_dict[('doc', 'sim', 'doc')] = []
    doc_title_to_ids = []
    for i, bert_similarity in enumerate(bert_similarities):
        doc_title_to_ids.append(bert_similarity[0])
        ids = np.where(bert_similarity[3]>0.7)[0]
        for id in ids:
            edge_type.append(edge_type_num)
            edge_weight.append(bert_similarity[3][id])
            rfs[i, bert_similarity[2][id], 0] = bert_similarity[3][id]
        g.add_edges(i+cum, bert_similarity[2][ids]+cum)
    edge_type_num += 1

    # data_dict[('doc', 'key_inclusion', 'doc')] = []
    # ids = list(np.where(key_inclusions>0.5))
    # logger.info(f'key inclusions {ids[0].shape[0]}')
    # for i, j in zip(ids[0], ids[1]):
    #     if i == j: continue
    #     edge_type.append(7)
    #     edge_weight.append(key_inclusions[i][j])
    #     rfs[i, j, 1] = key_inclusions[i][j]
    # mask = np.where(ids[0]!=ids[1])[0]
    # ids[0] = ids[0][mask]
    # ids[1] = ids[1][mask]
    # g.add_edges(ids[0]+cum, ids[1]+cum)
    # print(g.num_edges(), len(edge_type))

    # data_dict[('doc', 'ne_inclusion', 'doc')] = []
    # ids = list(np.where(ne_inclusions>0.5))
    # logger.info(f'ne inclusions {ids[0].shape[0]}')
    # for i, j in zip(ids[0], ids[1]):
    #     if i == j: continue
    #     edge_type.append(8)
    #     edge_weight.append(ne_inclusions[i][j])
    #     rfs[i, j, 2] = ne_inclusions[i][j]
    # mask = np.where(ids[0]!=ids[1])[0]
    # ids[0] = ids[0][mask]
    # ids[1] = ids[1][mask]
    # g.add_edges(ids[0]+cum, ids[1]+cum)
    # print(g.num_edges(), len(edge_type))

    # data_dict[('doc', 'ws_inclusion', 'doc')] = []
    # ids = np.where(ws_inclusions>0.5)
    # logger.info(f'ws inclusions {ids[0].shape[0]}')
    # for i, j in zip(ids[0], ids[1]):
    #     if i == j: continue
    #     edge_type.append(9)
    #     edge_weight.append(ws_inclusions[i][j])
    #     rfs[i, j, 3] = ws_inclusions[i][j]
    # g.add_edges(ids[0]+cum, ids[1]+cum)

    # data_dict[('doc', 'gt', 'doc')] = []
    src, dest = [], []
    for match in matching_table:
        src.append(doc_title_to_ids.index(match[0])+cum)
        dest.append(doc_title_to_ids.index(match[1])+cum)
        edge_type.append(edge_type_num)
        edge_weight.append(1)
    g.add_edges(src, dest)
    gt_edge_id = edge_type_num
    edge_type_num += 1
    


    # doc to word relations
    # data_dict[('doc', 'key_tf_idf', 'key')] = []
    # data_dict[('key', 'key_tf_idf', 'doc')] = []
    cum2 = 0
    ids = np.where(key_tf_idfs>0)
    for i, j in zip(ids[0], ids[1]):
        edge_type.append(edge_type_num)
        edge_weight.append(key_tf_idfs[i][j])
    g.add_edges(ids[0]+cum, ids[1]+cum2)
    edge_type_num += 1

    # data_dict[('doc', 'ne_tf_idf', 'ne')] = []
    # data_dict[('ne', 'ne_tf_idf', 'doc')] = []
    cum2 += key_num
    ids = np.where(ne_tf_idfs>0)
    for i, j in zip(ids[0], ids[1]):
        edge_type.append(edge_type_num)
        edge_weight.append(ne_tf_idfs[i][j])
    g.add_edges(ids[0]+cum, ids[1]+cum2)
    edge_type_num += 1

    # data_dict[('doc', 'ws_tf_idf', 'word')] = []
    # data_dict[('word', 'ws_tf_idf', 'doc')] = []
    # cum2 += ne_num
    # ids = np.where(ws_tf_idfs>0)
    # logger.info(f'ws tf idfs {ids[0].shape[0]}')
    # for i, j in zip(ids[0], ids[1]):
    #     edge_type.append(13)
    #     edge_weight.append(ws_tf_idfs[i][j])
    # g.add_edges(ids[0]+cum, ids[1]+cum2)


    cum += doc_num
    g.add_edges(g.edges()[1], g.edges()[0])
    word_features = np.concatenate(word_features, axis=0)
    edge_type += list(map(lambda x:x+edge_type_num, edge_type))
    edge_weight *= 2

    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    norm = in_deg ** -0.5
    norm[torch.isinf(norm)] = 0
    g.ndata['xxx'] = norm
    g.apply_edges(lambda edges: {'xxx': edges.dst['xxx'] * edges.src['xxx']})
    norm = g.edata.pop('xxx').squeeze()

    return g, edge_type, edge_weight, norm, word_features, key_num+ne_num+ws_num, doc_num, doc_mask, test_mask, rfs, edge_type_num, gt_edge_id

def load_corpus(dataset_str):
    """
    Loads input corpus from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training docs/words
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training docs as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test docs as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.adj => adjacency matrix of word/doc nodes as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.train.index => the indices of training docs in original doc list.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'adj']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, adj = tuple(objects)
    print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

    features = sp.vstack((allx, tx)).tolil()
    labels = np.vstack((ally, ty))
    print(len(labels))

    train_idx_orig = parse_index_file(
        "data/{}.train.index".format(dataset_str))
    train_size = len(train_idx_orig)

    val_size = train_size - x.shape[0]
    test_size = tx.shape[0]

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + val_size)
    idx_test = range(allx.shape[0], allx.shape[0] + test_size)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i]
                      for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (
        2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def loadWord2Vec(filename):
    """Read Word Vectors"""
    vocab = []
    embd = []
    word_vector_map = {}
    file = open(filename, 'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        if(len(row) > 2):
            vocab.append(row[0])
            vector = row[1:]
            length = len(vector)
            for i in range(length):
                vector[i] = float(vector[i])
            embd.append(vector)
            word_vector_map[row[0]] = vector
    print('Loaded Word Vectors!')
    file.close()
    return vocab, embd, word_vector_map

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()