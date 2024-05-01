"""
This code was adapted from https://github.com/lucamasera/AWX
"""
import os
import numpy as np
import networkx as nx
import keras
from itertools import chain
# import tensorflow
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Skip the root nodes 
to_skip = ['root', 'GO0003674', 'GO0005575', 'GO0008150']

def read_edge(edge_path):
    edges = []
    edge_file = os.path.join(edge_path)
    with open(edge_file, 'r') as f:
        for line in f.readlines():
            des, src, _ = line.split(' ')
            edges.append((src, des))
    return edges

class arff_data(): # For preparation
    def __init__(self, arff_file, tc_edge_path, is_GO):
        X, Y, A, terms, g, tc_edges = parse_arff(arff_file=arff_file, tc_edge_path=tc_edge_path, is_GO=is_GO)
        self.images = X
        # Preprocess NaN
        r_, c_ = np.where(np.isnan(X))
        m = np.nanmean(X, axis=0)
        for i, j in zip(r_, c_):
            self.images[i,j] = m[j]

        to_use = [t not in to_skip for t in terms]
        self.class_names = [t for t in terms if t not in to_skip]
        self.labels = Y[:, to_use]
        self.adj_matrix = A[to_use, :][:, to_use]

        self.nodes = self.class_names
        self.edges = []
        self.tc_edges = []

        for src, des in g.edges:
            if src in self.class_names and des in self.class_names:
                self.edges.append((src, des))
        for src, des in tc_edges:
            if src in self.class_names and des in self.class_names:
                self.tc_edges.append((src, des))

        # print(X.shape, Y.shape, A.shape)
        # print(self.images.shape, self.labels.shape, self.adj_matrix.shape)
        # print(len(g.nodes), len(g.edges), len(tc_edges))
        # print(len(self.nodes), len(self.edges), len(self.tc_edges))
        # print("===================================================================")

    def __add__(self, others):
        self.images = np.concatenate((self.images, others.images),0)
        self.labels = np.concatenate((self.labels, others.labels),0)

class arff_data2():
    def __init__(self, arff_file, edge_path, tc_edge_path):
        X, Y, A, terms, edges, tc_edges = parse_arff2(arff_file, edge_path, tc_edge_path)
        self.images = X
        # Preprocess NaN
        r_, c_ = np.where(np.isnan(X))
        m = np.nanmean(X, axis=0)
        for i, j in zip(r_, c_):
            self.images[i,j] = m[j]

        to_use = [t not in to_skip for t in terms]
        self.class_names = [t for t in terms if t not in to_skip]
        idx = {cls:i for i, cls in enumerate(self.class_names)}
        self.class_idx = idx
        self.labels = Y[:, to_use]
        self.adj_matrix = A[to_use, :][:, to_use]

        self.nodes = self.class_names
        self.edges = []
        self.tc_edges = []

        for src, des in edges:
            if src in self.class_names and des in self.class_names:
                self.edges.append((idx[src], idx[des]))
        for src, des in tc_edges:

            if src in self.class_names and des in self.class_names:
                self.tc_edges.append((idx[src], idx[des]))

        # import util
        # util.SAVE(self.tc_edges, 'saved_multi/edge/true_edge_imclef07d')
        # tc = util.LOAD('saved_multi/edge/true_edge_imclef07d')
        # print("a")
        # print(self.class_names)
        # print(self.images.shape, self.labels.shape)

    def __add__(self, others):
        self.images = np.concatenate((self.images, others.images),0)
        self.labels = np.concatenate((self.labels, others.labels),0)

def parse_arff(arff_file, tc_edge_path, is_GO=False):
    with open(arff_file, encoding = "utf-8") as f:
        read_data = False
        X = []
        Y = []
        g = nx.DiGraph()
        feature_types = []
        d = []
        cats_lens = []
        for num_line, l in enumerate(f):
            if l.startswith('@ATTRIBUTE'):
                if l.startswith('@ATTRIBUTE class'):
                    h = l.split('hierarchical')[1].strip()
                    for branch in h.split(','):
                        terms = branch.split('/')
                        if is_GO:
                            g.add_edge(terms[1], terms[0])
                        else:
                            if len(terms)==1 or 'root' in terms:
                                g.add_edge(terms[0], 'root')
                            else:
                                for i in range(2, len(terms) + 1):
                                    g.add_edge('.'.join(terms[:i]), '.'.join(terms[:i-1]))
                    nodes = sorted(g.nodes(), key=lambda x: (nx.shortest_path_length(g, x, 'root'), x) if is_GO else (len(x.split('.')),x))
                    nodes_idx = dict(zip(nodes, range(len(nodes))))
                    g_t = g.reverse()

                else:
                    try:
                        _, f_name, f_type = l.split()
                    except:
                        _, f_name, f_type_1, f_type_2 = l.split()
                        f_type = ''.join([f_type_1, f_type_2])
                        # print(f_type)
                    
                    if f_type == 'numeric' or f_type == 'NUMERIC':
                        d.append([])
                        cats_lens.append(1)
                        feature_types.append(lambda x,i: [float(x)] if x != '?' else [np.nan])
                        
                    else:
                        cats = f_type[1:-1].split(',')
                        cats_lens.append(len(cats))
                        d.append({key:keras.utils.to_categorical(i, len(cats)).tolist() for i,key in enumerate(cats)})
                        feature_types.append(lambda x,i: d[i].get(x, [0.0]*cats_lens[i]))
            elif l.startswith('@DATA'):
                read_data = True
            elif read_data:
                y_ = np.zeros(len(nodes))
                d_line = l.split('%')[0].strip().split(',')
                lab = d_line[len(feature_types)].strip()
                X.append(list(chain(*[feature_types[i](x,i) for i, x in enumerate(d_line[:len(feature_types)])])))

                for t in lab.split('@'):
                    # print(lab.split('@')) #
                    y_[[nodes_idx.get(a) for a in nx.ancestors(g_t, t.replace('/', '.'))]] =1
                    y_[nodes_idx[t.replace('/', '.')]] = 1
                Y.append(y_)
        X = np.array(X)
        Y = np.stack(Y)

    tc_edges = read_edge(tc_edge_path)
    return X, Y, np.array(nx.to_numpy_matrix(g, nodelist=nodes)), nodes, g, tc_edges


def parse_arff2(arff_file, edge_path, tc_edge_path):
    with open(arff_file, encoding="utf-8") as f:
        read_data = False
        X = []
        Y = []
        input_feature_types = []
        class_types = []

        for num_line, l in enumerate(f):
            if l.startswith('@ATTRIBUTE'):
                try:
                    _, f_name, f_type = l.split()  # @ATTRIBUTE Attr85 NUMERIC
                except:
                    _, f_name, f_type_1, f_type_2 = l.split()  # @ATTRIBUTE GO0003674 {0, 1}
                    f_type = ''.join([f_type_1, f_type_2])

                if f_type == 'numeric' or f_type == 'NUMERIC':
                    input_feature_types.append(f_name)
                else:
                    class_types.append(f_name)

            elif l.startswith('@DATA'):
                read_data = True
            elif read_data:
                input_dims = len(input_feature_types)
                num_classes = len(class_types)

                d_line = l.strip().split(',')
                x_ = np.array([float(x) for x in d_line[:input_dims]])
                y_ = np.array([float(y) for y in d_line[input_dims:]])
                assert len(y_) == num_classes

                X.append(x_)
                Y.append(y_)

        X = np.array(X)
        Y = np.stack(Y)

    edges = read_edge(edge_path)
    tc_edges = read_edge(tc_edge_path)

    g = nx.DiGraph()
    g.add_edges_from(tc_edges)
    # nodes = sorted(g.nodes(), key=lambda x: (nx.shortest_path_length(g, x, 'root'), x) if is_GO)
    nodes = sorted(g.nodes(), key=lambda x: (len(x.split('.')), x))
    nodes_idx = dict(zip(nodes, range(len(nodes))))
    # print(class_types)
    # print(nodes == class_types)
    # exit()
    return X, Y, np.array(nx.to_numpy_matrix(g, nodelist=nodes)), class_types, edges, tc_edges



# import datasets

# def initialize_dataset(name):
def initialize_dataset(name):
    is_GO = name[-2:] == 'GO'
    is_OTHERS = name[-2:] != 'GO' and name[-3:] != 'FUN'
    tc_edge_path = 'data/HMC_data/{}/hierarchy_tc.edgelist'.format(name)
    train_path = './data/HMC_data/{}/{}.train.arff'.format(name, name)
    test_path = './data/HMC_data/{}/{}.test.arff'.format(name, name)
    # return arff_data(train, is_GO), arff_data(val, is_GO), arff_data(test, is_GO)
    if is_OTHERS:
        return arff_data(train_path, tc_edge_path, is_GO), [], arff_data(test_path, tc_edge_path, is_GO)
    val_path = './data/HMC_data/{}/{}.valid.arff'.format(name, name)
    return arff_data(train_path, tc_edge_path, is_GO), arff_data(val_path, tc_edge_path, is_GO), arff_data(test_path, tc_edge_path, is_GO)


def initialize_other_dataset(name, datasets):
    is_GO, train, test = datasets[name]
    return arff_data(train, is_GO), arff_data(test, is_GO)



# # HMC dataset genraete npy file
# dataset_name = 'imclef07a'
# meta= {
# 'path_to_train': 'data/HMC_data/{}/train-normalized.arff'.format(dataset_name, dataset_name),
# 'path_to_val': 'data/HMC_data/{}/dev-normalized.arff'.format(dataset_name, dataset_name),
# 'path_to_test': 'data/HMC_data/{}/test-normalized.arff'.format(dataset_name, dataset_name),
# 'path_to_edges': 'data/HMC_data/{}/hierarchy.edgelist'.format(dataset_name),
# 'path_to_tc_edges': 'data/HMC_data/{}/hierarchy_tc.edgelist'.format(dataset_name)}
#
# train = arff_data2(meta['path_to_train'], meta['path_to_edges'], meta['path_to_tc_edges'])
# val = arff_data2(meta['path_to_val'], meta['path_to_edges'], meta['path_to_tc_edges'])
# test = arff_data2(meta['path_to_test'], meta['path_to_edges'], meta['path_to_tc_edges'])
# print(train.images[0])
# print(train.images.shape, train.labels.shape)
# train+val
# print(train.images.shape)
# np.save(os.path.join('data/HMC_data/{}'.format(dataset_name), 'formatted_train_images.npy'), train.images)
# np.save(os.path.join('data/HMC_data/{}'.format(dataset_name), 'formatted_train_labels.npy'), train.labels)
# print(test.images.shape, test.labels.shape)
# np.save(os.path.join('data/HMC_data/{}'.format(dataset_name), 'formatted_val_images.npy'), test.images)
# np.save(os.path.join('data/HMC_data/{}'.format(dataset_name), 'formatted_val_labels.npy'), test.labels)
