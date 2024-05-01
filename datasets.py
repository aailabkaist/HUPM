import os
import numpy as np
from PIL import Image
import torch
import copy
from torch.utils.data import Dataset
from torchvision import transforms
import config
import itertools
from util import *

def get_metadata(P):
    dataset_name = P['dataset']
    image_path = P['image_path']
    if dataset_name == 'cifar':
        meta = {
            'path_to_dataset': 'data/cifar',
            'path_to_images': '',
        }
    elif dataset_name == 'cars':
        meta = {
            'path_to_dataset': 'data/cars',
            'path_to_images': os.path.join(image_path,'STANFORD-CARS/')
        }
    elif dataset_name == 'cub2':
        meta = {
            'path_to_dataset': 'data/cub2',
            'path_to_images': os.path.join(image_path, 'CUB_200_2011/images')

        }
    elif dataset_name in ['imclef07a', 'imclef07d', 'cellcycle_FUN', 'derisi_FUN','expr_FUN', 'spo_FUN', \
                          'derisi_GO', 'spo_GO', 'cellcycle_GO', 'diatoms', 'enron_corr', 'expr_GO']:
        meta = {
            'path_to_dataset': 'data/HMC_data/{}'.format(dataset_name),
            'path_to_images': '',
            'path_to_train': 'data/HMC_data/{}/train-normalized.arff'.format(dataset_name, dataset_name),
            'path_to_val': 'data/HMC_data/{}/dev-normalized.arff'.format(dataset_name, dataset_name),
            'path_to_test': 'data/HMC_data/{}/test-normalized.arff'.format(dataset_name, dataset_name),
            'path_to_edges': 'data/HMC_data/{}/hierarchy.edgelist'.format(dataset_name),
            'path_to_tc_edges': 'data/HMC_data/{}/hierarchy_tc.edgelist'.format(dataset_name)}
    else:
        raise NotImplementedError('Metadata dictionary not implemented.')
    return meta

def get_imagenet_stats():
    '''
    Returns standard ImageNet statistics. 
    '''
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    return (imagenet_mean, imagenet_std)

def get_transforms():
    '''
    Returns image transforms.
    '''
    
    (imagenet_mean, imagenet_std) = get_imagenet_stats()
    tx = {}
    tx['train'] = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    tx['val'] = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    tx['test'] = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    return tx

def generate_split(num_ex, frac, rng):
    '''
    Computes indices for a randomized split of num_ex objects into two parts,
    so we return two index vectors: idx_1 and idx_2. Note that idx_1 has length
    (1.0 - frac)*num_ex and idx_2 has length frac*num_ex. Sorted index sets are 
    returned because this function is for splitting, not shuffling. 
    '''
    
    # compute size of each split:
    n_2 = int(np.round(frac * num_ex))
    n_1 = num_ex - n_2
    
    # assign indices to splits:
    idx_rand = rng.permutation(num_ex)
    idx_1 = np.sort(idx_rand[:n_1])
    idx_2 = np.sort(idx_rand[-n_2:])
    
    return (idx_1, idx_2)

def get_data(P):
    '''
    Given a parameter dictionary P, initialize and return the specified dataset. 
    '''
    
    # define transforms:
    tx = get_transforms()

    # select and return the right dataset:
    ds = multilabel(P, tx).get_datasets()

    # Optionally overwrite the observed training labels with clean labels:
    # assert P['train_set_variant'] in ['clean', 'observed']
    if P['train_set_variant'] == 'clean':
        # print('Using clean labels for training.')
        ds['train'].label_matrix_obs = copy.deepcopy(ds['train'].label_matrix)

    elif P['train_set_variant'] == 'multi_path':
        class_names = ds['train'].class_names
        original_labels = ds['train'].label_matrix.tolist()

        def get_paths(names):
            paths = []
            for i, name in enumerate(names):
                leaf = True
                for previous_name in names:
                    if name in previous_name and name != previous_name:
                        leaf = False
                        break
                if leaf:
                    # When name is leaf class
                    path_element = name.split('.')
                    path = [path_element[0:i + 1] for i in range(len(path_element))]
                    path = ['.'.join(sublist) for sublist in path]
                    paths.append(path)
            # paths = [['14', '14.04'], ['16', '16.07'], ['20', '20.01', '20.01.10'], ['20', '20.01', '20.01.21'], ['20', '20.09', '20.09.01'], ['42', '42.10', '42.10.05']]
            return paths

        multi_path_labels = []
        for o_label in original_labels:
            multi_path_label = np.zeros(ds['train'].label_matrix.shape[1])

            names = [name for name, label in zip(class_names, o_label) if label == 1]
            paths = get_paths(names)
            obs_names = [list(np.random.choice(path, 1))[0] for path in paths]
            obs_names = list(set(obs_names))
            obs_labels_idx = [i for i, name in enumerate(class_names) if name in obs_names]

            multi_path_label[obs_labels_idx] = 1
            multi_path_labels.append(multi_path_label)
        multi_path_labels = np.array(multi_path_labels)
        
        ds['train'].label_matrix_obs = copy.deepcopy(multi_path_labels)

    elif P['label_proportion'] != None:
        def sumup(arr):
            return np.sum(np.sum(arr,0))

        if P['label_proportion'] == 1:
            ds['train'].label_matrix_obs = copy.deepcopy(ds['train'].label_matrix)
        else:
            label_full =  ds['train'].label_matrix
            label_obs = ds['train'].label_matrix_obs
            label_hidden = label_full - label_obs
            label_additional = copy.deepcopy(label_obs)

            num_additiaonl = int(sumup(label_full) * P['label_proportion']) - int(sumup(label_obs))
            np.random.seed(P['seed'])
            pos_hidden = np.where(label_hidden > 0)
            idx = np.random.choice(len(pos_hidden[0]), num_additiaonl, replace=False)

            pos_additional = (pos_hidden[0][idx], pos_hidden[1][idx])
            label_additional[pos_additional] = 1

            ds['train'].label_matrix_obs = copy.deepcopy(label_additional)
    else:
        pass
        # print('Using single positive labels for training.')
    
    # Optionally overwrite the observed val labels with clean labels:
    assert P['val_set_variant'] in ['clean', 'observed']
    if P['val_set_variant'] == 'clean':
        # print('Using clean labels for validation.')
        ds['val'].label_matrix_obs = copy.deepcopy(ds['val'].label_matrix)
    else:
        print('Using single positive labels for validation.')
    
    # We always use a clean test set:
    ds['test'].label_matrix_obs = copy.deepcopy(ds['test'].label_matrix)
            
    return ds


def load_data(base_path, P):
    data = {}
    for phase in ['train', 'val']:
        data[phase] = {}
        data[phase]['labels'] = np.load(os.path.join(base_path, 'formatted_{}_labels.npy'.format(phase)))
        data[phase]['labels_obs'] = np.load(os.path.join(base_path, 'formatted_{}_labels_obs_{}.npy'.format(phase, P['seed'])))
        data[phase]['images'] = np.load(os.path.join(base_path, 'formatted_{}_images.npy'.format(phase)))
        # data[phase]['feats'] = np.load(P['{}_feats_file'.format(phase)]) if P['use_feats'] else []

    # from collections import Counter
    # import util
    # labels_list = util.GET(data['train']['labels'])
    # labels_test_list = util.GET(data['val']['labels'])
    # import itertools
    # new_list = list(itertools.chain.from_iterable(labels_list))
    # new_test_list = list(itertools.chain.from_iterable(labels_test_list))
    # # print(max(new_list), len(set(new_list)))
    # merged_list = new_list + new_test_list
    # counter = Counter(merged_list)
    # print(data['train']['labels'].shape, data['val']['labels'].shape)
    # one_count_labels = [k for k,v in counter.items() if v == 1]
    # # print(one_count_labels)`;/. 5
    #
    # exit()
    # # counter = Counter(util.GET(data['train']['labels_obs']).tolist())
    # for k,v in counter.items():
    #     print(k, v)
    # original_labels = data['train']['labels']
    # print(original_labels.shape)
    # print(original_labels.sum(0))
    # print(original_labels.sum(0).shape)
    # print(util.GET(original_labels))
    # exit()

    # Ignore
    if P['my_label']:
        loss = 'hw' if P['mod_scheme'] == 'hw' else 'hc'
        print("Changed To My Own Label")
        data['val']['labels'] = np.load(os.path.join(base_path, 'formatted_val_labels_{}.npy'.format(loss)))
    return data

class multilabel:

    def __init__(self, P, tx):
        
        # get dataset metadata:
        meta = get_metadata(P)
        self.base_path = meta['path_to_dataset']
        
        # load data:
        source_data = load_data(self.base_path, P)
        
        # generate indices to split official train set into train and val:
        split_idx = {}
        (split_idx['train'], split_idx['val']) = generate_split(
            len(source_data['train']['images']),
            P['val_frac'],
            np.random.RandomState(P['split_seed'])
            )
        
        # subsample split indices: # commenting this out makes the val set map be low?
        ss_rng = np.random.RandomState(P['ss_seed'])
        temp_train_idx = copy.deepcopy(split_idx['train'])
        for phase in ['train', 'val']:
            num_initial = len(split_idx[phase])
            num_final = int(np.round(P['ss_frac_{}'.format(phase)] * num_initial))
            split_idx[phase] = split_idx[phase][np.sort(ss_rng.permutation(num_initial)[:num_final])]


        if P['dataset'] in ['cars', 'cub2', 'cifar']:
            '''
            self.{phase} has dataset_name, num_classes, image_ids, label_matrix, label_matrix_obs, tx
            '''
            self.train = ds_multilabel(
                P['dataset'],
                source_data['train']['images'][split_idx['train']],
                source_data['train']['labels'][split_idx['train'], :],
                source_data['train']['labels_obs'][split_idx['train'], :],
                # source_data['train']['feats'][split_idx['train'], :] if P['use_feats'] else [],
                tx['train'],
                P
            )
            # define val set:
            self.val = ds_multilabel(
                P['dataset'],
                source_data['train']['images'][split_idx['val']],
                source_data['train']['labels'][split_idx['val'], :],
                source_data['train']['labels_obs'][split_idx['val'], :],
                # source_data['train']['feats'][split_idx['val'], :] if P['use_feats'] else [],
                tx['val'],
                P
            )
            # define test set:
            self.test = ds_multilabel(
                P['dataset'],
                source_data['val']['images'],
                source_data['val']['labels'],
                source_data['val']['labels_obs'],
                # source_data['val']['feats'],
                tx['test'],
                P
            )

        # For HMC
        if P['dataset'] not in ['cars', 'cub2', 'cifar']:
            '''
            self.{phase} has dataset_name, images, label_matrix, label_matrix_obs, adj_matrix, edges, tc_edges
            '''
            self.train = ds_arff(
                P['dataset'],
                source_data['train']['images'][split_idx['train']],
                source_data['train']['labels'][split_idx['train'], :],
                source_data['train']['labels_obs'][split_idx['train'], :],
                P
            ) # It has self.train.images, self.train.labels, self.train.adj_matrix, self.train.edges, self.train.tc_edges
            self.val = ds_arff(
                P['dataset'],
                source_data['train']['images'][split_idx['val']],
                source_data['train']['labels'][split_idx['val'], :],
                source_data['train']['labels_obs'][split_idx['val'], :],
                P
            )
            self.test = ds_arff(
                P['dataset'],
                source_data['val']['images'],
                source_data['val']['labels'],
                source_data['val']['labels_obs'],
                P
            )
        # define dict of dataset lengths: 
        self.lengths = {'train': len(self.train), 'val': len(self.val), 'test': len(self.test)}

    def get_datasets(self):

        import util
        from collections import defaultdict, Counter
        multilabels = util.GET(self.train.label_matrix)
        counter = []
        for ml in multilabels:
            for l in ml:
                counter.append(l)
        # print(Counter(counter))
        counter = Counter(counter)
        train_classes = counter.keys()

        multilabels = util.GET(self.test.label_matrix)
        test_counter = []
        for ml in multilabels:
            for l in ml:
                test_counter.append(l)
        # print(Counter(test_counter))
        test_counter = Counter(test_counter)
        test_classes = test_counter.keys()
        # print(len(test_classes))
        #
        # test_only = [c for c in test_classes if c not in train_classes]
        # train_only = [c for c in train_classes if c not in test_classes]

        return {'train': self.train, 'val': self.val, 'test': self.test}

class ds_arff(Dataset): # For train
    def __init__(self, dataset_name, images, label_matrix, label_matrix_obs, P):
        self.dataset_name = dataset_name
        self.image = images
        self.label_matrix = label_matrix
        self.label_matrix_obs = label_matrix_obs

        meta = get_metadata(P)

        self.num_classes = config._LOOKUP['num_classes'][dataset_name]
        import datasets_arff
        ds = datasets_arff.arff_data2(meta['path_to_train'], meta['path_to_edges'], meta['path_to_tc_edges'])

        self.class_names = ds.class_names
        self.edges = ds.edges
        self.tc_edges = ds.tc_edges

        self.adj_matrix = torch.tensor(ds.adj_matrix) + torch.eye(self.num_classes)
        self.adj_matrix = self.adj_matrix.unsqueeze(0)

        true_edge_path = ospj('saved_multi', 'edge', 'true_edge_{}'.format(P['dataset']))
        if not os.path.exists(true_edge_path):
            SAVE(self.tc_edges, true_edge_path)

    def __len__(self):
        return len(self.image)


    def __getitem__(self, idx):
        out = {
            'image': self.image[idx],
            'label_vec_obs': torch.FloatTensor(np.copy(self.label_matrix_obs[idx, :])),
            'label_vec_true': torch.FloatTensor(np.copy(self.label_matrix[idx, :])),
            'idx': idx,
            'adj_matrix': self.adj_matrix,
            'edges': self.edges,
            'tc_edges': self.tc_edges
        }
        return out


class ds_multilabel(Dataset):

    def __init__(self, dataset_name, image_ids, label_matrix, label_matrix_obs, tx, P):
        meta = get_metadata(P)
        self.dataset_name = dataset_name
        self.num_classes = config._LOOKUP['num_classes'][dataset_name]
        self.path_to_images = meta['path_to_images']

        self.image_ids = image_ids
        self.label_matrix = label_matrix
        self.label_matrix_obs = label_matrix_obs
        self.tx = tx
        # self.use_feats = use_feats

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self,idx):
        if self.dataset_name == 'cifar':
            image = self.image_ids[idx]
            image = image.reshape((3, 32, 32)).transpose(1, 2, 0)
            image = Image.fromarray(image)
            I = self.tx(image)
        else:
            # Set I to be an image:
            image_path = os.path.join(self.path_to_images, self.image_ids[idx])
            with Image.open(image_path) as I_raw:
                I = self.tx(I_raw.convert('RGB'))
        
        out = {
            'image': I,
            'label_vec_obs': torch.FloatTensor(np.copy(self.label_matrix_obs[idx, :])),
            'label_vec_true': torch.FloatTensor(np.copy(self.label_matrix[idx, :])),
            'idx': idx,
            # 'image_path': image_path # added for CAM visualization purpose
        }
        
        return out

# HMC dataset check from MBM code
# import datasets_arff
# # datasets_arff.arff_data('data/HMC_data/spo_GO/train.arff')
# # arff_file = 'data/HMC_data/spo_GO/train.arff'
# # arff_file =  'data/HMC_data/{}/{}.train.arff'.format('spo_GO', 'spo_GO')
# dataset='diatoms'
#
# for phase in ['train', 'test', 'dev']:
# # for phase in ['train']:
#     arff_file = 'data/HMC_data/{}/{}.arff'.format(dataset, phase)
#     tc_edge_path ='data/HMC_data/{}/hierarchy_tc.edgelist'.format(dataset)
#     edge_path ='data/HMC_data/{}/hierarchy.edgelist'.format(dataset)
#     X, Y, edges, tc_edges = datasets_arff.parse_arff2(arff_file=arff_file, edge_path=edge_path, tc_edge_path=tc_edge_path)
#     print(Y.shape)
#
# print("+++++++++++++++++")
#
# for phase in ['train']:
#     arff_file = 'data/HMC_data/{}/{}.arff'.format(dataset, phase)
#     tc_edge_path ='data/HMC_data/{}/hierarchy_tc.edgelist'.format(dataset)
#     edge_path ='data/HMC_data/{}/hierarchy.edgelist'.format(dataset)
#     X, Y, edges, tc_edges = datasets_arff.parse_arff2(arff_file=arff_file, edge_path=edge_path, tc_edge_path=tc_edge_path)
#     print(Y.shape)
# total_Y = Y
# for phase in ['dev']:
#     arff_file = 'data/HMC_data/{}/{}.arff'.format(dataset, phase)
#     tc_edge_path ='data/HMC_data/{}/hierarchy_tc.edgelist'.format(dataset)
#     edge_path ='data/HMC_data/{}/hierarchy.edgelist'.format(dataset)
#     X, Y, edges, tc_edges = datasets_arff.parse_arff2(arff_file=arff_file, edge_path=edge_path, tc_edge_path=tc_edge_path)
#     print(Y.shape)
#
# total_Y = np.concatenate((total_Y,Y), axis=0)
# print(total_Y.shape)
# import util
# from collections import Counter
# multilabels = util.GET(total_Y)
# # print(multilabels)
# counter = []
# for ml in multilabels:
#     for l in ml:
#         counter.append(l)
# # print(Counter(counter))
# counter = Counter(counter)
# # print(counter)
# train_classes = counter.keys()
# print(len(train_classes))
# exit()
# for phase in ['test']:
#     arff_file = 'data/HMC_data/{}/{}.arff'.format(dataset, phase)
#     tc_edge_path ='data/HMC_data/{}/hierarchy_tc.edgelist'.format(dataset)
#     edge_path ='data/HMC_data/{}/hierarchy.edgelist'.format(dataset)
#     X, Y, edges, tc_edges = datasets_arff.parse_arff2(arff_file=arff_file, edge_path=edge_path, tc_edge_path=tc_edge_path)
# multilabels = util.GET(Y)
# test_counter = []
# for ml in multilabels:
#     for l in ml:
#         test_counter.append(l)
# # print(Counter(test_counter))
# test_counter = Counter(test_counter)
# test_classes = test_counter.keys()
# print(len(test_classes))






# HMC dataset genreate npy file
# import datasets_arff
# name = ['cellcycle_FUN', 'cellcycle_GO', 'derisi_FUN', 'derisi_GO', 'diatoms', 'enron_corr',
#         'expr_FUN', 'expr_GO', 'imclef07a', 'imclef07d', 'spo_FUN', 'spo_GO']
# name = ['imclef07a', 'imclef07d']
# for dataset_name in name:
#     print(dataset_name)
#     train, val, test = datasets_arff.initialize_dataset(dataset_name)
#     if val == []:
#         train, _,  test = datasets_arff.initialize_dataset(dataset_name)
#
#     # print(len(train.images), len(val.images))
#     # train + val
#     # print(len(train.images))
#
#     np.save(os.path.join('data/HMC_data/{}'.format(dataset_name), 'formatted_val_images.npy'), test.images)
#     np.save(os.path.join('data/HMC_data/{}'.format(dataset_name), 'formatted_val_labels.npy'), test.labels)
#     np.save(os.path.join('data/HMC_data/{}'.format(dataset_name), 'formatted_train_images.npy'), train.images)
#     np.save(os.path.join('data/HMC_data/{}'.format(dataset_name), 'formatted_train_labels.npy'), train.labels)

