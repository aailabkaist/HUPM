
'''
Ver2 Add warmup_pretrained
Ver3 Add edge_init
Ver4 Add gamma_before/gamma_after, HMC dataset
Ver5 Add Edge initialization tuning, HMC dataset
Ver5.1 Add get_dist for analyze distibution
Ver6 Add ablation partial and get_edge
'''
import argparse
from munch import Munch as mch
import util
import torch
import os
from os.path import join as ospj
import itertools
import numpy as np

_DATASET = ('cub2', 'cifar', 'cars',
            'cellcycle_FUN', 'cellcycle_GO', 'derisi_FUN', 'derisi_GO', 'diatoms', 'enron_corr', 'expr_FUN', 'expr_GO', 'imclef07a', 'imclef07d', 'spo_FUN', 'spo_GO')
_TRAIN_SET_VARIANT = ('observed', 'clean', 'multi_path')
_OPTIMIZER = ('adam', 'sgd')
_SCHEMES = ('LL-R', 'LL-Ct', 'LL-Cp', 'an', 'an-ls', 'wan', 'role', 'bce', 'bce-ls', 'hmcnn', 'hw', 'hc', 'an-ls+hw', 'an-ls+hc', 'wan+hw', 'wan+hc')
_LOOKUP = {
    'feat_dim': {
        'resnet50': 2048
    },
    'num_classes': {
        'cub2': 250, 'cifar': 120, 'cars': 205,
        'cellcycle_FUN': 499, 'cellcycle_GO': 4122,
        'derisi_FUN': 499, 'derisi_GO': 4116,
        'diatoms': 398, 'enron_corr': 56,
        'expr_FUN': 499, 'expr_GO': 4128,
        'imclef07a': 96, 'imclef07d': 46,
        'spo_FUN': 499, 'spo_GO': 4116,

        #@@@@@@@@@@@@@@@Other dataset
    },
    'num_level_classes':{
        'cub2': [13, 37, 200], 'cifar': [20, 100], 'cars': [9, 196]
    },
    'num_level_classes_cum': {
        'cub2': [13, 50, 250], 'cifar': [20, 120], 'cars': [9, 205]
    },
    'expected_num_pos':{
        'cifar': 2, 'cars': 2, 'cub2': 3
    },
    'warmup_epoch': {
        'cub2': 30, 'cifar': 10, 'cars': 25,
    },
    'hier_threshold': {
        'cub2': 0.5, 'cifar': 0.4, 'cars': 0.2,
    },
    'gamma_before': {
        'cub2': 0.004, 'cifar': 0.4, 'cars': 0.005,
    },
    'gamma_after': {
        'cub2': 0.4, 'cifar': 0.6, 'cars': 1,
    }

}
_num_true_edges = {'cifar':100, 'cars':196, 'cub2':437}
_SEED = (1, 2, 3, 4, 5)

_input_dims = {'cellcycle_FUN': 77, 'cellcycle_GO': 77, 'derisi_FUN': 63, 'derisi_GO': 63,
              'diatoms': 371, 'enron_corr': 1001, 'expr_FUN': 561, 'expr_GO': 561, 'imclef07a': 80, 'imclef07d': 80,
              'spo_FUN': 86, 'spo_GO': 86}

_hidden_dims = {'cellcycle_FUN': 500, 'derisi_FUN': 500,  'expr_FUN': 1250, 'spo_FUN': 250,
                'cellcycle_GO': 1000, 'derisi_GO': 500,'expr_GO': 4000, 'spo_GO': 500,
                'diatoms': 2000, 'enron_corr': 1000, 'imclef07a': 1000, 'imclef07d': 1000}


def set_follow_up_configs(args):
    # args.stop = not args.no_stop
    args.train = not args.test and not args.correction and not args.edge and not args.dist
    assert '_' not in args.flag, "Do not use '_' in flag!"
    args.flag = args.flag + '_' + str(args.seed)
    args.extract = not args.no_extract

    args.feat_dim = _LOOKUP['feat_dim'][args.arch]
    args.num_classes = _LOOKUP['num_classes'][args.dataset]

    args.exp_name = '{}_{}_{}_{}_{}_{}_{}'.format(args.mod_scheme, args.dataset, util.str_float(args.lr), args.bsize,
                                                      args.iter_epoch, args.hier_threshold, args.flag)

    if args.train:
        # final_result.txt: Save final results and arguments
        args.result_path = util.SET_PATH(ospj(args.save_path, args.exp_name), 'final_result.txt')
        # Save directory for models and edges
        args.save_dir = util.SET_PATH(ospj(ospj(args.save_path, args.exp_name), args.flag))

    if args.mod_scheme in ['hw', 'hc', 'an-ls+hw', 'an-ls+hc', 'wan+hw', 'wan+hc']:
        args.my = True
    else:
        args.my = False

    if args.mod_scheme in ['LL-R', 'LL-Ct', 'LL-Cp']:
        # args.delta_rel /= 100
        args.clean_rate = 1

    if args.mod_scheme == 'role':
        args.expected_num_pos = _LOOKUP['expected_num_pos'][args.dataset]

    if args.dataset in ['cars', 'cub2', 'cifar']:
        args.HMC = False
        R_path = 'saved_multi/metadata/{}_mat.pkl'.format(args.dataset)
        R = util.LOAD(R_path)
        args.R = torch.tensor(R).unsqueeze(0)
    else:
        args.HMC = True
        args.input_dim = _input_dims[args.dataset]
        args.hidden_dim = _hidden_dims[args.dataset]
        args.non_lin = 'relu'

    if args.warmup_pretrained:
        assert args.my,                 'warmup_pretrained: Only available for my model'
        assert args.checkpoint != None, 'warmup_pretrained: Set checkpoint'
        assert args.warmup_epoch > 0,   'warmup_pretrained: Set warmup_epoch same as checkpoint'

        idx = args.checkpoint.find('@')
        # checkpoint_warmup_epoch = int()
        checkpoint_hier_threshold = float(args.checkpoint[idx - 3: idx])
        # start_at_last = checkpoint_hier_threshold != args.hier_threshold
        # print(checkpoint_hier_threshold, start_at_last, args.hier_threshold)
        args.skip_epoch = args.warmup_epoch - 1
        args.checkmodel = 'model_{}.pt'.format(args.warmup_epoch - 1)
        args.edge_init_type = 'none'

        # if start_at_last:
        #     # start at last warmup epoch (ex: epoch 30)
        #     args.skip_epoch = args.warmup_epoch - 1
        #     args.checkmodel = 'model_{}.pt'.format(args.warmup_epoch - 1)
        #     args.edge_init_type = 'none'
        #
        # else:
        #     # start at HAN-W epoch (ex: epoch 31)
        #     args.skip_epoch = args.warmup_epoch
        #     args.checkmodel = 'model_{}.pt'.format(args.warmup_epoch)
        #     args.edge_init_type = 'load'
        #     args.edge_load_epoch = args.warmup_epoch

        path = ospj(args.checkpoint, args.checkmodel)
        assert os.path.exists(path),    'warmup_pretrained: {} does not exist'.format(path)


    if args.edge_init_type == 'none':
        args.edges = []
        # if args.true_edge:
        #     true_edge_path = ospj('saved_multi', 'edge', 'true_edge_{}'.format(args.dataset))
        #     true_edge = util.LOAD(true_edge_path)
        #     args.edges = [(src, des, 1) for src, des in true_edge]

    elif args.edge_init_type == 'random':
        assert args.edge_random_prop > 0, \
            'edge_init_type random: Set edge_random_prop'
        # assert args.edge_random_weight <= args.hier_threshold, \
        #     'edge_init_type random: edge_random_weight should be lower than hier_threshold'

        total_comb = list(itertools.combinations(([i for i in range(args.num_classes)]), 2))
        num_total_comb = len(total_comb)
        num_rand_comb = int(len(total_comb) * args.edge_random_prop)

        rand_idx = np.random.choice(num_total_comb, num_rand_comb, replace=False)
        rand_comb = np.array(total_comb)[rand_idx]
        rand_direction = np.random.choice(2, len(rand_comb))

        rand_edges = []
        for i, (src, des) in enumerate(rand_comb):
            if rand_direction[i]:
                rand_edges.append((src, des, args.edge_random_weight))
            else:
                rand_edges.append((des, src, args.edge_random_weight))
        args.edges = np.array(rand_edges)

    elif args.edge_init_type == 'load':
        assert args.checkpoint != None,      'edge_init_type load: Set checkpoint'
        assert args.edge_load_epoch != None, 'edge_init_type load: Set edge_load_epoch (0: random, >1: warmup)'
        edge_file_list = [file for file in os.listdir(args.checkpoint) if file[:5] == 'edges']
        no_edge = True
        for edge_file in edge_file_list:
            epoch = util.get_epoch_from(edge_file)
            if epoch == args.edge_load_epoch:
                no_edge = False
                break
        if no_edge:
            print("Epoch {} edge does not exist!".format(args.edge_load_epoch))
            exit()

        path = ospj(args.checkpoint, edge_file)
        assert os.path.exists(path),        'edge_init_type load: {} does not exist'.format(path)
        with open(path, 'r') as f:
            lines = f.readlines()

        load_edges = []
        for i, line in enumerate(lines):
            if i < 3: continue
            line = line[:-2]
            src, des, weight, _ = line.split(', ')
            src, des, weight = int(src), int(des), float(weight)
            load_edges.append((src, des, weight))
        args.edges = np.array(load_edges)
        args.checkedge = path
        # args.skip_epoch = args.edge_load_epoch



    return args

def schemes():
    return _SCHEMES

def get_configs():
    parser = argparse.ArgumentParser()

    # Default settings
    parser.add_argument('--seed', type=int, required=True, choices=_SEED)
    parser.add_argument('--ss_seed', type=int, default=999, help='seed fo subsampling')
    parser.add_argument('--ss_frac_train', type=float, default=1.0, help='fraction of training set to subsample')
    parser.add_argument('--ss_frac_val', type=float, default=1.0, help='fraction of val set to subsample')
    parser.add_argument('--val_frac', type=float, default=0.2)
    parser.add_argument('--split_seed', type=int, default=1200)
    parser.add_argument('--train_set_variant', type=str, default='observed', choices=_TRAIN_SET_VARIANT)
    parser.add_argument('--val_set_variant', type=str, default='clean')
    parser.add_argument('--arch', type=str, default='resnet50')
    parser.add_argument('--use_pretrained', type=bool, default=True)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--label_proportion', type=float)


    # Util
    parser.add_argument('--image_path', type=str, default='../../../../data/')
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--exp_name', type=str, default='exp_default')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu_num', type=str, default='0')
    parser.add_argument('--flag', type=str, default='default')

    # Data
    parser.add_argument('--dataset', type=str, required=True, choices=_DATASET)

    # Baseline param
    parser.add_argument('--delta_rel', type=float, default=1e-4)
    parser.add_argument('--permanent', action='store_true')


    # param
    # parser.add_argument('--no_stop', action='store_true')
    parser.add_argument('--stop', action='store_true')
    parser.add_argument('--optimizer', type=str, default='adam', choices=_OPTIMIZER)
    parser.add_argument('--mod_scheme', type=str, required=True, choices=_SCHEMES)
    parser.add_argument('--bsize', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lr_mult', type=float, default=10)
    parser.add_argument('--hier_threshold', type=float, default=0.2)
    parser.add_argument('--iter_epoch', type=int, default=1)
    # parser.add_argument('--my', action='store_true')
    parser.add_argument('--my_cont', action='store_true')
    parser.add_argument('--true_edge', action='store_true') # Give true edge to get_pseudo_w_y

    # after thesis
    parser.add_argument('--weight_sampler', action='store_true')
    parser.add_argument('--weight_classifier', action='store_true')
    parser.add_argument('--weight_classifier_ver', type=str, default='1.1', choices=['1.1', '1.2', '2.1', '2.2'])
    parser.add_argument('--edge_w_mult', type=float, default=1)
    parser.add_argument('--no_extract', action='store_true')
    parser.add_argument('--edge_init_type', type=str, default='none', choices=['none', 'random', 'load'])
    parser.add_argument('--edge_random_prop', type=float, default=0)
    parser.add_argument('--edge_random_weight', type=float, default=0.2)
    parser.add_argument('--edge_load_epoch', type=int, help='0: random edge, >1: warmup edge')
    parser.add_argument('--skip_epoch', type=int, default=0)
    parser.add_argument('--edge_fix', action='store_true')
    parser.add_argument('--sampling_child', action='store_true')


    # warmup
    parser.add_argument('--mod_warmup', type=str, default='an', choices=_SCHEMES)
    parser.add_argument('--warmup_epoch', type=int, default=1)
    parser.add_argument('--warmup_pretrained', action='store_true')
    parser.add_argument('--gamma_before', type=float, help='warmup gamma')
    parser.add_argument('--gamma_after', type=float, default=1, help='after warmup gamma')

    # HMC
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('--nonlin', type=str, default='relu')



    # param search
    parser.add_argument('--bsize+', nargs='+', type=int)
    parser.add_argument('--lr+', nargs='+', type=float)
    parser.add_argument('--hier_threshold+', nargs='+', type=float)
    parser.add_argument('--iter_epoch+', nargs='+', type=int)
    parser.add_argument('--delta_rel+', nargs='+', type=float)
    parser.add_argument('--skip_i+', nargs='+', type=int)

    # test
    parser.add_argument('--checkpoint', type=str, help='./results/EXP_NAME/FLAG')
    parser.add_argument('--checkmodel', type=str, help='bestmodel.pt')
    parser.add_argument('--checkpoint+', nargs='+', type=str)
    parser.add_argument('--checkmodel+', nargs='+', type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--edge', action='store_true')
    parser.add_argument('--correction', action='store_true')
    parser.add_argument('--my_label', action='store_true')
    parser.add_argument('--val_best_th', type=float, default=-1)
    parser.add_argument('--test_all', action='store_true')
    parser.add_argument('--test_leaf', action='store_true')


    # tsne
    parser.add_argument('--tsne', action='store_true')
    parser.add_argument('--tsne_p', action='store_true')

    # ablation
    parser.add_argument('--ablation', action='store_true')
    parser.add_argument('--reduce_ratio', type=float, default=1.0)
    parser.add_argument('--dist', action='store_true')
    parser.add_argument('--hw_ratio', type=float, default=None)
    parser.add_argument('--hw_ratio+', nargs='+', type=float)
    parser.add_argument('--child_conf', type=float, default=0)
    parser.add_argument('--constraint', action='store_true')

    args = parser.parse_args()
    args = set_follow_up_configs(args)
    args = mch(**vars(args))
    return args


