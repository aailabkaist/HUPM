import datasets
import models
from util import *
from config import get_configs
from os.path import join as ospj
import pandas as pd
# csv_dir = ospj("saved_multi","cub_hierarchy.csv")
# df = pd.read_csv(csv_dir)

'''
Training중에 extracted된 edge가 있는 txt file 읽기
'''
# python extract_hierarchy.py --dataset cub2 --checkpoint results/hw_cub2_1e-5_8_1_0.3_warmup0.3@30_1/warmup0.3@30_1 --warmup_epoch 30 --seed 1 --mod_scheme hw
# P = get_configs()
epoch_pos = 4 # edge file name split -> position of epoch@@

edge_file = 'edge_cub_hw_2.txt'
edge_file = 'edge_cars_hw_3.txt'
edge_file = 'edge_cifar_hw_1.txt'

corr_edges = []
wrong_edges = []
edges = []
with open(ospj('saved_multi', 'edge', edge_file), 'r') as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    # print(i, line[:-2])
    if i < 3: continue
    line = line[:-2]
    src, des, weight, correct = line.split(', ')
    src, des, weight = int(src), int(des), float(weight)
    edge = (src, des, weight)
    correct = True if correct == '1' else False
    if correct:
        corr_edges.append(edge)
    else:
        wrong_edges.append(edge)
    edges.append(edge)

print(edges)
exit()
