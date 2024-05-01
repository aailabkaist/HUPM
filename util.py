import torch
from collections import defaultdict
import numpy as np
import pickle
import os
from os.path import join as ospj
import config
from tqdm import tqdm
import time


level_dic = {'cub': 3, 'cifar': 2, 'cars': 2, 'cub2':3}

def SET_PATH(dir_path, file=None):
    os.makedirs(dir_path, exist_ok=True)
    if file == None:
        return dir_path
    return ospj(dir_path, file)

def PATH_CHECK(path):
    # Similar to Windows file system (ex: result.txt already exists => result(1).txt)
    unique = 1
    ext_list = ['.png', '.jpg', '.jpeg', '.txt', '.pth.tar']
    file_name, file_ext = path, ""

    for ext in ext_list:
        idx = path.find(ext)
        if idx != -1:
            file_name, file_ext = path[:idx], path[idx:]
            break

    while os.path.exists(path):  
        uniq = str(unique)
        path = file_name + "(" + uniq + ")" + file_ext
        unique += 1
    return path

def SAVE(file, filepath):
    filepath = PATH_CHECK(filepath)
    with open(filepath, 'wb') as f:
        pickle.dump(file, f)

def LOAD(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data

def GET(label):
    # Change One-hot/Multi-hot label into Class id (ex: [1,0,1,0,0] => [0,2])
    if len(label.shape) == 1:
        return np.where(label)[0].tolist()
    else:
        new_label = []
        for l in label:
            new_label.append(np.where(l)[0].tolist())
        try:
            new_label_tensor = torch.tensor(new_label).squeeze()
        except:
            return new_label
        return new_label_tensor

def save_and_print(path, st):
    with open(path, 'a') as f:
        f.write(st)
        f.write('\n')
    print(st)

def str_float(number):
    if number == 1e-1:
        return '1e-1'
    elif number == 1e-2:
        return '1e-2'
    elif number == 1e-3:
        return '1e-3'
    elif number == 1e-4:
        return '1e-4'
    elif number == 1e-5:
        return '1e-5'
    else:
        return str(number)

def get_epoch_from(file_name, epoch_pos=3):
    # Input: 'edges_default_1_epoch12_153|158_rec78.1|pre96.8.txt'
    # Output: 12
    return int(file_name.split('_')[epoch_pos][5:])

def set_experiment(P):
    pass
    # P['exp_name'] = '{}_{}_{}_{}_{}_{}_{}'.format(P['mod_scheme'], P['dataset'], str_float(P['lr']), P['bsize'],
    #                                                   P['iter_epoch'], P['hier_threshold'], P['flag'])
    # os.makedirs(ospj(P['save_path'], P['exp_name']), exist_ok=True)
    # P['result_path'] = SET_PATH(ospj(P['save_path'], P['exp_name']), 'final_result.txt')

def set_experiment_search(P):
    for args in ['bsize+', 'lr+', 'hier_threshold+', 'iter_epoch+', 'delta_rel+', 'hw_ratio+']:
        if P[args] == None: P[args] = [P[args[:-1]]]

    EXP = {}
    i = 0
    EXP['train_mode'] = 'default'
    if P['mod_scheme'] == 'role': EXP['train_mode'] = 'role'
    if P['ablation']: EXP['train_mode'] = 'ablation'

    P['exp_name'] = '{}_{}_{}_{}_{}_{}_{}'.format(P['mod_scheme'], P['dataset'], P['lr+'], P['bsize+'], P['iter_epoch+'], P['hier_threshold+'], P['delta_rel+'])
    if P['checkpoint'] != None:
        path = ospj('saved_multi', 'model', P['checkpoint'], P['checkmodel'])
        if os.path.exists(path):
            print("Finetune:", path)
            P['exp_name'] = '{}_{}_ft'.format(P['checkpoint'], P['checkmodel'][:-3])

    if P['ablation']:
        if P['reduce_ratio'] < 1:
            P['exp_name'] = 'ABL_REDUCE_' + P['exp_name']
        elif P['dist']:
            P['exp_name'] = 'ABL_DIST_' + P['exp_name']
        else:
            P['exp_name'] = 'ABL_' + P['exp_name']

    P['result_path'] = SET_PATH(ospj(P['save_path'], P['exp_name']), 'final_result.txt')

    EXP['total_i'] = len(P['bsize+']) * len(P['lr+']) * len(P['hier_threshold+']) * len(P['iter_epoch+']) * len(P['delta_rel+'])
    if P['ablation'] and len(P['hw_ratio+']) > 1:
        EXP['total_i'] *= len(P['hw_ratio+'])
    EXP['flags'] = []
    EXP['params'] = []
    EXP['param_dic'] = []
    save_and_print(P['result_path'],"=======EXP LIST=======")
    for bsize in P['bsize+']:
        for lr in P['lr+']:
            for hier_threshold in P['hier_threshold+']:
                for iter_epoch in P['iter_epoch+']:
                    for delta_rel in P['delta_rel+']:
                        for hw_ratio in P['hw_ratio+']:
                            flag = '{}_{}_{}_{}_{}_{}'.format(P['mod_scheme'], P['dataset'], lr, bsize, iter_epoch, hier_threshold)
                            if len(P['delta_rel+']) > 1:
                                flag += '_{}'.format(str_float(delta_rel))
                            if P['ablation'] and len(P['hw_ratio+']) > 1:
                                flag += '_hw{}_hc{}'.format(round(hw_ratio, 2), round(1-hw_ratio, 2))
                                EXP['params'].append([bsize, lr, hier_threshold, iter_epoch, delta_rel, hw_ratio])
                                EXP['param_dic'].append(
                                    {'lr': lr, 'hier_threshold': hier_threshold, 'iter_epoch': iter_epoch,
                                     'bsize': bsize, 'delta_rel': delta_rel, 'hw_ratio':hw_ratio})
                            else:
                                EXP['params'].append([bsize, lr, hier_threshold, iter_epoch, delta_rel])
                                EXP['param_dic'].append(
                                    {'lr': lr, 'hier_threshold': hier_threshold, 'iter_epoch': iter_epoch,
                                     'bsize': bsize, 'delta_rel': delta_rel})
                            EXP['flags'].append(flag)
                            save_and_print(P['result_path'], '{}/{} {}'.format(i, EXP['total_i'] - 1, flag))
                            i+=1
    save_and_print(P['result_path'],"=====================")
    return EXP

def stop(bestmap_val, map_val, bestmap_epoch, epoch, max_em, P):
    if P['stop'] == False: return False

    if bestmap_val - map_val > 5 and P['mod_warmup'] != 'wan':
        print("Validataion score decrease over 5!!")
        print('Early stopped.\n\n')
        return True
    if (epoch > 10 and map_val < 10 and P['mod_warmup'] != 'wan') and P['edge_init_type'] != 'random':
        print("Validation score too low!!")
        print('Early stopped.\n\n')
        return True
    if (epoch - bestmap_epoch) >= 10 and P['dataset'] == 'cifar':
        print("Cifar takes too long!!")
        print('Early stopped.\n\n')
        return True
    if (epoch - bestmap_epoch) >= 20:
        print("Validation score not increases!!")
        print('Early stopped.\n\n')
        return True
    if P['my'] and epoch > 10 and len(P['edges']) == 0 and P['edge_init_type'] != 'random' and P['mod_warmup'] != 'wan':
        print("No Edge generated!!")
        print('Early stopped.\n\n')
        return True
    # if epoch > 10 and max_em < 0.1:
    #     print("Match ratio too low!!")
    #     return True
    return False

def get_parent_dir(path):
    try:
        splited_cp = path.split("\\")
        assert len(splited_cp) > 1
        parent_cp = '\\'.join(splited_cp[:-1])
    except:
        splited_cp = path.split("/")
        parent_cp = '/'.join(splited_cp[:-1])

    return parent_cp

def get_reduced_dataset(dataset, P):
    if P['reduce_ratio'] == 1: return dataset
    train_len = len(dataset['train'])
    reduce_ratio = P['reduce_ratio']
    reduce_idx = np.random.choice(train_len, int(train_len * reduce_ratio), replace=False)

    dataset['train'].image_ids = dataset['train'].image_ids[reduce_idx]
    dataset['train'].label_matrix_obs = dataset['train'].label_matrix_obs[reduce_idx]
    dataset['train'].label_matrix = dataset['train'].label_matrix[reduce_idx]

    reduced_train_len = len(dataset['train'])
    print("Train Dataset is reduced to {} | {}->{}".format(reduce_ratio, train_len, reduced_train_len))
    return dataset

def get_pseudo_w_y(y, output, P):
    edges = P['edges']
    if len(edges) == 0:
        return torch.zeros_like(y)
    weights = {}
    parents = defaultdict(list)
    children = defaultdict(list)
    for edge in edges:
        par, child, w = edge
        par, child = int(par), int(child)
        weights[(par, child)] = w
        parents[child].append(par)
        children[par].append(child)

    pseudo_y = torch.zeros_like(y)
    for i, (xx, yy) in enumerate(zip(output,y)):
        yy_label = GET(yy)
        # assert len(yy_label) == 1,  print("REDUCE LABELS TO SINGLE-LABEL!")

        this_label = yy_label[0]
        if len(parents[this_label]) > 0:
            for par in parents[this_label]:
                pseudo_y[i][par] = weights[(par, this_label)]

        if len(children[this_label]) > 0:
            child_w = [xx[child].item() for child in children[this_label]]
            if P['sampling_child']:
                pass
                #TODO
            else:
                if max(child_w) > P['child_conf']:
                    max_idx = child_w.index(max(child_w))
                    max_child = children[this_label][max_idx]
                    pseudo_y[i][max_child] = weights[(this_label, max_child)]

    if P['hw_ratio']  != None:
        pseudo_y[pseudo_y > 0] = P['hw_ratio'] * pseudo_y[pseudo_y > 0] + (1 - P['hw_ratio']) * torch.ones_like(pseudo_y[pseudo_y > 0])

    if P['mod_scheme'] in ['hc', 'an-ls+hc', 'wan+hc']:
        pseudo_y[pseudo_y>0] = torch.ones_like(pseudo_y[pseudo_y > 0])

    if P['edge_w_mult'] != 1:
        pseudo_y *= P['edge_w_mult']
    return pseudo_y

def get_output_matrix(output, label):
    num_classes = label.size(1)
    output_mat = torch.zeros([num_classes, num_classes])
    output_list = {k: [] for k in range(num_classes)}
    output_count_list = {k: 0 for k in range(num_classes)}
    for out, lab in zip(output, label):
        l = GET(lab)
        for ll in l:
            output_list[ll].append(out)
            output_count_list[ll] += 1

    for k, v in output_list.items():
        try:
            output_list[k] = sum(v) / output_count_list[k]
        except:
            # print("Class {} does not exist!!!!!!!!".format(k))
            output_list[k] = 0
    # output_list = {k: sum(v) / output_count_list[k] for k, v in output_list.items()}
    for l in range(num_classes):
        output_mat[l] = output_list[l]
    output_mat = output_mat.T
    return output_mat

def get_current_edges(output_mat, hier_threshold):
    over_th = torch.where(output_mat > hier_threshold)
    possible_edges = torch.stack([over_th[0], over_th[1]]).T.tolist()  # Parent, Child
    possible_edges = [(i, j) for i, j in possible_edges if i != j]
    weighted_edges = [(i, j, round(output_mat[i][j].item(), 3)) for i, j in possible_edges if i != j]

    redunt_edge = []
    for e_idx, edge in enumerate(possible_edges):
        reversed_edge = edge[::-1]
        if reversed_edge in possible_edges:
            if output_mat[edge[0], edge[1]] < output_mat[edge[1], edge[0]]:
                if edge not in redunt_edge:
                    redunt_edge.append(e_idx)
            else:
                if reversed_edge not in redunt_edge:
                    redunt_edge.append(possible_edges.index(reversed_edge))

    w_edges = [w_e for i, w_e in enumerate(weighted_edges) if i not in redunt_edge]
    _edges = [e for i, e in enumerate(possible_edges) if i not in redunt_edge]
    return w_edges, _edges

def get_true_edges(dataset, weight=True):
    true_edge_path = ospj('saved_multi', 'edge', 'true_edge_{}'.format(dataset))
    true_edge = LOAD(true_edge_path)
    if weight:
        true_edge = [(src, des, 1) for src, des in true_edge]
    return true_edge

def get_edge_set(w_edge, dataset='cifar'):
    if len(w_edge) == 0 or dataset=='coco' or dataset=='pascal':
        return [], [], []
    true_edge = get_true_edges(dataset, weight=False)

    if len(w_edge[0]) == 3:
        edge = [(i, j) for i, j, w in w_edge]
    else:
        edge = w_edge

    correct_idx = []
    for idx, e in enumerate(edge):
        if e in true_edge:
            correct_idx.append(idx)
    intersection = [e for idx, e in enumerate(w_edge) if idx in correct_idx]
    wrong = [e for idx, e in enumerate(w_edge) if idx not in correct_idx]
    return intersection, wrong, true_edge

def get_constr_out(x, R):
    # print(x.get_device(), R.get_device())
    """ Given the output of the neural network x returns the output of MCM given the hierarchy constraint expressed in the matrix R """
    # R = [1, num_cls, num_cls]
    c_out = x.double()
    c_out = c_out.unsqueeze(1)
    c_out = c_out.expand(len(x), R.shape[1], R.shape[1])
    R_batch = R.expand(len(x), R.shape[1], R.shape[1]).to(x.get_device())
    final_out, _ = torch.max(R_batch * c_out.double(), dim=2)
    return final_out

def get_sampler(dataset, P):
    if P['weight_sampler'] == False and P['weight_classifier'] == False:
        return {'train':None, 'val':None, 'test':None}
    from torch.utils.data import sampler
    from collections import Counter
    labels = GET(dataset['train'].label_matrix_obs).tolist()
    counter = Counter(labels)
    weights = [len(labels)/ float(counter[l]) for l in labels]
    if P['weight_classifier']:
        class_weights = [len(labels) / counter[c] for c in range(P['num_classes'])]
        if P['weight_classifier_ver'] in ['1.1', '2.1']:
            # Ver*.1
            P['class_weights'] = np.array(class_weights) / np.min(class_weights)
        elif P['weight_classifier_ver'] in ['1.2', '2.2']:
            # Ver*.2
            P['class_weights'] = np.array(class_weights) / np.max(class_weights)
        print("Apply Weight Classifier")
        return {'train':None, 'val':None, 'test':None}
    train_sampler = sampler.WeightedRandomSampler(weights, len(weights))
    print("Apply Weight Sampler")
    return {'train':train_sampler, 'val':None, 'test':None}

def _get_class_names(data='cifar'):
    if data == 'cifar':
        long_class_names = ['Aquatic mammals', 'Fish', 'Flowers', 'Food containers', 'Fruit and vegetables', 'Household electrical devices',\
                       'Household furniture', 'Insects', 'Large carnivores', 'Large man made outdoor things', 'Large natural outdoor scenes', 'Large omnivores and herbivores',\
                       'Medium sized mammals', 'Non insect invertebrates', 'People', 'Reptiles', 'Small mammals', 'Trees', 'Vehicles1', 'Vehicles2', 'Beaver', 'Dolphin', 'Otter', 'Seal', 'Whale',\
                       'Aquarium fish', 'Flatfish', 'Ray', 'Shark', 'Trout', 'Orchid', 'Poppy', 'Rose', 'Sunflower', 'Tulip', 'Bottle', 'Bowl',\
                       'Can', 'Cup', 'Plate', 'Apple', 'Mushroom', 'Orange', 'Pear', 'Sweet pepper', 'Clock', 'Keyboard', 'Lamp', 'Telephone', 'Television', 'Bed', 'Chair', 'Couch', 'Table', 'Wardrobe', 'Bee',\
                       'Beetle', 'Butterfly', 'Caterpillar', 'Cockroach', 'Bear', 'Leopard', 'Lion', 'Tiger', 'Wolf', 'Bridge', 'Castle', 'House', 'Road', 'Skyscraper', 'Cloud', 'Forest', 'Mountain', 'Plain', \
                       'Sea', 'Camel', 'Cattle', 'Chimpanzee', 'Elephant', 'Kangaroo', 'Fox', 'Porcupine', 'Possum', 'Raccoon', 'Skunk', 'Crab', 'Lobster', 'Snail', 'Spider', 'Worm', 'Baby', 'Boy', 'Girl', 'Man'\
                       , 'Woman', 'Crocodile', 'Dinosaur', 'Lizard', 'Snake', 'Turtle', 'Hamster', 'Mouse', 'Rabbit', 'Shrew', 'Squirrel', 'Maple tree', 'Oak tree', 'Palm tree', 'Pine tree', 'Willow tree', 'Bicycle', 'Bus', 'Motorcycle', 'Pickup truck', 'Train', 'Lawn mower', 'Rocket', 'Streetcar', 'Tank', 'Tractor']
        class_names = ['Aquatic mammals', 'Fish', 'Flowers', 'Food containers', 'Fruit/Veggie', 'Electrical devices', \
                            'Furniture', 'Insects', 'Carnivores', 'Man-made things',  'Natural scenes', 'Omni/Herbivores', \
                            'Medium mammals', 'Invertebrates', 'People', 'Reptiles', 'Small mammals',   'Trees', 'Vehicles1', 'Vehicles2', 'Beaver', 'Dolphin', 'Otter', 'Seal', 'Whale', \
                            'Aquarium fish', 'Flatfish', 'Ray', 'Shark', 'Trout', 'Orchid', 'Poppy', 'Rose',  'Sunflower', 'Tulip', 'Bottle', 'Bowl', \
                            'Can', 'Cup', 'Plate', 'Apple', 'Mushroom', 'Orange', 'Pear', 'Sweet pepper', 'Clock', 'Keyboard', 'Lamp', 'Telephone', 'Television', 'Bed', 'Chair', 'Couch', 'Table', 'Wardrobe', 'Bee', \
                            'Beetle', 'Butterfly', 'Caterpillar', 'Cockroach', 'Bear', 'Leopard', 'Lion', 'Tiger',  'Wolf', 'Bridge', 'Castle', 'House', 'Road', 'Skyscraper', 'Cloud', 'Forest', 'Mountain',  'Plain', \
                            'Sea', 'Camel', 'Cattle', 'Chimpanzee', 'Elephant', 'Kangaroo', 'Fox', 'Porcupine', 'Possum', 'Raccoon', 'Skunk', 'Crab', 'Lobster', 'Snail', 'Spider', 'Worm', 'Baby', 'Boy', 'Girl', 'Man' \
                            , 'Woman', 'Crocodile', 'Dinosaur', 'Lizard', 'Snake', 'Turtle', 'Hamster', 'Mouse', 'Rabbit', 'Shrew', 'Squirrel', 'Maple tree', 'Oak tree', 'Palm tree', 'Pine tree', 'Willow tree', 'Bicycle',
                            'Bus', 'Motorcycle', 'Pickup truck', 'Train', 'Lawn mower', 'Rocket', 'Streetcar', 'Tank', 'Tractor']
    elif data == 'cars':
        # Order ABC
        abc_class_names = ['Cab', 'Convertible', 'Coupe', 'Hatchback', 'Minivan', 'Sedan', 'SUV', 'Van', 'Wagon', 'AM General Hummer SUV', 'Acura Integra Type R', 'Acura RL Sedan', 'Acura TL Sedan', \
                        'Acura TL Type-S', 'Acura TSX Sedan', 'Acura ZDX Hatchback', 'Aston Martin V8 Vantage Convertible', 'Aston Martin V8 Vantage Coupe', 'Aston Martin Virage Convertible', \
                        'Aston Martin Virage Coupe', 'Audi 100 Sedan', 'Audi 100 Wagon', 'Audi A5 Coupe', 'Audi R8 Coupe', 'Audi RS 4 Convertible', 'Audi S4 Sedan 2007', 'Audi S4 Sedan 2012', \
                        'Audi S5 Convertible', 'Audi S5 Coupe', 'Audi S6 Sedan', 'Audi TTS Coupe', 'Audi TT Hatchback', 'Audi TT RS Coupe', 'Audi V8 Sedan', 'BMW 1 Series Convertible',\
                        'BMW 1 Series Coupe', 'BMW 3 Series Sedan', 'BMW 3 Series Wagon', 'BMW 6 Series Convertible', 'BMW ActiveHybrid 5 Sedan', 'BMW M3 Coupe', 'BMW M5 Sedan', 'BMW M6 Convertible',\
                        'BMW X3 SUV', 'BMW X5 SUV', 'BMW X6 SUV', 'BMW Z4 Convertible', 'Bentley Arnage Sedan', 'Bentley Continental Flying Spur Sedan', 'Bentley Continental GT Coupe 2007', \
                        'Bentley Continental GT Coupe 2012', 'Bentley Continental Supersports Conv. Convertible', 'Bentley Mulsanne Sedan', 'Bugatti Veyron 16.4 Convertible', 'Bugatti Veyron 16.4 Coupe',\
                        'Buick Enclave SUV', 'Buick Rainier SUV', 'Buick Regal GS', 'Buick Verano Sedan', 'Cadillac CTS-V Sedan', 'Cadillac Escalade EXT Crew Cab', 'Cadillac SRX SUV', \
                        'Chevrolet Avalanche Crew Cab', 'Chevrolet Camaro Convertible', 'Chevrolet Cobalt SS', 'Chevrolet Corvette Convertible', 'Chevrolet Corvette Ron Fellows Edition Z06', \
                        'Chevrolet Corvette ZR1', 'Chevrolet Express Cargo Van', 'Chevrolet Express Van', 'Chevrolet HHR SS', 'Chevrolet Impala Sedan', 'Chevrolet Malibu Hybrid Sedan', \
                        'Chevrolet Malibu Sedan', 'Chevrolet Monte Carlo Coupe', 'Chevrolet Silverado 1500 Classic Extended Cab', 'Chevrolet Silverado 1500 Extended Cab', \
                        'Chevrolet Silverado 1500 Hybrid Crew Cab',  'Chevrolet Silverado 1500 Regular Cab', 'Chevrolet Silverado 2500HD Regular Cab', 'Chevrolet Sonic Sedan', 'Chevrolet Tahoe Hybrid SUV',\
                        'Chevrolet TrailBlazer SS', 'Chevrolet Traverse SUV', 'Chrysler 300 SRT-8', 'Chrysler Aspen SUV', 'Chrysler Crossfire Convertible', 'Chrysler PT Cruiser Convertible', \
                        'Chrysler Sebring Convertible', 'Chrysler Town and Country Minivan', 'Daewoo Nubira Wagon', 'Dodge Caliber Wagon 2007', 'Dodge Caliber Wagon 2012', 'Dodge Caravan Minivan',\
                        'Dodge Challenger SRT8', 'Dodge Charger SRT-8', 'Dodge Charger Sedan', 'Dodge Dakota Club Cab', 'Dodge Dakota Crew Cab', 'Dodge Durango SUV 2007', 'Dodge Durango SUV 2012', \
                        'Dodge Journey SUV', 'Dodge Magnum Wagon', 'Dodge Ram Pickup 3500 Crew Cab', 'Dodge Ram Pickup 3500 Quad Cab', 'Dodge Sprinter Cargo Van', 'Eagle Talon Hatchback', 'FIAT 500 Abarth',\
                        'FIAT 500 Convertible', 'Ferrari 458 Italia Convertible', 'Ferrari 458 Italia Coupe', 'Ferrari California Convertible', 'Ferrari FF Coupe', 'Fisker Karma Sedan', 'Ford E-Series Wagon Van',\
                        'Ford Edge SUV', 'Ford Expedition EL SUV', 'Ford F-150 Regular Cab 2007', 'Ford F-150 Regular Cab 2012', 'Ford F-450 Super Duty Crew Cab', 'Ford Fiesta Sedan', 'Ford Focus Sedan',\
                        'Ford Freestar Minivan', 'Ford GT Coupe', 'Ford Mustang Convertible', 'Ford Ranger SuperCab', 'GMC AcadiaSUV', 'GMC Canyon Extended Cab', 'GMC Savana Van', 'GMC Terrain SUV', \
                        'GMC Yukon Hybrid SUV', 'Geo Metro Convertible', 'HUMMER H2 SUT Crew Cab', 'HUMMER H3T Crew Cab', 'Honda Accord Coupe', 'Honda Accord Sedan', 'Honda Odyssey Minivan 2007', \
                        'Honda Odyssey Minivan 2012', 'Hyundai Accent Sedan', 'Hyundai Azera Sedan', 'Hyundai Elantra Sedan', 'Hyundai Elantra Touring Hatchback', 'Hyundai Genesis Sedan', 'Hyundai Santa Fe SUV',\
                        'Hyundai Sonata Hybrid Sedan', 'Hyundai Sonata Sedan', 'Hyundai Tucson SUV', 'Hyundai Veloster Hatchback', 'Hyundai Veracruz SUV', 'Infiniti G Coupe IPL', 'Infiniti QX56 SUV', \
                        'Isuzu Ascender SUV', 'Jaguar XK XKR', 'Jeep Compass SUV', 'Jeep Grand Cherokee SUV', 'Jeep Liberty SUV', 'Jeep Patriot SUV', 'Jeep Wrangler SUV', 'Lamborghini Aventador Coupe',\
                        'Lamborghini Diablo Coupe', 'Lamborghini Gallardo LP 570-4 Superleggera', 'Lamborghini Reventon Coupe', 'Land Rover LR2 SUV', 'Land Rover Range Rover SUV', 'Lincoln Town Car Sedan',\
                        'MINI Cooper Roadster Convertible', 'Maybach Landaulet Convertible', 'Mazda Tribute SUV', 'McLaren MP4-12C Coupe', 'Mercedes-Benz 300-Class Convertible', 'Mercedes-Benz C-Class Sedan',\
                        'Mercedes-Benz E-Class Sedan', 'Mercedes-Benz S-Class Sedan', 'Mercedes-Benz SL-Class Coupe', 'Mercedes-Benz Sprinter Van', 'Mitsubishi Lancer Sedan', 'Nissan 240SX Coupe', \
                        'Nissan Juke Hatchback', 'Nissan Leaf Hatchback', 'Nissan NV Passenger Van', 'Plymouth Neon Coupe', 'Porsche Panamera Sedan', 'Ram CV Cargo Van Minivan', 'Rolls-Royce Ghost Sedan', \
                        'Rolls-Royce Phantom Drophead Coupe Convertible', 'Rolls-Royce Phantom Sedan', 'Scion xD Hatchback', 'Spyker C8 Convertible', 'Spyker C8 Coupe', 'Suzuki Aerio Sedan', 'Suzuki Kizashi Sedan',\
                        'Suzuki SX4 Hatchback', 'Suzuki SX4 Sedan', 'Tesla Model S Sedan', 'Toyota 4Runner SUV', 'Toyota Camry Sedan', 'Toyota Corolla Sedan', 'Toyota Sequoia SUV', 'Volkswagen Beetle Hatchback', \
                        'Volkswagen Golf Hatchback 1991', 'Volkswagen Golf Hatchback 2012', 'Volvo 240 Sedan', 'Volvo C30 Hatchback', 'Volvo XC90 SUV', 'smart fortwo Convertible']
        long_class_names = ['Cab', 'Convertible', 'Coupe', 'Hatchback', 'Minivan', 'Sedan', 'SUV', 'Van', 'Wagon', 'Cadillac Escalade EXT Crew Cab 2007', 'Chevrolet Silverado 1500 Hybrid Crew Cab 2012', 'Chevrolet Avalanche Crew Cab 2012', 'Chevrolet Silverado 2500HD Regular Cab 2012',
                       'Chevrolet Silverado 1500 Classic Extended Cab 2007', 'Chevrolet Silverado 1500 Extended Cab 2012', 'Chevrolet Silverado 1500 Regular Cab 2012', 'Dodge Ram Pickup 3500 Crew Cab 2010', 'Dodge Ram Pickup 3500 Quad Cab 2009', 'Dodge Dakota Crew Cab 2010',
                       'Dodge Dakota Club Cab 2007', 'Ford F-450 Super Duty Crew Cab 2012', 'Ford Ranger SuperCab 2011', 'Ford F-150 Regular Cab 2012', 'Ford F-150 Regular Cab 2007', 'GMC Canyon Extended Cab 2012', 'HUMMER H3T Crew Cab 2010', 'HUMMER H2 SUT Crew Cab 2009',
                       'Aston Martin V8 Vantage Convertible 2012', 'Aston Martin Virage Convertible 2012', 'Audi RS 4 Convertible 2008', 'Audi S5 Convertible 2012', 'BMW 1 Series Convertible 2012', 'BMW 6 Series Convertible 2007', 'BMW M6 Convertible 2010', 'BMW Z4 Convertible 2012',
                       'Bentley Continental Supersports Conv. Convertible 2012', 'Bugatti Veyron 16.4 Convertible 2009', 'Chevrolet Corvette Convertible 2012', 'Chevrolet Camaro Convertible 2012', 'Chrysler Sebring Convertible 2010', 'Chrysler Crossfire Convertible 2008', 'Chrysler PT Cruiser Convertible 2008',
                       'FIAT 500 Convertible 2012', 'Ferrari California Convertible 2012', 'Ferrari 458 Italia Convertible 2012', 'Ford Mustang Convertible 2007', 'Geo Metro Convertible 1993', 'MINI Cooper Roadster Convertible 2012', 'Maybach Landaulet Convertible 2012',
                       'Mercedes-Benz 300-Class Convertible 1993', 'Rolls-Royce Phantom Drophead Coupe Convertible 2012', 'Spyker C8 Convertible 2009', 'smart fortwo Convertible 2012', 'Acura Integra Type R 2001', 'Aston Martin V8 Vantage Coupe 2012', 'Aston Martin Virage Coupe 2012', 'Audi A5 Coupe 2012',
                       'Audi TTS Coupe 2012', 'Audi R8 Coupe 2012', 'Audi S5 Coupe 2012', 'Audi TT RS Coupe 2012', 'BMW 1 Series Coupe 2012', 'BMW M3 Coupe 2012', 'Bentley Continental GT Coupe 2012', 'Bentley Continental GT Coupe 2007', 'Bugatti Veyron 16.4 Coupe 2009',
                       'Chevrolet Corvette ZR1 2012', 'Chevrolet Corvette Ron Fellows Edition Z06 2007', 'Chevrolet Cobalt SS 2010', 'Chevrolet Monte Carlo Coupe 2007', 'Dodge Challenger SRT8 2011', 'FIAT 500 Abarth 2012', 'Ferrari FF Coupe 2012', 'Ferrari 458 Italia Coupe 2012', 'Ford GT Coupe 2006',
                       'Honda Accord Coupe 2012', 'Infiniti G Coupe IPL 2012', 'Jaguar XK XKR 2012', 'Lamborghini Reventon Coupe 2008', 'Lamborghini Aventador Coupe 2012', 'Lamborghini Gallardo LP 570-4 Superleggera 2012', 'Lamborghini Diablo Coupe 2001', 'McLaren MP4-12C Coupe 2012',
                       'Mercedes-Benz SL-Class Coupe 2009', 'Nissan 240SX Coupe 1998', 'Plymouth Neon Coupe 1999', 'Spyker C8 Coupe 2009', 'Acura ZDX Hatchback 2012', 'Audi TT Hatchback 2011', 'Eagle Talon Hatchback 1998', 'Hyundai Veloster Hatchback 2012', 'Hyundai Elantra Touring Hatchback 2012',
                       'Nissan Leaf Hatchback 2012', 'Nissan Juke Hatchback 2012', 'Scion xD Hatchback 2012', 'Suzuki SX4 Hatchback 2012', 'Volkswagen Golf Hatchback 2012', 'Volkswagen Golf Hatchback 1991', 'Volkswagen Beetle Hatchback 2012', 'Volvo C30 Hatchback 2012',
                       'Chevrolet HHR SS 2010', 'Chrysler Town and Country Minivan 2012', 'Dodge Caravan Minivan 1997', 'Ford Freestar Minivan 2007', 'Honda Odyssey Minivan 2012', 'Honda Odyssey Minivan 2007', 'Ram CV Cargo Van Minivan 2012', 'Acura RL Sedan 2012', 'Acura TL Sedan 2012',
                       'Acura TL Type-S 2008', 'Acura TSX Sedan 2012', 'Audi V8 Sedan 1994', 'Audi 100 Sedan 1994', 'Audi S6 Sedan 2011', 'Audi S4 Sedan 2012', 'Audi S4 Sedan 2007', 'BMW ActiveHybrid 5 Sedan 2012', 'BMW 3 Series Sedan 2012', 'BMW M5 Sedan 2010', 'Bentley Arnage Sedan 2009',
                       'Bentley Mulsanne Sedan 2011', 'Bentley Continental Flying Spur Sedan 2007', 'Buick Regal GS 2012', 'Buick Verano Sedan 2012', 'Cadillac CTS-V Sedan 2012', 'Chevrolet Impala Sedan 2007', 'Chevrolet Sonic Sedan 2012', 'Chevrolet Malibu Hybrid Sedan 2010',
                       'Chevrolet Malibu Sedan 2007', 'Chrysler 300 SRT-8 2010', 'Dodge Charger Sedan 2012', 'Dodge Charger SRT-8 2009', 'Fisker Karma Sedan 2012', 'Ford Focus Sedan 2007', 'Ford Fiesta Sedan 2012', 'Honda Accord Sedan 2012', 'Hyundai Sonata Hybrid Sedan 2012',
                       'Hyundai Elantra Sedan 2007', 'Hyundai Accent Sedan 2012', 'Hyundai Genesis Sedan 2012', 'Hyundai Sonata Sedan 2012', 'Hyundai Azera Sedan 2012', 'Lincoln Town Car Sedan 2011', 'Mercedes-Benz C-Class Sedan 2012', 'Mercedes-Benz E-Class Sedan 2012', 'Mercedes-Benz S-Class Sedan 2012',
                       'Mitsubishi Lancer Sedan 2012', 'Porsche Panamera Sedan 2012', 'Rolls-Royce Ghost Sedan 2012', 'Rolls-Royce Phantom Sedan 2012', 'Suzuki Aerio Sedan 2007', 'Suzuki Kizashi Sedan 2012', 'Suzuki SX4 Sedan 2012', 'Tesla Model S Sedan 2012', 'Toyota Camry Sedan 2012',
                       'Toyota Corolla Sedan 2012', 'Volvo 240 Sedan 1993', 'AM General Hummer SUV 2000', 'BMW X5 SUV 2007', 'BMW X6 SUV 2012', 'BMW X3 SUV 2012', 'Buick Rainier SUV 2007', 'Buick Enclave SUV 2012', 'Cadillac SRX SUV 2012', 'Chevrolet Traverse SUV 2012',
                       'Chevrolet Tahoe Hybrid SUV 2012', 'Chevrolet TrailBlazer SS 2009', 'Chrysler Aspen SUV 2009', 'Dodge Journey SUV 2012', 'Dodge Durango SUV 2012', 'Dodge Durango SUV 2007', 'Ford Expedition EL SUV 2009', 'Ford Edge SUV 2012', 'GMC Terrain SUV 2012', 'GMC Yukon Hybrid SUV 2012',
                       'GMC Acadia SUV 2012', 'Hyundai Santa Fe SUV 2012', 'Hyundai Tucson SUV 2012', 'Hyundai Veracruz SUV 2012', 'Infiniti QX56 SUV 2011', 'Isuzu Ascender SUV 2008', 'Jeep Patriot SUV 2012', 'Jeep Wrangler SUV 2012', 'Jeep Liberty SUV 2012', 'Jeep Grand Cherokee SUV 2012',
                       'Jeep Compass SUV 2012', 'Land Rover Range Rover SUV 2012', 'Land Rover LR2 SUV 2012', 'Mazda Tribute SUV 2011', 'Toyota Sequoia SUV 2012', 'Toyota 4Runner SUV 2012', 'Volvo XC90 SUV 2007', 'Chevrolet Express Cargo Van 2007', 'Chevrolet Express Van 2007',
                       'Dodge Sprinter Cargo Van 2009', 'GMC Savana Van 2012', 'Mercedes-Benz Sprinter Van 2012', 'Nissan NV Passenger Van 2012', 'Audi 100 Wagon 1994', 'BMW 3 Series Wagon 2012', 'Daewoo Nubira Wagon 2002', 'Dodge Caliber Wagon 2012', 'Dodge Caliber Wagon 2007',
                       'Dodge Magnum Wagon 2008', 'Ford E-Series Wagon Van 2012']
        class_names = ['Cab', 'Convertible', 'Coupe', 'Hatchback', 'Minivan', 'Sedan', 'SUV', 'Van', 'Wagon', 'Cadillac Escalade EXT Crew Cab', 'Chevrolet Silverado 1500 Hybrid Crew Cab',
                       'Chevrolet Avalanche Crew Cab', 'Chevrolet Silverado 2500HD Regular Cab', 'Chevrolet Silverado 1500 Classic Extended Cab', 'Chevrolet Silverado 1500 Extended Cab',
                       'Chevrolet Silverado 1500 Regular Cab', 'Dodge Ram Pickup 3500 Crew Cab', 'Dodge Ram Pickup 3500 Quad Cab', 'Dodge Dakota Crew Cab', 'Dodge Dakota Club Cab', 'Ford F-450 Super Duty Crew Cab',
                       'Ford Ranger SuperCab', 'Ford F-150 Regular Cab 2012', 'Ford F-150 Regular Cab 2007', 'GMC Canyon Extended Cab', 'HUMMER H3T Crew Cab', 'HUMMER H2 SUT Crew Cab', 'Aston Martin V8 Vantage Convertible',
                       'Aston Martin Virage Convertible', 'Audi RS 4 Convertible', 'Audi S5 Convertible', 'BMW 1 Series Convertible', 'BMW 6 Series Convertible', 'BMW M6 Convertible', 'BMW Z4 Convertible',
                       'Bentley Continental Supersports Conv. Convertible', 'Bugatti Veyron 16.4 Convertible', 'Chevrolet Corvette Convertible', 'Chevrolet Camaro Convertible','Chrysler Sebring Convertible',
                       'Chrysler Crossfire Convertible', 'Chrysler PT Cruiser Convertible', 'FIAT 500 Convertible', 'Ferrari California Convertible', 'Ferrari 458 Italia Convertible', 'Ford Mustang Convertible',
                       'Geo Metro Convertible', 'MINI Cooper Roadster Convertible', 'Maybach Landaulet Convertible', 'Mercedes-Benz 300-Class Convertible', 'Rolls-Royce Phantom Drophead Coupe Convertible',
                       'Spyker C8 Convertible', 'smart fortwo Convertible', 'Acura Integra Type R', 'Aston Martin V8 Vantage Coupe', 'Aston Martin Virage Coupe', 'Audi A5 Coupe', 'Audi TTS Coupe', 'Audi R8 Coupe',
                       'Audi S5 Coupe', 'Audi TT RS Coupe', 'BMW 1 Series Coupe', 'BMW M3 Coupe', 'Bentley Continental GT Coupe 2012', 'Bentley Continental GT Coupe 2007', 'Bugatti Veyron 16.4 Coupe',
                       'Chevrolet Corvette ZR1', 'Chevrolet Corvette Ron Fellows Edition Z06', 'Chevrolet Cobalt SS', 'Chevrolet Monte Carlo Coupe', 'Dodge Challenger SRT8', 'FIAT 500 Abarth', 'Ferrari FF Coupe',
                       'Ferrari 458 Italia Coupe', 'Ford GT Coupe', 'Honda Accord Coupe', 'Infiniti G Coupe IPL', 'Jaguar XK XKR', 'Lamborghini Reventon Coupe', 'Lamborghini Aventador Coupe',
                       'Lamborghini Gallardo LP 570-4 Superleggera', 'Lamborghini Diablo Coupe', 'McLaren MP4-12C Coupe', 'Mercedes-Benz SL-Class Coupe', 'Nissan 240SX Coupe', 'Plymouth Neon Coupe', 'Spyker C8 Coupe',
                       'Acura ZDX Hatchback', 'Audi TT Hatchback', 'Eagle Talon Hatchback', 'Hyundai Veloster Hatchback', 'Hyundai Elantra Touring Hatchback', 'Nissan Leaf Hatchback', 'Nissan Juke Hatchback',
                       'Scion xD Hatchback', 'Suzuki SX4 Hatchback', 'Volkswagen Golf Hatchback 2012', 'Volkswagen Golf Hatchback 1991', 'Volkswagen Beetle Hatchback', 'Volvo C30 Hatchback', 'Chevrolet HHR SS',
                       'Chrysler Town and Country Minivan', 'Dodge Caravan Minivan', 'Ford Freestar Minivan', 'Honda Odyssey Minivan 2012', 'Honda Odyssey Minivan 2007', 'Ram CV Cargo Van Minivan', 'Acura RL Sedan',
                       'Acura TL Sedan', 'Acura TL Type-S', 'Acura TSX Sedan', 'Audi V8 Sedan', 'Audi 100 Sedan', 'Audi S6 Sedan', 'Audi S4 Sedan 2012', 'Audi S4 Sedan 2007', 'BMW ActiveHybrid 5 Sedan',
                       'BMW 3 Series Sedan', 'BMW M5 Sedan', 'Bentley Arnage Sedan', 'Bentley Mulsanne Sedan', 'Bentley Continental Flying Spur Sedan', 'Buick Regal GS', 'Buick Verano Sedan', 'Cadillac CTS-V Sedan',
                       'Chevrolet Impala Sedan', 'Chevrolet Sonic Sedan', 'Chevrolet Malibu Hybrid Sedan', 'Chevrolet Malibu Sedan', 'Chrysler 300 SRT-8', 'Dodge Charger Sedan', 'Dodge Charger SRT-8', 'Fisker Karma Sedan',
                       'Ford Focus Sedan', 'Ford Fiesta Sedan', 'Honda Accord Sedan', 'Hyundai Sonata Hybrid Sedan', 'Hyundai Elantra Sedan', 'Hyundai Accent Sedan', 'Hyundai Genesis Sedan', 'Hyundai Sonata Sedan',
                       'Hyundai Azera Sedan', 'Lincoln Town Car Sedan', 'Mercedes-Benz C-Class Sedan', 'Mercedes-Benz E-Class Sedan', 'Mercedes-Benz S-Class Sedan', 'Mitsubishi Lancer Sedan', 'Porsche Panamera Sedan',
                       'Rolls-Royce Ghost Sedan', 'Rolls-Royce Phantom Sedan', 'Suzuki Aerio Sedan', 'Suzuki Kizashi Sedan', 'Suzuki SX4 Sedan', 'Tesla Model S Sedan', 'Toyota Camry Sedan', 'Toyota Corolla Sedan',
                       'Volvo 240 Sedan', 'AM General Hummer SUV', 'BMW X5 SUV', 'BMW X6 SUV', 'BMW X3 SUV', 'Buick Rainier SUV', 'Buick Enclave SUV', 'Cadillac SRX SUV', 'Chevrolet Traverse SUV',
                       'Chevrolet Tahoe Hybrid SUV', 'Chevrolet TrailBlazer SS', 'Chrysler Aspen SUV', 'Dodge Journey SUV', 'Dodge Durango SUV 2012', 'Dodge Durango SUV 2007', 'Ford Expedition EL SUV', 'Ford Edge SUV',
                       'GMC Terrain SUV', 'GMC YukonHybrid SUV', 'GMC Acadia SUV', 'Hyundai Santa Fe SUV', 'Hyundai Tucson SUV', 'Hyundai Veracruz SUV', 'Infiniti QX56 SUV', 'Isuzu Ascender SUV', 'Jeep Patriot SUV',
                       'Jeep Wrangler SUV', 'Jeep Liberty SUV', 'Jeep Grand Cherokee SUV', 'Jeep Compass SUV', 'Land Rover Range Rover SUV', 'Land Rover LR2 SUV', 'Mazda Tribute SUV', 'Toyota Sequoia SUV',
                       'Toyota 4Runner SUV', 'Volvo XC90 SUV', 'Chevrolet Express Cargo Van', 'Chevrolet Express Van', 'Dodge Sprinter Cargo Van', 'GMC Savana Van', 'Mercedes-Benz Sprinter Van', 'Nissan NVPassenger Van',
                       'Audi 100 Wagon', 'BMW 3 Series Wagon', 'Daewoo Nubira Wagon', 'Dodge Caliber Wagon 2012', 'Dodge Caliber Wagon 2007', 'Dodge Magnum Wagon', 'Ford E-Series Wagon Van']
    elif data == 'cub' or data == 'cub2':
        class_names = ['Anseriformes', 'Apodiformes', 'Caprimulgiformes', 'Charadriiformes', 'Coraciiformes', 'Cuculiformes', 'Gaviiformes', 'Passeriformes', 'Pelecaniformes', 'Piciformes', 'Podicipediformes', 'Procellariiformes', 'Suliformes', 'Anatidae', 'Trochilidae', 'Caprimulgidae',\
                       'Alcidae', 'Laridae', 'Stercorariidae', 'Alcedinidae', 'Cuculidae', 'Gaviidae', 'Alaudidae', 'Bombycillidae', 'Cardinalidae', 'Certhiidae', 'Corvidae', 'Fringillidae', 'Hirundinidae', 'Icteridae', 'Icteriidae', 'Laniidae', 'Mimidae', 'Motacillidae', 'Parulidae',\
                       'Passerellidae', 'Passeridae', 'Ptilonorhynchidae', 'Sittidae', 'Sturnidae', 'Troglodytidae', 'Tyrannidae', 'Vireonidae', 'Pelecanidae', 'Picidae', 'Podicipedidae', 'Diomedeida', 'Procellariidae', 'Phalacrocoracidae', 'Fregatidae', 'Black-footed Albatross', 'Laysan Albatross',\
                       'Sooty Albatross', 'Groove-billed Ani', 'Crested Auklet', 'Least Auklet', 'Parakeet Auklet', 'Rhinoceros Auklet', 'Brewer Blackbird', 'Red-winged Blackbird', 'Rusty Blackbird', 'Yellow-headed Blackbird', 'Bobolink', 'Indigo Bunting', 'Lazuli Bunting','Painted Bunting', \
                       'Cardinal', 'Spotted Catbird', 'Gray Catbird', 'Yellow-breasted Chat', 'Eastern Towhee', "Chuck-will's-widow", 'Brandt Cormorant', 'Red-faced Cormorant', 'Pelagic Cormorant', 'Bronzed Cowbird', 'Shiny Cowbird', 'Brown Creeper', 'American Crow', 'Fish Crow',\
                       'Black-billed Cuckoo', 'Mangrove Cuckoo', 'Yellow-billed Cuckoo', 'Gray-crowned rosy finch', 'Purple Finch', 'Northern Flicker', 'Acadian Flycatcher', 'Great Crested Flycatcher', 'Least Flycatcher', 'Olive-sided Flycatcher', 'Scissor-tailed Flycatcher',  \
                       'Vermilion Flycatcher', 'Yellow-bellied Flycatcher', 'Frigatebird', 'Northern Fulmar', 'Gadwall', 'American Goldfinch', 'European Goldfinch', 'Boat-tailed Grackle', 'Eared Grebe', 'Horned Grebe', 'Pied-billed Grebe', 'Western Grebe', 'Blue Grosbeak', 'Evening Grosbeak',\
                       'Pine Grosbeak', 'Rose-breasted Grosbeak', 'Pigeon Guillemot', 'California Gull', 'Glaucous-winged Gull', 'Heermann Gull', 'Herring Gull', 'Ivory Gull', 'Ring-billed Gull', 'Slaty-backed Gull', 'Western Gull', 'Anna Hummingbird', 'Ruby-throated Hummingbird',  'Rufous Hummingbird',\
                       'Green Violetear', 'Long-tailed Jaeger', 'Pomarine Jaeger', 'Blue Jay', 'Florida Jay', 'Green Jay', 'Dark eyed Junco', 'Tropical Kingbird', 'Gray Kingbird', 'Belted Kingfisher', 'Green Kingfisher', 'Pied Kingfisher', 'Ringed Kingfisher', 'White-breasted Kingfisher',\
                       'Red-legged Kittiwake', 'Horned Lark', 'Pacific Loon', 'Mallard', 'Western Meadowlark', 'Hooded Merganser', 'Red-breasted Merganser', 'Mockingbird', 'Nighthawk', 'Clark Nutcracker', 'White breasted Nuthatch', 'Baltimore Oriole', 'Hooded Oriole', 'Orchard Oriole',\
                       'Scott Oriole', 'Ovenbird', 'Brown Pelican', 'White Pelican', 'Western Wood Pewee', 'Sayornis', 'American Pipit', 'Eastern whip-poor-will', 'Horned Puffin', 'Common Raven', 'White-necked Raven', 'American Redstart', 'Geococcyx', 'Loggerhead Shrike', 'Great Grey Shrike',\
                       'Baird Sparrow', 'Black-throated Sparrow', 'Brewer Sparrow', 'Chipping Sparrow', 'Clay-colored Sparrow', 'House Sparrow', 'Field Sparrow', 'Fox Sparrow', 'Grasshopper Sparrow', 'Harris Sparrow', 'Henslow Sparrow', "LeConte's sparrow", 'Lincoln Sparrow',\
                       "Nelson's sparrow", 'Savannah Sparrow', 'Seaside Sparrow', 'Song Sparrow', 'Tree Sparrow', 'Vesper Sparrow', 'White crowned Sparrow', 'White-throated Sparrow', 'Cape Glossy Starling', 'Bank Swallow', 'Barn Swallow', 'Cliff Swallow', 'Tree Swallow', 'Scarlet Tanager', \
                       'Summer Tanager', 'Artic Tern', 'Black Tern', 'Caspian Tern', 'Common Tern', 'Elegant Tern', 'Forsters Tern', 'Least Tern', 'Green-tailed Towhee', 'Brown Thrasher', 'Sage Thrasher', 'Black-capped Vireo', 'Blue-headed Vireo', 'Philadelphia Vireo', 'Red-eyed Vireo', \
                       'Warbling Vireo', 'White-eyed Vireo', 'Yellow-throated Vireo', 'Bay-breasted Warbler', 'Black and white Warbler', 'Black-throated blue warbler', 'Blue-winged Warbler', 'Canada Warbler', 'Cape May Warbler', 'Cerulean Warbler', 'Chestnut-sided Warbler', 'Golden-winged Warbler',\
                       'Hooded Warbler', 'Kentucky Warbler', 'Magnolia Warbler', 'Mourning Warbler', 'Myrtle Warbler', 'Nashville Warbler', 'Orange-crowned Warbler', 'Palm Warbler', 'Pine Warbler', 'Prairie Warbler', 'Prothonotary Warbler', 'Swainson Warbler', 'Tennessee Warbler', 'Wilson Warbler',\
                       'Worm eating Warbler', 'Yellow Warbler', 'Northern Waterthrush', 'Louisiana Waterthrush', 'Bohemian Waxwing', 'Cedar Waxwing', 'American three-toed woodpecker', 'Pileated Woodpecker', 'Red-bellied Woodpecker', 'Red-cockaded Woodpecker', 'Red-headed Woodpecker', 'Downy Woodpecker',\
                       'Bewick Wren', 'Cactus Wren', 'Carolina Wren', 'House Wren', 'Marsh Wren', 'Rock Wren', 'Winter Wren', 'Common Yellowthroat']
    elif data == 'imclef07a':
        class_names = ['2', '3', '4', '5', '6', '7', '8', '9', '2.0', '2.1', '2.3', '3.1', '3.2', '3.3', '4.1', '4.2', '4.3', '4.4', '4.5', '4.6', '5.0', '5.1', '6.1', '6.2', '7.0', '7.1', '8.0', '9.1', '9.2', '9.3', '9.4', '9.5', '9.6', '2.0.0', '2.1.2', '2.1.3', '2.1.5', '2.1.6', '2.3.0', '2.3.3', '3.1.0', '3.1.1', '3.2.0', '3.3.0', '3.3.1', '3.3.3', '4.1.1', '4.1.3', '4.1.4', '4.1.5', '4.2.1', '4.2.2', '4.3.3', '4.3.4', '4.3.7', '4.3.8', '4.4.1', '4.4.2', '4.5.1', '4.5.2', '4.5.4', '4.6.2', '4.6.3', '4.6.6', '4.6.7', '5.0.0', '5.1.4', '5.1.5',
                        '5.1.6', '5.1.7', '6.1.0', '6.2.0', '7.0.0', '7.1.0', '8.0.0', '9.1.1', '9.1.4', '9.1.5', '9.1.7', '9.1.8', '9.1.9', '9.1.a', '9.2.1', '9.2.2', '9.3.0', '9.3.3', '9.3.4', '9.4.1', '9.4.2', '9.4.3', '9.5.0', '9.5.1', '9.5.3', '9.5.6', '9.6.1', '9.6.2']
    elif data == 'imclef07d':
        class_names = ['1', '2', '3', '4', '1.1', '1.2', '2.0', '2.1', '2.2', '2.3', '2.4', '3.1', '3.2', '4.1', '4.2', '4.3', '4.6', '4.9', '4.a', '4.b', '1.1.0', '1.1.2', '1.1.5', '1.1.6', '1.2.0', '1.2.1', '1.2.7', '1.2.9', '1.2.f', '2.0.0', '2.1.0', '2.1.1', '2.2.0', '2.2.8', '2.2.9', '2.3.0', '2.4.0', '3.1.0', '3.2.0', '4.1.0', '4.2.0', '4.3.0', '4.6.0', '4.9.0', '4.a.0', '4.b.0']
    else:
        print('NO CLASS NAME')
    return class_names

def get_edge_metric(n_correct, n_wrong, n_true):
    try:
        edge_rec = n_correct / n_true * 100
        edge_prec = n_correct / (n_correct + n_wrong) * 100
        edge_f1 = 2 * 1 / (1/edge_rec + 1/edge_prec)
    except:
        edge_rec, edge_prec, edge_f1 = 0, 0, 0
    return edge_rec, edge_prec, edge_f1

def save_edge_info(edges, epoch, P, edge_path=None, save=True):
    correct, wrong, true_edges = get_edge_set(edges, P['dataset'])
    edge_rec, edge_prec, edge_f1 = get_edge_metric(len(correct), len(wrong), len(true_edges))

    if edge_path == None:
        edge_file_name = 'edges_{}_epoch{}_{}|{}_rec{:.1f}|pre{:.1f}.txt'.format(P['flag'], epoch, len(correct), len(edges), edge_rec, edge_prec)
        edge_path = ospj(P['save_dir'], edge_file_name)
        if not torch.cuda.is_available():
            edge_file_name = 'edge_{}_epoch{}.txt'.format(P['flag'], epoch)
            edge_path = ospj(P['save_dir'], edge_file_name)

    with open(edge_path , 'a') as f:
        f.write('Epoch{} | Correct: {} Wrong: {} \n'.format(epoch, len(correct), len(wrong)))
        f.write('---------------------------------\n')
        f.write('SRC, DES, WEIGHT, Corr:1/Wrong:0 \n')
        for c_edge in correct:
            src, des, weight = c_edge
            f.write('{}, {}, {}, {} \n'.format(src, des, weight, 1))
        for w_edge in wrong:
            src, des, weight = w_edge
            f.write('{}, {}, {}, {} \n'.format(src, des, weight, 0))
    return correct, wrong, edge_rec, edge_prec


# from typing import List, Tuple, Union, Dict, Any, Optional
# from pathlib import Path
# import json
# import jsonlines
#
# def smart_read(path: Path, **kwargs) -> List[Dict]:
#     data = []
#     if path.suffix == ".arff":
#         if kwargs['num_labels'] is None:
#             raise ValueError(
#                 "No. of labels is needed"
#                 f" to read an .arff file but is None"
#             )
#         reader = ARFFReader(num_labels=kwargs['num_labels'])
#         data = list(reader.read_internal(str(path.absolute())))
#     else:
#         with open(path) as f:
#             if path.suffix == ".json":
#                 data = json.load(f)
#             elif path.suffix == ".jsonl":
#                 data = list(jsonlines.Reader(f))
#             else:
#                 raise ValueError(
#                     f"file extension can only be .json/.jsonl/.arff but is {path.suffix}"
#                 )
#
#     return data
