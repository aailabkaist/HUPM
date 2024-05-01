from instrumentation import compute_metrics
from instrumentation import get_max_em
from util import *
import datasets
import models

def run_test(P, phase, best_th, model=None, use_tqdm=True):
    dataset = datasets.get_data(P)
    dataloader = {}
    dataloader[phase] = torch.utils.data.DataLoader(
        dataset[phase],
        batch_size = P['bsize'],
        shuffle = phase == 'train',
        sampler = None,
        num_workers = P['num_workers'],
        drop_last = False,
        pin_memory = True
    )

    role = (P['mod_scheme'] == 'role')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    if model == None:
        model = models.ImageClassifier(P)
        if role:
            model = models.MultilabelModel(P, dataset['train'].label_matrix_obs, None)
        model_path = ospj(P['checkpoint'], P['checkmodel'])
        model_state, _ = torch.load(model_path)
        model.load_state_dict(model_state)

    model.to(device)
    model.eval()
    y_pred = np.zeros((len(dataset[phase]), P['num_classes']))
    y_true = np.zeros((len(dataset[phase]), P['num_classes']))
    batch_stack = 0
    dataloader_loop = tqdm(dataloader[phase]) if use_tqdm else dataloader[phase]
    with torch.set_grad_enabled(False):
        for batch in dataloader_loop:
            # Move data to GPU
            image = batch['image'].to(device, non_blocking=True)
            label_vec_true = batch['label_vec_true'].clone().numpy()

            # Forward pass
            if role:
                try:
                    logits = model.f(image)
                except:
                    logits = model.f(image.float())
            else:
                try:
                    logits = model(image)
                except:
                    logits = model(image.float())
            if logits.dim() == 1:
                logits = torch.unsqueeze(logits, 0)
            preds = torch.sigmoid(logits)
            preds = get_constr_out(preds, P['R'].to(device)) if P['mod_scheme'] == 'hmcnn' else preds

            preds_np = preds.cpu().numpy()
            this_batch_size = preds_np.shape[0]
            y_pred[batch_stack : batch_stack+this_batch_size] = preds_np
            y_true[batch_stack : batch_stack+this_batch_size] = label_vec_true
            batch_stack += this_batch_size

    # SAVE([y_pred, y_true], ospj(P['save_path'], P['checkpoint'], 'test_inferenced_{}'.format(P['checkmodel'][:-3])))
    if P['HMC']:
        metrics = compute_metrics(y_pred, y_true, P)
        map_test, ems_test, cv_test, auprc_test = metrics['map'], metrics['ems'], metrics['cv'], metrics['au_prc']
        if best_th < 0:
            max_th_test, max_em_test = get_max_em(ems_test)
            return [map_test, max_em_test, max_th_test, cv_test, auprc_test]
        else:
            max_em_test = ems_test[best_th]
            return [map_test, max_em_test, best_th, cv_test, auprc_test]

    metrics = compute_metrics(y_pred, y_true, P)
    map_test, ems_test, cv_test = metrics['map'], metrics['ems'], metrics['cv']

    if P['test_leaf']:
        parent_cp = get_parent_dir(P['checkpoint'])
        result_path = ospj(parent_cp, "final_result.txt")  # SET DIRECTORY TO SAVE FILE
        save_and_print(result_path, "Best epoch {} leaf acc:  {:.3f}".format(phase, metrics['leaf_acc']))

    if best_th < 0:
        max_th_test, max_em_test = get_max_em(ems_test)
        return [map_test, max_em_test, max_th_test, cv_test]
    else:
        max_em_test = ems_test[best_th]
        return [map_test, max_em_test, best_th, cv_test]


def get_dist(P):
    dataset = datasets.get_data(P)
    dataloader = {}
    for phase in ['train']:
        dataloader[phase] = torch.utils.data.DataLoader(
            dataset[phase],
            batch_size = P['bsize'],
            shuffle = phase == 'train',
            sampler = None,
            num_workers = P['num_workers'],
            drop_last = False,
            pin_memory = True
        )

    role = (P['mod_scheme'] == 'role')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #Load model
    model = models.ImageClassifier(P)
    if role:
        model = models.MultilabelModel(P, dataset['train'].label_matrix_obs, None)
    model_path = ospj(P['checkpoint'], P['checkmodel'])
    model_state, _ = torch.load(model_path)


    #
    model.load_state_dict(model_state)

    model.to(device)
    model.eval()

    phase = 'train' # 'val'
    model.eval()
    y_pred = np.zeros((len(dataset[phase]), P['num_classes']))
    y_pred_TP = np.zeros((len(dataset[phase]), P['num_classes']))
    y_pred_FN = np.zeros((len(dataset[phase]), P['num_classes']))
    y_pred_TN = np.zeros((len(dataset[phase]), P['num_classes']))

    batch_stack = 0
    epoch_dist = []
    with torch.set_grad_enabled(False):
        for batch in tqdm(dataloader[phase]):
            # Move data to GPU
            image = batch['image'].to(device, non_blocking=True)
            label_vec_obs = batch['label_vec_obs'].to(device, non_blocking=True)
            label_vec_true = batch['label_vec_true']
            label_vec_hidden = label_vec_true.clone().numpy() - label_vec_obs.cpu().detach().numpy()
            label_vec_neg = (~label_vec_true.bool()).float().clone().numpy()

            # Forward pass
            if role:
                try:
                    logits = model.f(image)
                except:
                    logits = model.f(image.float())
            else:
                try:
                    logits = model(image)
                except:
                    logits = model(image.float())
            if logits.dim() == 1:
                logits = torch.unsqueeze(logits, 0)
            preds = torch.sigmoid(logits)
            preds = get_constr_out(preds, P['R'].to(device)) if P['mod_scheme'] == 'hmcnn' else preds

            preds_np = preds.cpu().detach().numpy()
            preds_TP = preds_np * label_vec_obs.cpu().detach().numpy()
            preds_FN = preds_np * label_vec_hidden
            preds_TN = preds_np * label_vec_neg

            this_batch_size = preds_np.shape[0]
            y_pred[batch_stack : batch_stack+this_batch_size] = preds_np
            y_pred_TP[batch_stack: batch_stack + this_batch_size] = preds_TP
            y_pred_FN[batch_stack: batch_stack + this_batch_size] = preds_FN
            y_pred_TN[batch_stack: batch_stack + this_batch_size] = preds_TN
            # y_true[batch_stack : batch_stack+this_batch_size] = label_vec_true
            batch_stack += this_batch_size

            # epoch_dist.extend(list(preds_FN[torch.where(preds_FN>0)].numpy()))


    result = {'FN':[], 'TP':[], 'TN':[]}
    result['FN'] = {'mean':y_pred_FN[y_pred_FN > 0].mean(), 'std':y_pred_FN[y_pred_FN > 0].std()}
    result['TP'] = {'mean':y_pred_TP[y_pred_TP > 0].mean(), 'std':y_pred_TP[y_pred_TP > 0].std()}
    result['TN'] = {'mean':y_pred_TN[y_pred_TN > 0].mean(), 'std':y_pred_TN[y_pred_TN > 0].std()}


    return result

def get_edges(P, edge_path=None):
    dataset = datasets.get_data(P)
    dataloader = {}
    for phase in ['train']:
        dataloader[phase] = torch.utils.data.DataLoader(
            dataset[phase],
            batch_size=P['bsize'],
            shuffle=phase == 'train',
            sampler=None,
            num_workers=P['num_workers'],
            drop_last=False,
            pin_memory=True
        )

    model = models.ImageClassifier(P)
    role = False
    if 'role' in P['checkpoint']:
        role = True
        model = models.MultilabelModel(P, dataset['train'].label_matrix_obs, None)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    path = ospj(P['checkpoint'], P['checkmodel'])
    model_state, _ = torch.load(path)
    model.load_state_dict(model_state)
    epoch = int(P['checkmodel'].split('_')[-1][:-3])

    train_label, train_output = [], []
    model.train()
    with torch.set_grad_enabled(False):
        for batch in tqdm(dataloader['train']):
            # Move data to GPU
            image = batch['image'].to(device, non_blocking=True)
            label_vec_obs = batch['label_vec_obs'].to(device, non_blocking=True)
            label_vec_true = batch['label_vec_true'].clone().numpy()
            idx = batch['idx']

            # Forward pass
            if role:
                logits = model.f(image)
            else:
                logits = model(image)
            if logits.dim() == 1:
                logits = torch.unsqueeze(logits, 0)

            preds = torch.sigmoid(logits)
            preds = get_constr_out(preds, P['R'].to(device)) if P['mod_scheme'] == 'hmcnn' else preds

            train_label.extend(label_vec_obs.tolist())
            train_output.append(preds.cpu())
            torch.cuda.empty_cache()

    train_label = torch.tensor(train_label)
    train_output = torch.cat(train_output, dim=0)
    output_mat = get_output_matrix(train_output, train_label)

    edges, _ = get_current_edges(output_mat, P['hier_threshold'])
    correct, wrong, edge_rec, edge_prec = save_edge_info(edges, epoch, P, edge_path=edge_path)

    return edge_rec, edge_prec, correct, wrong

def run_correction(P):
    # Only for Level 2 Dataset (Cars, Cifar)
    from sklearn.metrics import recall_score, precision_score
    correction, hidden = LOAD(P['checkpoint'])
    start_epoch = 1
    max_epoch = 100

    num_level_classes = config._LOOKUP['num_level_classes_cum'][P['dataset']]
    num_level_classes = [0] + num_level_classes # 'cub2': [13, 50, 250], 'cifar': [20, 120], 'cars': [9, 205]
    level_range = []
    for i, n_l_c in enumerate(num_level_classes):
        if i == 0: continue
        level_range.append(list(range(num_level_classes[i-1], n_l_c)))

    parent_cp = get_parent_dir(P['checkpoint'])
    corr_result_path = ospj(parent_cp, "correction_result.csv")  # SET DIRECTORY TO SAVE FILE
    with open(corr_result_path, 'a') as f:
        f.write("ALL: All Classes are considered   \n")
        f.write("P0_C1: Parent Unobserved (y=0) & Child Observed (y=1)   \n")
        f.write("C0_P1: Child Unobserved (y=0)  & Parent Observed (y=1)  \n")
        f.write("     , ALL, ALL, ALL,  P0_C1, P0_C1, P0_C1, C0_P1, C0_P1, C0_P1 \n")
        f.write("EPOCH, ACC, REC, PREC, ACC  , REC  , PREC , ACC  , REC  , PREC  \n")

    for epoch in range(start_epoch, max_epoch+1):
        print('Epoch {}/{}'.format(epoch, max_epoch))
        corr = correction[epoch]
        hidd = hidden[epoch]
        accuracy, recall, precision = [], [], []

        # All
        acc = np.all(np.array(corr) == np.array(hidd), axis=1).mean()
        accuracy.append(acc * 100)
        rec = recall_score(np.array(hidd), np.array(corr), average='samples', zero_division=0)
        recall.append(rec * 100)
        prec = precision_score(np.array(hidd), np.array(corr), average='samples', zero_division=0)
        precision.append(prec * 100)

        for l in level_range:
            l_range = np.array(l)
            corr_level = corr[:, l_range]
            hidd_level = hidd[:, l_range]
            acc = np.all(np.array(corr_level) == np.array(hidd_level), axis=1).mean()
            accuracy.append(acc * 100)
            rec = recall_score(np.array(hidd_level), np.array(corr_level), average='samples', zero_division=0)
            recall.append(rec * 100)
            prec = precision_score(np.array(hidd_level), np.array(corr_level), average='samples', zero_division=0)
            precision.append(prec * 100)

        with open(corr_result_path, 'a') as f:
            f.write("{}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f} \n".format(epoch, accuracy[0], recall[0], precision[0], accuracy[1], recall[1], precision[1], accuracy[2], recall[2], precision[2]))

#python main.py --dataset cars --mod_scheme an --seed 1 --checkpoint research\Hierarchy\HMC\results\hw_cars_1e-5_8_1_0.2_default-warmup0.2@25@0.005@1_1\correction_idx_default-warmup0.2@25@0.005@1_1 --correction

def run_test_const(P, phase, best_th, model=None, use_tqdm=True):
    dataset = datasets.get_data(P)
    dataloader = {}
    dataloader[phase] = torch.utils.data.DataLoader(
        dataset[phase],
        batch_size=P['bsize'],
        shuffle=phase == 'train',
        sampler=None,
        num_workers=P['num_workers'],
        drop_last=False,
        pin_memory=True
    )

    role = (P['mod_scheme'] == 'role')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    if model == None:
        model = models.ImageClassifier(P)
        if role:
            model = models.MultilabelModel(P, dataset['train'].label_matrix_obs, None)
        model_path = ospj(P['checkpoint'], P['checkmodel'])
        model_state, _ = torch.load(model_path)
        model.load_state_dict(model_state)

    if P['edge_init_type'] == 'load':
        import networkx as nx
        print('Init edges by load: {}'.format(P['checkedge']))
        correct, wrong, true_edges = get_edge_set(P['edges'], P['dataset'])
        print("Correct_e: {} Wrong_e: {}".format(len(correct), len(wrong)))

        edges = [(int(src), int(des)) for src, des, weight in P['edges']]
        g = nx.DiGraph()
        g.add_edges_from(edges)
        nodes = [cls for cls in range(P['num_classes'])]
        adj_matrix = np.array(nx.to_numpy_matrix(g, nodelist=nodes))
        adj_matrix = torch.tensor(adj_matrix) + torch.eye(P['num_classes'])
        P['R'] = adj_matrix.unsqueeze(0)


    model.to(device)
    model.eval()
    y_pred = np.zeros((len(dataset[phase]), P['num_classes']))
    y_true = np.zeros((len(dataset[phase]), P['num_classes']))
    batch_stack = 0
    dataloader_loop = tqdm(dataloader[phase]) if use_tqdm else dataloader[phase]
    with torch.set_grad_enabled(False):
        for batch in dataloader_loop:
            # Move data to GPU
            image = batch['image'].to(device, non_blocking=True)
            label_vec_true = batch['label_vec_true'].clone().numpy()

            # Forward pass
            if role:
                try:
                    logits = model.f(image)
                except:
                    logits = model.f(image.float())
            else:
                try:
                    logits = model(image)
                except:
                    logits = model(image.float())
            if logits.dim() == 1:
                logits = torch.unsqueeze(logits, 0)
            preds = torch.sigmoid(logits)
            preds = get_constr_out(preds, P['R'].to(device))

            preds_np = preds.cpu().numpy()
            this_batch_size = preds_np.shape[0]
            y_pred[batch_stack: batch_stack + this_batch_size] = preds_np
            y_true[batch_stack: batch_stack + this_batch_size] = label_vec_true
            batch_stack += this_batch_size

    # SAVE([y_pred, y_true], ospj(P['save_path'], P['checkpoint'], 'test_inferenced_{}'.format(P['checkmodel'][:-3])))
    if P['HMC']:
        metrics = compute_metrics(y_pred, y_true, P)
        map_test, ems_test, cv_test, auprc_test = metrics['map'], metrics['ems'], metrics['cv'], metrics['au_prc']
        if best_th < 0:
            max_th_test, max_em_test = get_max_em(ems_test)
            return [map_test, max_em_test, max_th_test, cv_test, auprc_test]
        else:
            max_em_test = ems_test[best_th]
            return [map_test, max_em_test, best_th, cv_test, auprc_test]

    metrics = compute_metrics(y_pred, y_true, P)
    map_test, ems_test, cv_test = metrics['map'], metrics['ems'], metrics['cv']
    if best_th < 0:
        max_th_test, max_em_test = get_max_em(ems_test)
        return [map_test, max_em_test, max_th_test, cv_test]
    else:
        max_em_test = ems_test[best_th]
        return [map_test, max_em_test, best_th, cv_test]