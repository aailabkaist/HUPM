import losses
import datasets
import models
from util import *
from test import run_test
from sklearn.metrics import recall_score, precision_score

def warmup(epoch, P):
    return epoch <= P['warmup_epoch']

def extract(epoch, P):
    # epoch for hierarchy extraction
    if P['edge_fix'] and len(P['edges']) > 0:
        return False
    if P['extract'] and ((epoch + 1) % P['iter_epoch'] == 0):
        return True

def exploit(epoch, P):
    # epoch for applying HAN loss
    if P['my']:
        if epoch == 1:
            if P['edge_init_type'] != 'none':
                return True
            if P['true_edge']:
                # print("Apply true edge")
                return True
        if (epoch > 1) and ((epoch) % P['iter_epoch'] == 0) and (not warmup(epoch, P)):
            return True
    return False




def run_train(P):
    # result file
    epoch_result_path = ospj(P['save_path'], P['exp_name'], "epoch_result.csv")
    with open(epoch_result_path, 'a') as f:
        f.write(P['exp_name'] + '\n')
        if P['HMC']:
            f.write(
                "EPOCH, mAP, EM, CV, EM_TH, AUPRC, mAP_TEST, EM_TEST, CV_TEST, AUPRC_TEST, CORR_ACC, CORR_REC, CORR_PREC, EDGE_REC, EDGE_PREC, CORRECT_EDGE, WRONG_EDGE \n")
        else:
            f.write("EPOCH, mAP, EM, CV, EM_TH, mAP_TEST, EM_TEST, CV_TEST, CORR_ACC, CORR_REC, CORR_PREC, EDGE_REC, EDGE_PREC, CORRECT_EDGE, WRONG_EDGE \n")

    save_and_print(P['result_path'], str(P))
    save_and_print(P['result_path'], P['exp_name'])
    save_and_print(P['result_path'], 'Running on GPU {}'.format(P['gpu_num']))
    save_and_print(P['result_path'], 'Early Stop {}'.format('On' if P['stop'] else 'Off'))
    if P['warmup_epoch'] > 1:
        save_and_print(P['result_path'], 'Warmup with gamma_before: {}, gamma_after: {}'.format(P['gamma_before'] if P['gamma_before'] != None else 'wan', P['gamma_after']))

    # Dataloader
    dataset = datasets.get_data(P)
    dataset = get_reduced_dataset(dataset, P)
    sampler = get_sampler(dataset, P)

    dataloader = {}
    for phase in ['train']:
        dataloader[phase] = torch.utils.data.DataLoader(
            dataset[phase],
            batch_size=P['bsize'],
            shuffle=(phase == 'train' and sampler[phase] == None),
            sampler=sampler[phase],
            num_workers=P['num_workers'],
            drop_last=True,
            pin_memory=True
        )


    model = models.ImageClassifier(P)

    role = False
    hmc = False
    if P['mod_scheme'] == 'role':
        role = True
        model = models.MultilabelModel(P,dataset['train'].label_matrix_obs, None)
    elif P['dataset'] not in ['cars', 'cub2', 'cifar']:
        hmc = True
        P['R'] = dataset['train'].adj_matrix
        model = models.ConstrainedFFNNModel(P)

    # optimizer setting
    opt_params = model.get_config_optim(P)
    if P['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(opt_params, lr=P['lr']) if not hmc else torch.optim.Adam(opt_params, lr=P['lr'], weight_decay=1e-5)
    
    # model setting
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    P['R'].to(device)

    # finetune model
    # path = ospj('saved_multi', 'model', P['checkpoint'], P['checkmodel'])
    if P['checkpoint'] != None and P['checkmodel'] != None:
        path = ospj(P['checkpoint'], P['checkmodel'])
        assert os.path.exists(path), '{} does not exist'.format(path)
        save_and_print(P['result_path'],
                       'Use pretrained model: {}'.format(path))
        model_state, _ = torch.load(path)
        model.load_state_dict(model_state)
        torch.save((model.state_dict(), P), ospj(P['save_dir'], 'checkmodel.pt'))

    # Set edge
    if P['edge_init_type'] == 'random':
        save_and_print(P['result_path'],
                       'Init edges by random: proportion = {}%, weight = {} {}'.format(P['edge_random_prop'], P['edge_random_weight'], 'FIXED!' if P['edge_fix'] else ''))
        correct, wrong, edge_rec, edge_prec = save_edge_info(P['edges'], 0, P)
        save_and_print(P['result_path'],
                       "Init edge: Correct_e: {} Wrong_e: {}".format(len(correct), len(wrong)))
        with open(epoch_result_path, 'a') as f:
            f.write('0, , , , , , , , , , , {}, {}, {}, {} \n'.format(edge_rec, edge_prec, len(correct), len(wrong)))
    elif P['edge_init_type'] == 'load':
        save_and_print(P['result_path'],
                       'Init edges by load: {} {}'.format(P['checkedge'], 'FIXED!' if P['edge_fix'] else ''))
        correct, wrong, _, _ = save_edge_info(P['edges'], P['skip_epoch'], P)
        save_and_print(P['result_path'],
                       "Init edge: Correct_e: {} Wrong_e: {}".format(len(correct), len(wrong)))
    elif P['true_edge']:
        save_and_print(P['result_path'],
                       'Init edges by ground truth')
        P['edges'] = get_true_edges(P['dataset'])
        # print(P['edges'])

    # For Correction analysis
    epoch_correct = {i:[] for i in range(P['num_epochs']+1)}
    epoch_correct_hidden = {i: [] for i in range(P['num_epochs'] + 1)}

    # Initialize
    if P['my']:
        P['pseudo_labels'] = torch.zeros(dataset['train'].label_matrix_obs.shape).to(device)
    bestmap_val = 0
    start_time = time.time()

    # Start train
    for epoch in tqdm(range(1, P['num_epochs']+1)):
        if epoch <= P['skip_epoch']:
            print('Skip epoch', epoch)
            continue

        train_label, train_output = [], []
        model.train()

        # For Correction analysis
        y_correct = []
        y_correct_hidden = []

        with torch.set_grad_enabled(True):
            # 1. Train classifier
            for batch in dataloader['train']:
                image = batch['image'].to(device, non_blocking=True)
                label_vec_obs = batch['label_vec_obs'].to(device, non_blocking=True)
                label_vec_true = batch['label_vec_true'].clone().numpy()
                label_vec_hidden = label_vec_true - label_vec_obs.cpu().detach().numpy()
                idx = batch['idx']

                # Forward pass
                optimizer.zero_grad()
                if role:
                    logits = model.f(image)
                    P['label_vec_est'] = model.g(idx)
                else:
                    try:
                        logits = model(image)
                    except:
                        logits = model(image.float())

                if logits.dim() == 1:
                    logits = torch.unsqueeze(logits, 0)
                preds = torch.sigmoid(logits)
                preds = get_constr_out(preds, P['R'].to(device)) if P['mod_scheme'] == 'hmcnn' else preds

                if exploit(epoch, P):
                    P['pseudo_labels'][idx] = get_pseudo_w_y(label_vec_obs.cpu(), preds, P).to(device)
                else:
                    if P['my'] and not P['my_cont']:
                        P['pseudo_labels'][idx] = torch.zeros_like(label_vec_obs).to(device)
                loss, correction_idx = losses.compute_batch_loss(logits, label_vec_obs, P, idx, warmup(epoch, P))

                loss.backward()
                optimizer.step()

                # For Correction analysis
                label_vec_correct = torch.zeros(label_vec_obs.size(0), P['num_classes'])
                if P['mod_scheme'] in ['LL-Ct', 'LL-Cp', 'hw', 'hc']:
                    if correction_idx != None:
                        label_vec_correct[correction_idx] = 1
                        if (P['mod_scheme'] == 'LL-Cp') or (P['mod_scheme'] == 'LL-Ct' and P['permanent']):
                            dataset[phase].label_matrix_obs[idx[correction_idx[0].cpu()], correction_idx[1].cpu()] = 1.0
                            if len(correction_idx[0]) > 0:
                                print(epoch, correction_idx)
                                print(dataset[phase].label_matrix_obs[idx[correction_idx[0].cpu()], correction_idx[1].cpu()])
                                exit()

                    y_correct.append(label_vec_correct)
                    y_correct_hidden.append(torch.tensor(label_vec_hidden))



                if extract(epoch, P):
                    train_label.extend(label_vec_obs.tolist())
                    train_output.append(preds)
            # 2. Extract Hierarchy
            edge_rec, edge_prec, correct, wrong = 0, 0, [], []
            if P['true_edge']:
                # true_edge_path = ospj('saved_multi', 'edge', 'true_edge_{}'.format(P['dataset']))
                # true_edge = LOAD(true_edge_path)
                # P['edges'] = [(src, des, 1) for src, des in true_edge]
                edge_rec, edge_prec, correct, wrong = 100, 100, P['edges'], []
                # print(P['edges'])
                # print('given true edge')
            else:
                if extract(epoch, P):
                    train_label = torch.tensor(train_label).cpu()
                    train_output = torch.cat(train_output, dim=0).cpu()
                    output_mat = get_output_matrix(train_output, train_label)
                    edges, _ = get_current_edges(output_mat, P['hier_threshold'])
                    P['edges'] = edges
                    correct, wrong, edge_rec, edge_prec = save_edge_info(edges, epoch, P)
                if P['edge_fix']:
                    correct, wrong, edge_rec, edge_prec = save_edge_info(P['edges'], epoch, P)

        if P['mod_scheme'] in ['LL-R', 'LL-Ct', 'LL-Cp']:
            P['clean_rate'] -= P['delta_rel']

        # For Correction analysis
        accuracy, recall, precision = 0, 0, 0
        if P['mod_scheme'] in ['LL-Ct', 'LL-Cp', 'hc', 'hw']:
            corr, hidd = torch.cat(y_correct), torch.cat(y_correct_hidden)
            accuracy = np.all(np.array(corr) == np.array(hidd), axis=1).mean()
            try:
                recall = recall_score(np.array(hidd), np.array(corr), average='samples', zero_division=0)
                precision = precision_score(np.array(hidd), np.array(corr), average='samples', zero_division=0)

            except:
                for i, cor in enumerate(corr):
                    print("corr", GET(corr[i]))
                    print("hidd", GET(hidd[i]))
                recall = recall_score(np.array(hidd), np.array(corr), average='samples')
                precision = precision_score(np.array(hidd), np.array(corr), average='samples')
            accuracy, recall, precision = accuracy * 100, recall * 100, precision * 100

            epoch_correct[epoch], epoch_correct_hidden[epoch] = corr, hidd

        # Validation
        if P['HMC']:
            map_val, max_em_val, max_th_val, cv_val, auprc_val = run_test(P, phase='val', best_th=-1, model=model, use_tqdm=False)
        else:
            map_val, max_em_val, max_th_val, cv_val = run_test(P, phase='val', best_th=-1, model=model, use_tqdm=False)

        path = ospj(P['save_dir'], 'model_{}.pt'.format(epoch))
        torch.save((model.state_dict(), P), path)

        if bestmap_val < map_val or P['test_all']:
            if bestmap_val < map_val:
                bestmap_val, bestmap_epoch, best_em_val, best_th_val = map_val, epoch, max_em_val, max_th_val
                best_path = ospj(P['save_dir'], 'bestmodel.pt')
                torch.save((model.state_dict(), P), best_path)

            # Test
            P['checkpoint'], P['checkmodel'] = P['save_dir'], 'bestmodel.pt'
            if P['HMC']:
                map_test, max_em_test, _, cv_test, auprc_test = run_test(P, phase='test', best_th=max_th_val, model=model, use_tqdm=False)
            else:
                map_test, max_em_test, _, cv_test = run_test(P, phase='test', best_th=max_th_val, model=model, use_tqdm=False)

        if P['HMC']:
            print("EPOCH{} | Val: {:.1f}, {:.1f} ({:.2f}), {:.1f} Test: {:.1f}, {:.1f}, {:.1f} | Acc: {:.1f} Rec: {:.1f} Prec: {:.1f} | Correct_e: {} Wrong_e: {}"
                  .format(epoch, map_val, max_em_val, max_th_val, auprc_val, map_test, max_em_test, auprc_test, accuracy, recall, precision, len(correct), len(wrong)))
            with open(epoch_result_path, 'a') as f:
                f.write('{}, {:.3f}, {:.3f}, {:.3f}, {:.2f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.1f}, {:.1f}, {:.1f}, {}, {}, {}, {} \n'.\
                            format(epoch,
                             map_val, max_em_val, cv_val, max_th_val, auprc_val,
                             map_test, max_em_test, cv_test, auprc_test,
                             accuracy, recall, precision,
                             edge_rec, edge_prec, len(correct), len(wrong)))
        else:
            print("EPOCH{} | Val: {:.1f}, {:.1f} ({:.2f}) Test: {:.1f}, {:.1f} | Acc: {:.1f} Rec: {:.1f} Prec: {:.1f} | Correct_e: {} Wrong_e: {}"
                  .format(epoch, map_val, max_em_val, max_th_val, map_test, max_em_test, accuracy, recall, precision, len(correct), len(wrong)))
            with open(epoch_result_path, 'a') as f:
                f.write('{}, {:.3f}, {:.3f}, {:.3f}, {:.2f}, {:.3f}, {:.3f}, {:.3f}, {:.1f}, {:.1f}, {:.1f}, {}, {}, {}, {} \n'.\
                            format(epoch,
                             map_val, max_em_val, cv_val, max_th_val,
                             map_test, max_em_test, cv_test,
                             accuracy, recall, precision,
                             edge_rec, edge_prec, len(correct), len(wrong)))

        if stop(bestmap_val, map_val, bestmap_epoch, epoch, max_em_val, P):
            break


    print('Training procedure completed!')
    save_and_print(P['result_path'], P['exp_name'])
    save_and_print(P['result_path'], 'Best epoch:     {}'.format(bestmap_epoch))
    save_and_print(P['result_path'], 'Training time:  {:.2f}h for epoch {}'.format((time.time()-start_time) / 3600, epoch))
    save_and_print(P['result_path'], 'Best epoch val score:     mAP {:.3f}, EM {:.3f} ({:.2f}), CV {:.3f} ({})'.format(bestmap_val, best_em_val, best_th_val, cv_val, bestmap_epoch))
    save_and_print(P['result_path'], 'Best epoch test score:    mAP {:.3f}, EM {:.3f} ({:.2f}), CV {:.3f} ({})\n\n\n: '.format(map_test, max_em_test, best_th_val, cv_test, bestmap_epoch))

    # For Correction analysis
    if P['mod_scheme'] in ['LL-Ct', 'LL-Cp', 'hw', 'hc']:
        SAVE([epoch_correct, epoch_correct_hidden], ospj(P['save_path'], P['exp_name'], 'correction_idx_{}'.format(P['flag'])))

    return bestmap_epoch, [bestmap_val, best_em_val, best_th_val], [map_test, max_em_test, best_th_val]