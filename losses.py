import torch
import torch.nn.functional as F
import math
import util


''''''
LOG_EPSILON = 1e-5
def neg_log(x):
    return - torch.log(x + LOG_EPSILON)

def expected_positive_regularizer(preds, expected_num_pos, norm='2'):
    # Assumes predictions in [0,1].
    if norm == '1':
        reg = torch.abs(preds.sum(1).mean(0) - expected_num_pos)
    elif norm == '2':
        reg = (preds.sum(1).mean(0) - expected_num_pos)**2
    else:
        raise NotImplementedError
    return reg
''''''

'''
loss functions
'''
def loss_bce(logits, observed_labels, P):
    assert not torch.any(observed_labels == -1)
    assert P['train_set_variant'] == 'clean'
    # compute loss:
    loss_matrix = F.binary_cross_entropy_with_logits(logits, observed_labels, reduction='none')
    return loss_matrix, None

def loss_bce_ls(logits, observed_labels, P):
    assert not torch.any(observed_labels == -1)
    assert P['train_set_variant'] == 'clean'

    ls_coef = 0.1
    preds = torch.sigmoid(logits)

    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = (1.0 - ls_coef) * neg_log(preds[observed_labels == 1]) + ls_coef * neg_log(
        1.0 - preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = (1.0 - ls_coef) * neg_log(1.0 - preds[observed_labels == 0]) + ls_coef * neg_log(
        preds[observed_labels == 0])
    return loss_mtx, None

def loss_an(logits, observed_labels, P):
    assert torch.min(observed_labels) >= 0
    # compute loss:
    loss_matrix = F.binary_cross_entropy_with_logits(logits, observed_labels, reduction='none')
    corrected_loss_matrix = F.binary_cross_entropy_with_logits(logits, torch.logical_not(observed_labels).float(), reduction='none')
    return loss_matrix, corrected_loss_matrix

def loss_hw(logits, observed_labels, P, idx):
    # observed_labels # n * C
    preds = torch.sigmoid(logits)

    device = logits.get_device()
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels).to(device)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = neg_log(1.0 - preds[observed_labels == 0])
    loss_mtx[observed_labels == 0] *= P['gamma_after']

    # # if P['wan']:
    #     loss_mtx[observed_labels == 0] = neg_log(1.0 - preds[observed_labels == 0]) / float(P['num_classes'] - 1)

    # pseudo_labels
    pseudo_labels = P['pseudo_labels'][idx] #conditional prob from true label
    if P['mod_scheme'] == 'hc':
        assert len(torch.unique(pseudo_labels)) < 3, print("Correction method : pseudo label should be 0 OR 1!!!!!!")

    correction_idx = torch.where(pseudo_labels > 0)
    loss_mtx[correction_idx] = (1 - pseudo_labels[correction_idx]) * neg_log(1 - preds[correction_idx]) \
                                 + (pseudo_labels[correction_idx]) * neg_log(preds[correction_idx])

    if P['weight_classifier']:
        labels = util.GET(observed_labels.cpu()).cpu().detach().numpy()
        if P['weight_classifier_ver'] in ['1.1', '1.2']:
            # Ver1: Apply all labels
            weights = torch.tensor(P['class_weights'][labels]).unsqueeze(1).expand_as(loss_mtx).to(device)
            loss_mtx *= weights
        elif P['weight_classifier_ver'] in ['2.1', '2.2']:
            # Ver2: Apply positive
            weights = torch.tensor(P['class_weights'][labels]).to(device)
            loss_mtx[observed_labels == 1] *= weights

    return loss_mtx, correction_idx

def loss_an_ls(logits, observed_labels, P):
    ls_coef = 0.1
    preds = torch.sigmoid(logits)

    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = (1.0 - ls_coef) * neg_log(preds[observed_labels == 1]) + ls_coef * neg_log(1.0 - preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = (1.0 - ls_coef) * neg_log(1.0 - preds[observed_labels == 0]) + ls_coef * neg_log(preds[observed_labels == 0])
    return loss_mtx, None

def loss_an_ls_hw(logits, observed_labels, P):
    pseudo_labels = P['pseudo_labels'] # conditional prob from true label
    ls_coef = 0.1
    preds = torch.sigmoid(logits)

    if P['mod_scheme'] == 'an_ls+hc':
        assert len(torch.unique(pseudo_labels)) < 3, "Correction method : pseudo label should be 0 OR 1!!!!!!"

    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = (1.0 - ls_coef) * neg_log(preds[observed_labels == 1]) + ls_coef * neg_log(1.0 - preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = neg_log(1.0 - preds[observed_labels == 0])
    loss_mtx[pseudo_labels > 0] = (1 - pseudo_labels[pseudo_labels > 0]) * neg_log(1 - preds[pseudo_labels > 0]) \
                                  + (pseudo_labels[pseudo_labels > 0]) * neg_log(preds[pseudo_labels > 0])

    return loss_mtx, None

def loss_wan(logits, observed_labels, P):
    preds = torch.sigmoid(logits)

    gamma = 1 / float(P['num_classes'] - 1)
    if P['mod_scheme'] == 'hw' and P['gamma_before'] != None:
        gamma = P['gamma_before']

    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = neg_log(1.0 - preds[observed_labels == 0])
    loss_mtx[observed_labels == 0] *= gamma

    # print(loss_mtx[observed_labels == 1])
    # print(loss_mtx[observed_labels == 0])
    return loss_mtx, None

def loss_wan_hw(logits, observed_labels, P, idx):
    pseudo_labels = P['pseudo_labels'][idx] # conditional prob from true label
    preds = torch.sigmoid(logits)

    device = logits.get_device()
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels).to(device)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = neg_log(1.0 - preds[observed_labels == 0])
    # loss_mtx[observed_labels == 0] = neg_log(1.0 - preds[observed_labels == 0]) / float(P['num_classes'] - 1)

    correction_idx = torch.where(pseudo_labels > 0)
    loss_mtx[pseudo_labels > 0] = ((1 - pseudo_labels[pseudo_labels > 0]) / float(P['num_classes'] - 1)) * neg_log(1 - preds[pseudo_labels > 0]) \
                                  + (pseudo_labels[pseudo_labels > 0]) * neg_log(preds[pseudo_labels > 0])
    return loss_mtx, None

def loss_epr(logits, observed_labels, P):
    preds = torch.sigmoid(logits)
    # compute loss w.r.t. observed positives:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    # compute regularizer:
    reg_loss = expected_positive_regularizer(preds, P['expected_num_pos'], norm='2') / (P['num_classes'] ** 2)
    return loss_mtx, reg_loss

def loss_role(logits, observed_labels, P):
    preds = torch.sigmoid(logits)
    assert P['label_vec_est'] != None, print('ROLE needs label_vec_est')
    estimated_labels = P['label_vec_est']

    # (image classifier) compute loss w.r.t. observed positives:
    loss_mtx_pos_1 = torch.zeros_like(observed_labels)
    loss_mtx_pos_1[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    # (image classifier) compute loss w.r.t. label estimator outputs:
    estimated_labels_detached = estimated_labels.detach()
    loss_mtx_cross_1 = estimated_labels_detached * neg_log(preds) + (1.0 - estimated_labels_detached) * neg_log(
        1.0 - preds)

    # (label estimator) compute loss w.r.t. observed positives:
    loss_mtx_pos_2 = torch.zeros_like(observed_labels)
    loss_mtx_pos_2[observed_labels == 1] = neg_log(estimated_labels[observed_labels == 1])
    # (label estimator) compute loss w.r.t. image classifier outputs:
    preds_detached = preds.detach()
    loss_mtx_cross_2 = preds_detached * neg_log(estimated_labels) + (1.0 - preds_detached) * neg_log(
        1.0 - estimated_labels)
    # compute final loss matrix:
    loss_mtx = 0.5 * (loss_mtx_pos_1 + loss_mtx_pos_2)
    loss_mtx += 0.5 * (loss_mtx_cross_1 + loss_mtx_cross_2)

    # (image classifier) compute regularizer:
    reg_1 = expected_positive_regularizer(preds, P['expected_num_pos'], norm='2') / (P['num_classes'] ** 2)
    # (label estimator) compute regularizer:
    reg_2 = expected_positive_regularizer(estimated_labels, P['expected_num_pos'], norm='2') / (P['num_classes'] ** 2)
    reg_loss = 0.5 * (reg_1 + reg_2)
    return loss_mtx, reg_loss

def loss_hmcnn(logits, observed_labels, P):
    device = logits.get_device()
    R = P['R']
    assert torch.min(observed_labels) >= 0
    output = torch.sigmoid(logits)
    try:
        constr_output = util.get_constr_out(output, R.to(device))
        hmcnn_output = observed_labels * output.double()
        hmcnn_output = util.get_constr_out(hmcnn_output, R.to(device))
    except:
        constr_output = util.get_constr_out(output, R)
        hmcnn_output = observed_labels * output.double()
        hmcnn_output = util.get_constr_out(hmcnn_output, R)
    hmcnn_output = (1 - observed_labels) * constr_output.double() + observed_labels * hmcnn_output

    # compute loss:
    loss_matrix = F.binary_cross_entropy(hmcnn_output, observed_labels.double(), reduction='none')

    return loss_matrix, None

'''
top-level wrapper
'''
loss_functions = {
    'LL-R': loss_an,
    'LL-Cp': loss_an,
    'LL-Ct': loss_an,
    'an': loss_an,
    'an-ls': loss_an_ls,
    'wan': loss_wan,
    'epr': loss_epr,
    'bce': loss_bce,
    'bce-ls': loss_bce_ls,
    'role': loss_role,
    'hmcnn': loss_hmcnn,
    'hw': loss_hw,
    'hc': loss_hw,
    'wan+hw': loss_wan_hw,
    'wan+hc': loss_wan_hw,
    'an-ls+hw': loss_an_ls_hw,
    'an-ls+hc': loss_an_ls_hw,
}



def compute_batch_loss(preds, label_vec, P, idx=None, warmup=False): # "preds" are actually logits (not sigmoid activated !)
    mod_scheme = P['mod_scheme']
    assert preds.dim() == 2

    batch_size = int(preds.size(0))
    num_classes = int(preds.size(1))
    
    unobserved_mask = (label_vec == 0)

    if mod_scheme in ['hw', 'hc', 'wan+hw', 'an-ls+hw']:
        if warmup:
            mod_scheme = P['mod_warmup']
            loss_matrix, reg_loss = loss_functions[mod_scheme](preds, label_vec.clamp(0), P)
        else:
            loss_matrix, reg_loss = loss_functions[mod_scheme](preds, label_vec.clamp(0), P, idx)
    else:
        loss_matrix, reg_loss = loss_functions[mod_scheme](preds, label_vec.clamp(0), P)


    correction_idx = None
    if mod_scheme == 'an':
        final_loss_matrix = loss_matrix
        reg_loss = None

    elif mod_scheme in ['LL-Ct', 'LL-Cp', 'LL-R']:
        if P['clean_rate'] == 1:
            final_loss_matrix = loss_matrix
            reg_loss = None
        else:
            corrected_loss_matrix = reg_loss
            reg_loss = None
            if mod_scheme == 'LL-Cp':
                k = math.ceil(batch_size * num_classes * P['delta_rel'])
            else:
                k = math.ceil(batch_size * num_classes * (1-P['clean_rate']))
            unobserved_loss = unobserved_mask.bool() * loss_matrix
            topk = torch.topk(unobserved_loss.flatten(), k)
            topk_lossvalue = topk.values[-1]
            correction_idx = torch.where(unobserved_loss > topk_lossvalue)
            if mod_scheme in ['LL-Ct', 'LL-Cp']:
                final_loss_matrix = torch.where(unobserved_loss < topk_lossvalue, loss_matrix, corrected_loss_matrix)
                # final_loss_matrix = torch.where(unobserved_loss < topk_lossvalue, loss_matrix, reg_loss)
            elif mod_scheme in ['LL-R']:
                zero_loss_matrix = torch.zeros_like(loss_matrix)
                final_loss_matrix = torch.where(unobserved_loss < topk_lossvalue, loss_matrix, zero_loss_matrix)

    elif mod_scheme in ['hw', 'hc', 'wan+hw', 'an-ls+hw']:
        final_loss_matrix = loss_matrix
        correction_idx = reg_loss
        return final_loss_matrix.mean(), correction_idx

    else:
        final_loss_matrix = loss_matrix

    main_loss = final_loss_matrix.mean()
    if reg_loss != None:
        main_loss += reg_loss

    return main_loss, correction_idx