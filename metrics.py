from sklearn.metrics import average_precision_score, accuracy_score, f1_score
from util import *

def check_inputs(targs, preds):
    '''
    Helper function for input validation.
    '''
    
    assert (np.shape(preds) == np.shape(targs))
    assert type(preds) is np.ndarray
    assert type(targs) is np.ndarray
    assert (np.max(preds) <= 1.0) and (np.min(preds) >= 0.0)
    assert (np.max(targs) <= 1.0) and (np.min(targs) >= 0.0)
    assert (len(np.unique(targs)) <= 2)

def compute_avg_precision(targs, preds):
    '''
    Compute average precision.
    
    Parameters
    targs: Binary targets.
    preds: Predicted probability scores.
    '''
    
    check_inputs(targs,preds)
    
    if np.all(targs == 0):
        # If a class has zero true positives, we define average precision to be zero.
        metric_value = 0.0
    else:
        metric_value = average_precision_score(targs, preds)
    
    return metric_value

def compute_precision_at_k(targs, preds, k):
    '''
    Compute precision@k. 
    
    Parameters
    targs: Binary targets.
    preds: Predicted probability scores.
    k: Number of predictions to consider.
    '''
    
    check_inputs(targs, preds)
    
    classes_rel = np.flatnonzero(targs == 1)
    if len(classes_rel) == 0:
        return 0.0
    
    top_k_pred = np.argsort(preds)[::-1][:k]
    
    metric_value = float(len(np.intersect1d(top_k_pred, classes_rel))) / k
    
    return metric_value

def compute_recall_at_k(targs, preds, k):
    '''
    Compute recall@k. 
    
    Parameters
    targs: Binary targets.
    preds: Predicted probability scores.
    k: Number of predictions to consider.
    '''
    
    check_inputs(targs,preds)
    
    classes_rel = np.flatnonzero(targs == 1)
    if len(classes_rel) == 0:
        return 0.0
    
    top_k_pred = np.argsort(preds)[::-1][:k]
    
    metric_value = float(len(np.intersect1d(top_k_pred, classes_rel))) / len(classes_rel)
    
    return metric_value


def compute_cv(preds, P):
    bsize, num_classes = preds.shape[0], preds.shape[1]
    # print(P['R'].shape)
    adj = P['R'].expand([bsize, num_classes, num_classes])
    # edge_path = ospj('saved_multi', 'edge', 'true_edge_{}'.format(P['dataset']))
    # true_edge = LOAD(edge_path)

    positive_prob_diag = preds[:,:,None]
    positive_prob_stack = preds[:,None,:]


    denominator = torch.sum(adj)
    numerator = torch.sum((adj * (positive_prob_stack > positive_prob_diag)))
    assert denominator > 0
    cv = (numerator/denominator).item()

    # count = 0
    # for pred in preds:
    #     for (par, chd) in true_edge:
    #         if pred[par] - pred[chd] < 0:
    #             count += 1
    # cv = count / (preds.shape[0] * len(true_edge))
    return cv * 100
