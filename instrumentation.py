import numpy as np
import copy
import metrics

def get_max_em(ems):
    em = [em for em in ems.values()]
    max_idx = em.index(max(em))
    max_th = [th for th in ems.keys()][max_idx]
    max_em = max(em)
    return max_th, max_em
        
def compute_metrics(y_pred, y_true, P):
    '''
    Given predictions and labels, compute a few metrics.
    '''
    
    num_examples, num_classes = np.shape(y_true)
    
    results = {}
    average_precision_list = []
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_true = np.array(y_true == 1, dtype=np.float32) # convert from -1 / 1 format to 0 / 1 format

    for j in range(num_classes):
        # print(y_true[:, j])
        average_precision_list.append(metrics.compute_avg_precision(y_true[:, j], y_pred[:, j]))
        
    results['map'] = 100.0 * float(np.mean(average_precision_list))
    # = metrics.compute_avg_precision(y_true, y_pred, average='micro')

    from sklearn.metrics import f1_score, average_precision_score
    results['au_prc'] = average_precision_score(y_true, y_pred, average='micro')

    thresholds = [round(th, 2) for th in np.arange(0, 1, 0.01)]
    results['ems'] = {}
    for th in thresholds:
        predicted = np.array(y_pred) > th
        results['ems'][th] = np.all(np.array(y_true) == np.array(predicted),axis=1).mean() * 100
    results['cv'] = metrics.compute_cv(y_pred, P)

    if P['test_leaf']:
        import config, util
        n_level_classes = config._LOOKUP['num_level_classes_cum'][P['dataset']]
        leaf_range = np.array([i for i in range(n_level_classes[-2], n_level_classes[-1])])

        y_leaf_pred = y_pred[:, leaf_range]
        y_leaf_pred = np.argmax(y_leaf_pred, 1)

        y_leaf_true = y_true[:, leaf_range]
        y_leaf_true = np.array(util.GET(y_leaf_true))

        acc = (np.array(y_leaf_true) == np.array(y_leaf_pred)).mean() * 100
        results['leaf_acc'] = acc


    # for k in [1, 3, 5]:
    #     rec_at_k = np.array([metrics.compute_recall_at_k(y_true[i, :], y_pred[i, :], k) for i in range(num_examples)])
    #     prec_at_k = np.array([metrics.compute_precision_at_k(y_true[i, :], y_pred[i, :], k) for i in range(num_examples)])
    #     results['rec_at_{}'.format(k)] = np.mean(rec_at_k)
    #     results['prec_at_{}'.format(k)] = np.mean(prec_at_k)
    #     results['top_{}'.format(k)] = np.mean(prec_at_k > 0)
    #
    return results
