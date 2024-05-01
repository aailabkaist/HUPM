import traceback
from config import get_configs
from train import run_train
# from train_role import run_train_role
# from train_ablation import run_train_ablation
from test import run_test, get_edges, get_dist, run_correction, run_test_const
from util import *


def main():
    P = get_configs()
    os.environ['CUDA_VISIBLE_DEVICES'] = P['gpu_num']

    if P['train']:
        # set_experiment(P)
        print('###### Train start ######')
        run_train(P)

    if P['test']:
        mod = 'default'
        if P['constraint']: mod = 'const'

        test_functions = {
            'default': run_test,
            'const': run_test_const
        }
        if P['checkpoint+'] == None:
            P['checkpoint+'] = [P['checkpoint']]
        if P['checkmodel+'] == None:
            P['checkmodel+'] = [P['checkmodel']]


        for checkpoint in P['checkpoint+']:
            for checkmodel in P['checkmodel+']:
                print(checkpoint, checkmodel)

                # Validation Set
                [map_test, max_mr, max_th, cv] = test_functions[mod](P, 'val', -1)
                print('val: mAP {:.3f}, EM {:.3f} ({:.2f}), CV {:.3f}'.format(map_test, max_mr, max_th, cv))

                # Test Set
                [map_test, max_mr, max_th, cv] = test_functions[mod](P, 'test', -1)
                print('test: mAP {:.3f}, EM {:.3f}, CV {:.3f}'.format(map_test, max_mr, max_th, cv))

    if P['dist']:
        # Get mean and std for Epoch 1 to 100
        parent_cp = get_parent_dir(P['checkpoint'])
        print(parent_cp)
        dist_result_path = ospj(parent_cp, "dist_result.csv") # SET DIRECTORY TO SAVE FILE

        with open(dist_result_path, 'a') as f:
            f.write("EPOCH, FN_MEAN, FN_STD, TP_MEAN, TP_STD, TN_MEAN, TN_STD  \n")

        max_epoch = 100
        for epoch in range(1, max_epoch + 1):
            print('Epoch {}/{}'.format(epoch, max_epoch))

            P['checkmodel'] = 'model_{}.pt'.format(epoch)
            path = ospj(P['checkpoint'], P['checkmodel'])
            assert os.path.exists(path), '{} does not exist'.format(path)
            result = get_dist(P)
            with open(dist_result_path, 'a') as f:
                f.write("{}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n".format(epoch,
                                              result['FN']['mean'], result['FN']['std'],
                                              result['TP']['mean'], result['TP']['std'],
                                              result['TN']['mean'], result['TN']['std']))

    if P['edge']:
        # python3.6 main.py --mod_scheme an --dataset cars --hier_threshold 0.2 --edge --checkpoint results/hw_cars_1e-5_8_1_0.2_default-warmup0.2@25@0.005@1_1/default-warmup0.2@25@0.005@1_1 --seed 1

        parent_cp = get_parent_dir(P['checkpoint'])
        print(parent_cp)
        edge_result = ospj(parent_cp, "edge_result.csv")  # SET DIRECTORY TO SAVE FILE

        with open(edge_result, 'a') as f:
            f.write("EPOCH, EDGE_REC, EDGE_PREC, CORRECT_EDGE, WRONG_EDGE\n")

        max_epoch = 100
        max_epoch = P['num_epochs']
        for epoch in range(1, max_epoch + 1):
            print('Epoch {}/{}'.format(epoch, max_epoch))

            P['checkmodel'] = 'model_{}.pt'.format(epoch)
            edge_path = SET_PATH(ospj(parent_cp, 'edges_{}'.format(P['hier_threshold'])), file='edge_{}.txt'.format(epoch))
            edge_rec, edge_prec, correct, wrong = get_edges(P, edge_path)
            with open(edge_result, 'a') as f:
                f.write("{}, {:.3f}, {:.3f}, {}, {}\n".format(epoch, edge_rec, edge_prec, len(correct), len(wrong)))

    if P['correction']:
        run_correction(P)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(traceback.format_exc())
