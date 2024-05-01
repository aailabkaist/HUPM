# HUPM

[Args]
0. Get Dataset
    STANFORD-CARS, CUB_200_2011, CIFAR100

1. Set IMAGE_PATH
    * Dataset:
        'home/aailab/data/CUB_200_2011', 'home/aailab/data/STANFORD-CARS', 'home/aailab/data/CIFAR100'
    * Main directory:
        'home/aailab/USER_NAME/HMC-partials/'

    IMAGE_PATH = ../../data
    => Change --image_path in config.py

2. Decide
    * flag
        name of experiment
        ex: default, warmup20, true-edge, ...
    * gpu_num
        Do not use CUDA_VISIBLE_DEVICES
    * seed
        [1, 2, 3, 4, 5] only available

4. Try
    python main.py --mod_scheme hw --dataset cars --lr 1e-4 --bsize 16 --iter_epoch 1 --hier_threshold 0.2 --image_path IMAGE_PATH --flag TEST --gpu_num 0 --seed 1

    => EXP_NAME = hw_cars_1e-4_16_1_0.2_TEST_1
                = {mod_scheme}_{dataset}_{lr}_{bsize}_{iter_epoch}_{hier_threshold}_{flag}_{seed}
    => Check results in results\{EXP_NAME}

5. Etc
    --stop              :stop early when conditions are satisfied (check util.stop())
    --true_edge         :give true edge for HAN-W
    --weight_sampler    :apply batch sampler
    --no_extract        :not extract hierarchy (use only for baseline, training time will be shorter)

6. Initialize edge
    * random
        python main.py --dataset cub2 --mod_scheme hw --hier_threshold 0.3 --edge_init_type random --edge_random_prop 0.05 --edge_random_weight 0.2 --flag edge@randomTEST --seed 1

    * load
	    Model starts at scratch -> performance low
        python main.py --dataset cub2 --mod_scheme hw --hier_threshold 0.3 --edge_init_type load --checkpoint results/hw_cub2_1e-5_8_1_0.3_warmup0.3@30TEST_1/warmup0.3@30TEST_1 --edge_load_epoch 31 --flag edge@loadTEST --seed 1

7. Apply warmup to our loss
    * from scratch
        python main.py --mod_scheme hw --dataset DATASET --hier_threshold ${hier_threshold} --flag warmup${hier_threshold}@${warmup_epoch} --mod_warmup wan --warmup_epoch ${warmup_epoch} --seed 1

    * Use Pretrained warmup model
        Set ${dataset}, ${lr}, ${bsize}, ${warmup_epoch} same as checkpoint

        IF checkpoint hier_threshold == ${hier_threshold}
            No Initialized edge (edge_init_type='none') & Train start at ${warmup_epoch}
            python main.py --dataset cub2 --mod_scheme hw --hier_threshold 0.4 --flag warmup0.4@30TEST --mod_warmup wan --warmup_epoch 30 --seed 1 --warmup_pretrained --checkpoint results/hw_cub2_1e-5_8_1_0.3_warmup0.3@30_1/warmup0.3@30_1/

        ELSE checkpoint hier_threshold == ${hier_threshold}
            Initialized edge loaded (edge_init_type='load') & Train start at ${warmup_epoch}+1
          python main.py --mod_scheme hw --dataset cub2 --hier_threshold 0.3 --flag warmup0.3@30TEST --mod_warmup wan --warmup_epoch 30 --seed 1 --warmup_pretrained --checkpoint results/hw_cub2_1e-5_8_1_0.3_warmup0.3@30_1/warmup0.3@30_1/



[During Training]
You can check training process in display.

EPOCH{} | Val: {mAP}, {EM} ({EM threshold}) Test: {mAP}, {EM} | Acc: {Correction Accuracy} Rec: {Correction Recall} Prec: {Correction Precision} | Correct_e: {Correct Edge} Wrong_e: {Wrong Edge}
EPOCH1  | Val: 14.4,  0.0  (0.00)           Test: 11.7,   0.0 | Acc:  0.0                  Rec: 0.0                 Prec: 0.0                    | Correct_e: 196            Wrong_e: 3108
1%|▉      | 1/100 [02:35<4:16:54, 155.70s/it]

* Correction Accuracy/Recall/Precision
    Metric to check HAN-W or LL-Ct correctly correct False Negative (unobserved positive) into True label




[Result]
1. results\{EXP_NAME}
    - epoch_result.csv
        Validation: every epoch
        Test: only at best epoch
    - final_result.txt
        Arguments, Val best epoch result
    - correction_idx_{flag}.pkl
        Corrected label indices at each epoch
        Only for HW, (HC), LL-Ct

2. results\{EXP_NAME}\{flag}_{seed}
    - model_{epoch}.pt
        saved in every epoch
    - bestmodel.pt
        saved in best epoch (changed during training)
    - edges_{flag}_epoch{epoch}_{correct}|{correct+wrong}_rec{Edge Recall}_pre{Edge Precision}.txt
        saved in every extract epoch

* Check example
    results\hw_cars_1e-05_8_1_0.5_warmup10
    results\hw_cars_1e-05_8_1_0.5_warmup10\warmup10


[After Training]
You can check validation best (mAP) result in display.

hw_cub2_1e-5_8_1_0.3_warmup0.3@30_1
Best epoch:     36
Training time:  2.02h
Best epoch val score:     mAP 72.072, EM 24.270 (0.51), CV 6.497 (36)
Best epoch test score:    mAP 68.964, EM 24.059 (0.51), CV 5.299 (36)




[Test]
During training, Testing runs only on the best validation epoch.
You can check test performance of individual model.

1. Test single model
    python main.py --test --mod_scheme hw --dataset DATASET --checkpoint result/{EXP_NAME}/{FLAG} --checkmodel MODEL_NAME --image_path IMAGE_PATH
    python main.py --test --mod_scheme hw --dataset cars    --checkpoint results/hw_cars_1e-05_8_1_0.5_warmup10/warmup10 --checkmodel bestmodel.pt --image_path IMAGE_PATH

2. Test multiple models
    - multi checkpoints
    python main.py --test --mod_scheme hw --dataset DATASET --checkpoint+ CHECKPOINT1 CHECKPOINT2 CHECKPOINT3 --checkmodel MODEL_NAME --image_path IMAGE_PATH

    - multi checkmodels
    python main.py --test --mod_scheme hw --dataset DATASET --checkpoint CHECKPOINT -
