#!/bin/bash

# nohup bash -c '

STEPS="4 5 6 7" 
EPOCHS_LIST="300" 
DI_SEEDS="0" 
USE_RES_LIST="False" 
USE_FINAL_LIST="False" 
SHUFFLE_LIST="False" 
SOLVER="dpmpp" 
DEVICES="0"
LOSS_CONFIG="configs/loss/imagenet64_multi.yaml"

EPS=1e-2
EPS_MIN=1e-3

for STEP in $STEPS
    do

    for EPOCHS in $EPOCHS_LIST
        do

    for DI_SEED in $DI_SEEDS
        do

    for USE_RES in $USE_RES_LIST
        do

    for USE_FINAL in $USE_FINAL_LIST
        do

    for SHUFFLE in $SHUFFLE_LIST
        do

        echo STEP$STEP-EPOCHS$EPOCHS

        if [ $STEP -eq 4 ]; then
            GT_NUM_STEPS=22
        elif [ $STEP -eq 5 ]; then
            GT_NUM_STEPS=21
        elif [ $STEP -eq 6 ]; then
            GT_NUM_STEPS=21
        elif [ $STEP -eq 7 ]; then
            GT_NUM_STEPS=25
        else
            GT_NUM_STEPS=25
        fi
        
        SOLVER_FLAGS="--num_steps=$STEP"
        SCHEDULE_FLAGS="--schedule_type=polynomial --schedule_rho=7" 
        ADDITIONAL_FLAGS="--max_order=3 --predict_x0=True --lower_order_final=True"
        # SCHEDULE_FLAGS="--schedule_type=logsnr"
        # SCHEDULE_FLAGS="--schedule_type=time_uniform --schedule_rho=2"
        CUDA_VISIBLE_DEVICES=$DEVICES python train-${SOLVER}.py --dataset_name="imagenet64" --batch 64 $SOLVER_FLAGS $SCHEDULE_FLAGS \
            --outdir trained_emb --epochs $EPOCHS --seeds '50000-51023' --lr 1e-3 --lr_min 1e-3 --weight_decay 1e-8 \
            --gt_solver dpmpp --solver $SOLVER --gt_num_steps $GT_NUM_STEPS --use_res $USE_RES \
            --use_final $USE_FINAL --eps $EPS --eps_min $EPS_MIN --di_seed $DI_SEED --shuffle $SHUFFLE \
            --mini_batch 16 $ADDITIONAL_FLAGS \
            --loss_config $LOSS_CONFIG

        done
        done
        done
        done
        done
    done

# ' > ./train_cifar.out 2>&1 &
