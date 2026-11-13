#!/bin/bash

STEPS="4" 
EPOCHS_LIST="200" # 30 40 50 60
DI_SEEDS="0" # 0 1 2
USE_RES_LIST="False" # True False
USE_FINAL_LIST="False" # True False
SHUFFLE_LIST="False" # True False
SOLVER="euler" 
DEVICES="${DEVICES:-0,1,2,3}" # 0,1,2,3
MASTER_PORT="${MASTER_PORT:-11111}"
LOSS_CONFIG="configs/loss/coco.yaml"

IFS=',' read -ra DEVICE_ARRAY <<< "$DEVICES"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#DEVICE_ARRAY[@]}}"

EPS=1e-9
EPS_MIN=1e-9

# bash -c '

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

        if [ "$STEP" -eq 4 ]; then
            GT_NUM_STEPS=22
        elif [ "$STEP" -eq 5 ]; then
            GT_NUM_STEPS=21
        elif [ "$STEP" -eq 6 ]; then
            GT_NUM_STEPS=21
        elif [ "$STEP" -eq 7 ]; then
            GT_NUM_STEPS=25
        else
            GT_NUM_STEPS=22
        fi

        SOLVER_FLAGS="--num_steps=$STEP"
        SCHEDULE_FLAGS="--schedule_type=discrete --schedule_rho=1"
        ADDITIONAL_FLAGS="--max_order=2 --predict_x0=False --lower_order_final=True"
        GUIDANCE_FLAGS="--guidance_type=cfg --guidance_rate=7.5"

        CUDA_VISIBLE_DEVICES=$DEVICES  PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT \
            train_coco.py --dataset_name="ms_coco" --batch 64 $SOLVER_FLAGS $SCHEDULE_FLAGS \
            --outdir trained_emb --epochs $EPOCHS --seeds '30000-30255' --lr 5e-2 --lr_min 2e-2 --weight_decay 1e-8 $GUIDANCE_FLAGS \
            --gt_solver dpmpp --solver $SOLVER --gt_num_steps $GT_NUM_STEPS --use_res $USE_RES --gt_batch 4 \
            --use_final $USE_FINAL --eps $EPS --eps_min $EPS_MIN --di_seed $DI_SEED --shuffle $SHUFFLE \
            --mini_batch 2 --loss_config $LOSS_CONFIG $ADDITIONAL_FLAGS \
        # > ./train_coco.out 2>&1
        done
        done
        done
        done
        done
    done


