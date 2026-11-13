#!/bin/bash

# nohup bash -c '

DEVICES="3"
STEPS="4" # 6 11 16 21
EPOCHS_LIST="300" # 30 40 50 60
SOLVER="ipndm" # euler ipndm
TRAIN_SEED="50000-50255" # 50255 51023 52047
USE_FINAL_LIST="False" # True False
DI_SEEDS="0" # 0 1 2
USE_RES_LIST="False" # True False
SHUFFLE_LIST="False" # True False
DATASET="cifar10" # cifar10 ffhq imagenet64 
LOSS_CONFIG="configs/loss/cifar_multi.yaml"
LOSS_ID=$(python - <<PY
from loss_config_utils import load_loss_id
print(load_loss_id("$LOSS_CONFIG"))
PY
)

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

        NFE=$((STEP-1))
        OUTDIR="samples_di/$DATASET/$SOLVER/STEP$STEP/use_final=$USE_FINAL/use_res=$USE_RES/di_seed=$DI_SEED/shuffle=$SHUFFLE/loss_id=$LOSS_ID/$DATASET-$SOLVER-nfe$((NFE))_epoch-$EPOCHS-seed$TRAIN_SEED"
        echo OUTDIR="$OUTDIR"
        echo STEP$STEP-EPOCHS$EPOCHS

        if [ $STEP -eq 4 ]; then
            GT_NUM_STEPS=22
            LR=0.02
        elif [ $STEP -eq 5 ]; then
            GT_NUM_STEPS=21
            LR=0.02
        elif [ $STEP -eq 6 ]; then
            GT_NUM_STEPS=21
            LR=0.02
        elif [ $STEP -eq 7 ]; then
            GT_NUM_STEPS=25
            LR=0.02
        else
            GT_NUM_STEPS=25
        fi

        
        SOLVER_FLAGS="--num_steps=$STEP"
        SCHEDULE_FLAGS="--schedule_type=polynomial --schedule_rho=7"
        # SCHEDULE_FLAGS="--schedule_type=time_uniform --schedule_rho=2"

        CUDA_VISIBLE_DEVICES=$DEVICES python sample_time_emb.py --dataset_name=$DATASET --batch 32 $SOLVER_FLAGS $SCHEDULE_FLAGS \
            --outdir "${OUTDIR}" --epochs $EPOCHS --seeds '0-49999' --lr $LR --solver $SOLVER \
            --port 1232 --train_seed $TRAIN_SEED --save_npzs False --max_order 4 --use_final $USE_FINAL \
            --di_seed $DI_SEED --use_res $USE_RES --shuffle $SHUFFLE \
            --gt_num_steps $GT_NUM_STEPS --loss_config $LOSS_CONFIG
            # euler heun ipndm
            # ffhq cifar10 lsun_bedroom_ldm imagenet64 0-49999
        
        FID_LOG="./fid_logs/fid-cifar10.out"
        mkdir -p "$(dirname "$FID_LOG")"
        echo "===== $(date '+%Y-%m-%d %H:%M:%S') | $OUTDIR =====" >> "$FID_LOG"
        echo "steps: $STEP" >> "$FID_LOG"
        echo "solver: $SOLVER" >> "$FID_LOG"
        { echo "$SCHEDULE_FLAGS"; echo "--loss_config $LOSS_CONFIG"; } | sed 's/ --/\n--/g' | sed 's/^--//; s/[= ]/: /' >> "$FID_LOG"
        {
            echo "loss_id: $LOSS_ID"
            echo "loss_config_content:"
            sed 's/^/  /' "$LOSS_CONFIG"
        } >> "$FID_LOG"
        CUDA_VISIBLE_DEVICES=$DEVICES python fid.py calc \
        --images=$OUTDIR \
        --ref="stats/cifar10-32x32.npz" --port 1232 2>&1 \
        | tee >(tr '\r' '\n' | grep -vE '%\|' >> "$FID_LOG") # cifar10-32x32.npz ffhq-64x64.npz imagenet-64x64.npz
       

        # rm -r $OUTDIR
        done
        done
        done
        done
        done
    done

# ' > ./logs/sample-imagenet64-euler.out 2>&1 &