#!/bin/bash

DEVICES="0,1"
STEPS="4" 
EPOCHS_LIST="200" 
SOLVER="euler" 
TRAIN_SEED="30000-30255"
USE_FINAL_LIST="False" 
DI_SEEDS="0" 
USE_RES_LIST="False" 
SHUFFLE_LIST="False" 

GT_SOLVER="dpmpp" 
SCHEDULE_TYPE="discrete"  
DATASET="ms_coco"
PORT="1237"
BATCH="6" # per-GPU batch size
SEEDS="0-29999"
EMB_DIR="trained_emb"
LOSS_CONFIG="configs/loss/coco.yaml"
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
                        OUTDIR="samples_di/$DATASET/$SOLVER/STEP$STEP/use_final=$USE_FINAL/use_res=$USE_RES/di_seed=$DI_SEED/shuffle=$SHUFFLE/loss_id=$LOSS_ID/$DATASET-$SOLVER-nfe${NFE}_epoch-$EPOCHS-seed$TRAIN_SEED"
                        echo OUTDIR="$OUTDIR"
                        echo STEP"$STEP"-EPOCHS"$EPOCHS"
                        echo LOSS_ID="$LOSS_ID"

                        if [ "$STEP" -eq 4 ]; then
                            GT_NUM_STEPS=22
                            LR=0.05
                        elif [ "$STEP" -eq 5 ]; then
                            GT_NUM_STEPS=21
                            LR=0.05
                        elif [ "$STEP" -eq 6 ]; then
                            GT_NUM_STEPS=21
                            LR=0.05
                        elif [ "$STEP" -eq 7 ]; then
                            GT_NUM_STEPS=25
                            LR=0.05
                        else
                            GT_NUM_STEPS=25
                            LR=0.05
                        fi

                        SOLVER_FLAGS="--num_steps=$STEP"
                        SCHEDULE_FLAGS="--schedule_type=$SCHEDULE_TYPE --schedule_rho=1"
                        ADDITIONAL_FLAGS="--max_order=2 --predict_x0=False --lower_order_final=True"
                        GUIDANCE_FLAGS="--guidance_type=cfg --guidance_rate=7.5"

                        python sample_coco.py --gpus "$DEVICES" --master_port "$PORT" \
                            --dataset_name="$DATASET" --batch "$BATCH" $SOLVER_FLAGS $SCHEDULE_FLAGS \
                            --outdir "$OUTDIR" --epochs "$EPOCHS" --seeds "$SEEDS" --lr "$LR" --solver "$SOLVER" \
                            --port "$PORT" --train_seed "$TRAIN_SEED" --save_npzs False --use_final "$USE_FINAL" $GUIDANCE_FLAGS \
                            --di_seed "$DI_SEED" --use_res "$USE_RES" --shuffle "$SHUFFLE" \
                            --gt_num_steps "$GT_NUM_STEPS" --gt_solver "$GT_SOLVER" \
                            --emb_dir "$EMB_DIR" --loss_config "$LOSS_CONFIG" $ADDITIONAL_FLAGS
                    done
                done
            done
        done
    done
done
