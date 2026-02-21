#!/bin/bash
# Optimized training script with automatic performance enhancements

if [ $epoch_time ]; then
    EPOCH_TIME=$epoch_time
else
    EPOCH_TIME=20
fi

if [ $out_dir ]; then
    OUT_DIR=$out_dir
else
    OUT_DIR='./model_out/optimized'
fi

if [ $cfg ]; then
    CFG=$cfg
else
    CFG='configs/swin_tiny_patch4_window7_224_lite.yaml'
fi

if [ $data_dir ]; then
    DATA_DIR=$data_dir
else
    DATA_DIR='datasets/data/Synapse'
fi

if [ $learning_rate ]; then
    LEARNING_RATE=$learning_rate
else
    LEARNING_RATE=0.01
fi

if [ $img_size ]; then
    IMG_SIZE=$img_size
else
    IMG_SIZE=224
fi

if [ $batch_size ]; then
    BATCH_SIZE=$batch_size
else
    BATCH_SIZE=24
fi

if [ $num_workers ]; then
    NUM_WORKERS=$num_workers
else
    NUM_WORKERS=2
fi

# AMP and optimization flags
if [ $use_amp ]; then
    USE_AMP=$use_amp
else
    USE_AMP='O1'
fi

echo "=========================================="
echo "OPTIMIZED TRAINING START"
echo "=========================================="
echo "Epochs: $EPOCH_TIME"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Num Workers: $NUM_WORKERS"
echo "AMP Level: $USE_AMP"
echo "=========================================="

python train.py \
    --dataset Synapse \
    --cfg $CFG \
    --root_path $DATA_DIR \
    --max_epochs $EPOCH_TIME \
    --output_dir $OUT_DIR \
    --img_size $IMG_SIZE \
    --base_lr $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --use-checkpoint \
    --amp-opt-level $USE_AMP
