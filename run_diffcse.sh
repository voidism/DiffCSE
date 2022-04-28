#!/bin/bash

LR=7e-6
MASK=0.30
LAMBDA=0.005

python train.py \
    --model_name_or_path bert-base-uncased \
    --generator_name distilbert-base-uncased \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir your_output_dir \
    --num_train_epochs 2 \
    --per_device_train_batch_size 64 \
    --learning_rate $LR \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --logging_first_step \
    --logging_dir your_logging_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --batchnorm \
    --lambda_weight $LAMBDA \
    --fp16 --masking_ratio $MASK
