#!/bin/bash

LR=2e-6
MASK=0.15
LAMBDA=0.05

for SEED in $1
do
	python train.py \
	    --model_name_or_path bert-base-uncased \
	    --train_file data/wiki1m_for_simcse.txt \
	    --output_dir results/bert_trans_sd$SEED \
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
	    --logging_dir logs/bert_trans_sd$SEED \
	    --temp 0.05 \
	    --do_train \
	    --do_eval \
	    --batchnorm \
	    --lambda_weight $LAMBDA \
	    --fp16 --masking_ratio $MASK --seed $SEED > log.bert_trans_sd$SEED 2>&1
  python evaluation.py \
    --model_name_or_path results/bert_trans_sd$SEED \
    --pooler cls_before_pooler \
    --task_set full \
    --mode test > log.eval.bert_trans_sd$SEED 2>&1
done
