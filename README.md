# DiffCSE: Difference-based Contrastive Learning for Sentence Embeddings

## Setups

### Requirements
* Python 3.9.5

### Install our customized Transformers package
```
cd transformers-4.2.1
pip install .
```
> If you have already install transformers==4.2.1 through pip, you need to put `modeling_bert.py` into `<your_python_env>/site-packages/transformers/models/bert/modeling_bert.py` and `modeling_roberta.py` into `<your_python_env>/site-packages/transformers/models/bert/modeling_roberta.py`.
> We modify these two files in the package so that we can perform _conditional_ pretraining tasks using BERT/RoBERTa. If possible, please directly pip install our customized Transformers package.

### Install other packages
```
pip install -r requirements.txt
```

### Download the pretraining dataset
```
cd data
bash download_wiki.sh
```

### Download the downstream dataset
```
cd SentEval/data/downstream/
bash download_dataset.sh
```

## Training
(The same as `run_diffcse.sh`.)
```bash
python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir <your_output_model_dir> \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate 7e-6 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --logging_first_step \
    --logging_dir <your_logging_dir> \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --batchnorm \
    --lambda_weight 0.005 \
    --fp16 --masking_ratio 0.30
```

Our new arguments:
* `--lambda_weight`: the lambda coefficient mentioned in Section 3 of our paper.
* `--masking_ratio`: the masking ratio for MLM generator to randomly replace tokens.


Arguments from [SimCSE](https://github.com/princeton-nlp/SimCSE):
* `--train_file`: Training file path (`data/wiki1m_for_simcse.txt`). 
* `--model_name_or_path`: Pre-trained checkpoints to start with such as BERT-based models (`bert-base-uncased`, `bert-large-uncased`, etc.) and RoBERTa-based models (`RoBERTa-base`, `RoBERTa-large`).
* `--temp`: Temperature for the contrastive loss. We always use `0.05`.
* `--pooler_type`: Pooling method.
* `--mlp_only_train`: For unsupervised SimCSE or DiffCSE, it works better to train the model with MLP layer but test the model without it. You should use this argument when training unsupervised SimCSE/DiffCSE models.

For results in the paper, we use Nvidia 2080Ti GPUs with CUDA 11.2. Using different types of devices or different versions of CUDA/other softwares may lead to slightly different performance.

## Evaluation
```bash
python evaluation.py \
    --model_name_or_path <your_output_model_dir> \
    --pooler cls_before_pooler \
    --task_set sts \
    --mode test
```

For more detailed information, please check [SimCSE's GitHub repo](https://github.com/princeton-nlp/SimCSE).


## Pretrained models

* DiffCSE-BERT-base (STS): https://drive.google.com/file/d/1CIxxsruPsscrOJPT42FIgyQBaOe5xrAj/view?usp=sharing
* DiffCSE-BERT-base (transfer tasks): https://drive.google.com/file/d/1IzUs3Xa6Be4t2t0TZIPb-dfnD54pneHw/view?usp=sharing
* DiffCSE-RoBERTa-base (STS): https://drive.google.com/file/d/1qHEs0TAOMkLQR4t_2VM5g4ePWOseE81j/view?usp=sharing
* DiffCSE-RoBERTa-base (transfer tasks): https://drive.google.com/file/d/1vGsjkYQU2w_n4_cUy2u9T5ywSo1Vq4Pl/view?usp=sharing

You can use [gdown](https://pypi.org/project/gdown/) to download them. And then use
```
tar zxvf [downloaded tar.gz file]
```
to extract the models.