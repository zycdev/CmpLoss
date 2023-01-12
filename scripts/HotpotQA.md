## Main Experiments

- Baseline (Longformer)

```shell
export CUDA_VISIBLE_DEVICES=0
export PLM=allenai/longformer-base-4096
export DS=hotpot
export NE=3
export LR=3e-5
export WR=0.1
python run_hotpot.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --corpus_file data/hotpot/corpus.distractor.tsv \
  --train_file data/hotpot/train.tsv \
  --validation_file data/hotpot/dev.tsv \
  --do_train --load_best_model_at_end \
  --max_seq_length 2048 --max_q_len 160 --max_p_len 256 \
  --max_answer_length 64 --n_best_size 5 \
  --logging_dir runs/${DS} --logging_steps 200 \
  --output_dir ckpts/${DS} --save_steps 1000 \
  --evaluation_strategy steps --eval_steps 1000 --save_total_limit 3 \
  --metric_for_best_model f1 --greater_is_better True \
  --per_device_train_batch_size 6 --per_device_eval_batch_size 6 \
  --num_train_epochs ${NE} --learning_rate ${LR} --warmup_ratio ${WR} --fp16
```

- Comparative Loss with CmpDrop

```shell
export CUDA_VISIBLE_DEVICES=0
export PLM=allenai/longformer-base-4096
export DS=hotpot
export NE=3
export LR=3e-5
export WR=0.1
export ND=1
export NC=0
export TGT=ultra
python run_hotpot.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --corpus_file data/hotpot/corpus.distractor.tsv \
  --train_file data/hotpot/train.tsv \
  --validation_file data/hotpot/dev.tsv \
  --do_train --load_best_model_at_end \
  --max_seq_length 2048 --max_q_len 160 --max_p_len 256 \
  --max_answer_length 64 --n_best_size 5 \
  --logging_dir runs/${DS} --logging_steps 200 \
  --output_dir ckpts/${DS} --save_steps 1000 \
  --evaluation_strategy steps --eval_steps 1000 --save_total_limit 3 \
  --metric_for_best_model f1 --greater_is_better True \
  --per_device_train_batch_size 3 --per_device_eval_batch_size 3 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs ${NE} --learning_rate ${LR} --warmup_ratio ${WR} --fp16 \
  --n_drop ${ND} --n_crop ${NC} --just_cmp --loss_target ${TGT}
```

- Comparative Loss with CmpDrop

```shell
export CUDA_VISIBLE_DEVICES=0
export PLM=allenai/longformer-base-4096
export DS=hotpot
export NE=3
export LR=3e-5
export WR=0.1
export ND=0
export NC=1
export TGT=ultra
python run_hotpot.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --corpus_file data/hotpot/corpus.distractor.tsv \
  --train_file data/hotpot/train.tsv \
  --validation_file data/hotpot/dev.tsv \
  --do_train --load_best_model_at_end \
  --max_seq_length 2048 --max_q_len 160 --max_p_len 256 \
  --max_answer_length 64 --n_best_size 5 \
  --logging_dir runs/${DS} --logging_steps 200 \
  --output_dir ckpts/${DS} --save_steps 1000 \
  --evaluation_strategy steps --eval_steps 1000 --save_total_limit 3 \
  --metric_for_best_model f1 --greater_is_better True \
  --per_device_train_batch_size 3 --per_device_eval_batch_size 3 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs ${NE} --learning_rate ${LR} --warmup_ratio ${WR} --fp16 \
  --n_drop ${ND} --n_crop ${NC} --just_cmp --loss_target ${TGT}
```

- Comparative Loss with CmpCrop and CmpDrop

```shell
export CUDA_VISIBLE_DEVICES=0
export PLM=allenai/longformer-base-4096
export DS=hotpot
export NE=3
export LR=3e-5
export WR=0.1
export ND=1
export NC=1
export TGT=ultra
python run_hotpot.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --corpus_file data/hotpot/corpus.distractor.tsv \
  --train_file data/hotpot/train.tsv \
  --validation_file data/hotpot/dev.tsv \
  --do_train --load_best_model_at_end \
  --max_seq_length 2048 --max_q_len 160 --max_p_len 256 \
  --max_answer_length 64 --n_best_size 5 \
  --logging_dir runs/${DS} --logging_steps 200 \
  --output_dir ckpts/${DS} --save_steps 1000 \
  --evaluation_strategy steps --eval_steps 1000 --save_total_limit 3 \
  --metric_for_best_model f1 --greater_is_better True \
  --per_device_train_batch_size 3 --per_device_eval_batch_size 3 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs ${NE} --learning_rate ${LR} --warmup_ratio ${WR} --fp16 \
  --n_drop ${ND} --n_crop ${NC} --just_cmp --loss_target ${TGT}
```

## Effect of Weighting Strategy

- AVERAGE

```shell
export CUDA_VISIBLE_DEVICES=0
export PLM=allenai/longformer-base-4096
export DS=hotpot
export NE=3
export LR=3e-5
export WR=0.1
export ND=1
export NC=1
export TGT=ultra
python run_hotpot.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --corpus_file data/hotpot/corpus.distractor.tsv \
  --train_file data/hotpot/train.tsv \
  --validation_file data/hotpot/dev.tsv \
  --do_train --load_best_model_at_end \
  --max_seq_length 2048 --max_q_len 160 --max_p_len 256 \
  --max_answer_length 64 --n_best_size 5 \
  --logging_dir runs/${DS} --logging_steps 200 \
  --output_dir ckpts/${DS} --save_steps 1000 \
  --evaluation_strategy steps --eval_steps 1000 --save_total_limit 3 \
  --metric_for_best_model f1 --greater_is_better True \
  --per_device_train_batch_size 3 --per_device_eval_batch_size 3 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs ${NE} --learning_rate ${LR} --warmup_ratio ${WR} --fp16 \
  --n_drop ${ND} --n_crop ${NC} --loss_primary avg --cr_weight 0
```

- FIRST

```shell
export CUDA_VISIBLE_DEVICES=0
export PLM=allenai/longformer-base-4096
export DS=hotpot
export NE=3
export LR=3e-5
export WR=0.1
export ND=1
export NC=1
export TGT=ultra
python run_hotpot.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --corpus_file data/hotpot/corpus.distractor.tsv \
  --train_file data/hotpot/train.tsv \
  --validation_file data/hotpot/dev.tsv \
  --do_train --load_best_model_at_end \
  --max_seq_length 2048 --max_q_len 160 --max_p_len 256 \
  --max_answer_length 64 --n_best_size 5 \
  --logging_dir runs/${DS} --logging_steps 200 \
  --output_dir ckpts/${DS} --save_steps 1000 \
  --evaluation_strategy steps --eval_steps 1000 --save_total_limit 3 \
  --metric_for_best_model f1 --greater_is_better True \
  --per_device_train_batch_size 3 --per_device_eval_batch_size 3 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs ${NE} --learning_rate ${LR} --warmup_ratio ${WR} --fp16 \
  --n_drop ${ND} --n_crop ${NC} --loss_primary fst --cr_weight 0
```

- MAX

```shell
export CUDA_VISIBLE_DEVICES=0
export PLM=allenai/longformer-base-4096
export DS=hotpot
export NE=3
export LR=3e-5
export WR=0.1
export ND=1
export NC=1
export TGT=ultra
python run_hotpot.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --corpus_file data/hotpot/corpus.distractor.tsv \
  --train_file data/hotpot/train.tsv \
  --validation_file data/hotpot/dev.tsv \
  --do_train --load_best_model_at_end \
  --max_seq_length 2048 --max_q_len 160 --max_p_len 256 \
  --max_answer_length 64 --n_best_size 5 \
  --logging_dir runs/${DS} --logging_steps 200 \
  --output_dir ckpts/${DS} --save_steps 1000 \
  --evaluation_strategy steps --eval_steps 1000 --save_total_limit 3 \
  --metric_for_best_model f1 --greater_is_better True \
  --per_device_train_batch_size 3 --per_device_eval_batch_size 3 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs ${NE} --learning_rate ${LR} --warmup_ratio ${WR} --fp16 \
  --n_drop ${ND} --n_crop ${NC} --loss_primary max --cr_weight 0
```

## Effect of Context Size

```shell
export CUDA_VISIBLE_DEVICES=0
export PLM=allenai/longformer-base-4096
export DIR=ckpts/hotpot/allenai
for CKPT in "longformer-base-4096#l2048#b6_e3_lr3e-05w0.1_fp16" "longformer-base-4096#l2048#b3x2_e3_lr3e-05w0.1_p1_ultra_fp16" "longformer-base-4096#l2048#b3x2_e3_lr3e-05w0.1_p+1_ultra_fp16" "longformer-base-4096#l2048#b3x2_e3_lr3e-05w0.1_p1+1_ultra_fp16"
do
  for NP in $(seq 2 2)
    do
      python run_hotpot.py \
        --tokenizer_name ${PLM} --model_name_or_path ${DIR}/${CKPT} \
        --corpus_file data/hotpot/corpus.distractor.tsv \
        --validation_file data/hotpot/dev.tsv \
        --do_eval --per_device_eval_batch_size 6 \
        --max_seq_length 4096 --max_q_len 160 --max_p_len 256 \
        --max_answer_length 64 --n_best_size 5 \
        --max_p_num ${NP} --output_dir ${DIR}/${CKPT}/${NP}
    done
done
```

## Effect of Dropout Rate

```shell
export CUDA_VISIBLE_DEVICES=0
export PLM=allenai/longformer-base-4096
export DIR=ckpts/hotpot/allenai
for CKPT in "longformer-base-4096#l2048#b6_e3_lr3e-05w0.1_fp16" "longformer-base-4096#l2048#b3x2_e3_lr3e-05w0.1_p1_ultra_fp16" "longformer-base-4096#l2048#b3x2_e3_lr3e-05w0.1_p+1_ultra_fp16" "longformer-base-4096#l2048#b3x2_e3_lr3e-05w0.1_p1+1_ultra_fp16"
do
  for DR in $(seq 1 3)
  do
    python run_hotpot.py \
      --tokenizer_name ${PLM} --model_name_or_path ${CKPT} \
      --corpus_file data/hotpot/corpus.distractor.tsv \
      --validation_file data/hotpot/dev.tsv \
      --do_eval --per_device_eval_batch_size 6 \
      --max_seq_length 4096 --max_q_len 160 --max_p_len 256 \
      --max_answer_length 64 --n_best_size 5 \
      --force_dropout ${DR} --output_dir ${CKPT}/_${DR}
  done
done
```

## Effect of Data Size

```shell
export CUDA_VISIBLE_DEVICES=0
export PLM=allenai/longformer-base-4096
export DS=hotpot
export NE=10
export LR=3e-5
export WR=0.1
python run_hotpot.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --corpus_file data/hotpot/corpus.distractor.tsv \
  --train_file data/hotpot/train.tsv \
  --validation_file data/hotpot/dev.tsv \
  --do_train --load_best_model_at_end \
  --max_seq_length 2048 --max_q_len 160 --max_p_len 256 \
  --max_answer_length 64 --n_best_size 5 \
  --logging_dir runs/${DS} --logging_steps 50 \
  --output_dir ckpts/${DS} --save_steps 100 \
  --evaluation_strategy steps --eval_steps 100 --save_total_limit 3 \
  --metric_for_best_model f1 --greater_is_better True \
  --per_device_train_batch_size 6 --per_device_eval_batch_size 6 \
  --num_train_epochs ${NE} --learning_rate ${LR} --warmup_ratio ${WR} --fp16 \
  --max_train_samples 1000 --comment 1000

# CmpDrop only
export ND=1
export NC=0
export TGT=ultra
python run_hotpot.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --corpus_file data/hotpot/corpus.distractor.tsv \
  --train_file data/hotpot/train.tsv \
  --validation_file data/hotpot/dev.tsv \
  --do_train --load_best_model_at_end \
  --max_seq_length 2048 --max_q_len 160 --max_p_len 256 \
  --max_answer_length 64 --n_best_size 5 \
  --logging_dir runs/${DS} --logging_steps 50 \
  --output_dir ckpts/${DS} --save_steps 100 \
  --evaluation_strategy steps --eval_steps 100 --save_total_limit 3 \
  --metric_for_best_model f1 --greater_is_better True \
  --per_device_train_batch_size 3 --per_device_eval_batch_size 3 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs ${NE} --learning_rate ${LR} --warmup_ratio ${WR} --fp16 \
  --n_drop ${ND} --n_crop ${NC} --just_cmp --loss_target ${TGT} \
  --max_train_samples 1000 --comment 1000

# CmpCrop only
export ND=0
export NC=1
export TGT=ultra
python run_hotpot.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --corpus_file data/hotpot/corpus.distractor.tsv \
  --train_file data/hotpot/train.tsv \
  --validation_file data/hotpot/dev.tsv \
  --do_train --load_best_model_at_end \
  --max_seq_length 2048 --max_q_len 160 --max_p_len 256 \
  --max_answer_length 64 --n_best_size 5 \
  --logging_dir runs/${DS} --logging_steps 50 \
  --output_dir ckpts/${DS} --save_steps 100 \
  --evaluation_strategy steps --eval_steps 100 --save_total_limit 3 \
  --metric_for_best_model f1 --greater_is_better True \
  --per_device_train_batch_size 3 --per_device_eval_batch_size 3 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs ${NE} --learning_rate ${LR} --warmup_ratio ${WR} --fp16 \
  --n_drop ${ND} --n_crop ${NC} --just_cmp --loss_target ${TGT} \
  --max_train_samples 1000 --comment 1000

# CmpDrop + CmpCrop
export ND=1
export NC=1
export TGT=ultra
python run_hotpot.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --corpus_file data/hotpot/corpus.distractor.tsv \
  --train_file data/hotpot/train.tsv \
  --validation_file data/hotpot/dev.tsv \
  --do_train --load_best_model_at_end \
  --max_seq_length 2048 --max_q_len 160 --max_p_len 256 \
  --max_answer_length 64 --n_best_size 5 \
  --logging_dir runs/${DS} --logging_steps 50 \
  --output_dir ckpts/${DS} --save_steps 100 \
  --evaluation_strategy steps --eval_steps 100 --save_total_limit 3 \
  --metric_for_best_model f1 --greater_is_better True \
  --per_device_train_batch_size 3 --per_device_eval_batch_size 3 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs ${NE} --learning_rate ${LR} --warmup_ratio ${WR} --fp16 \
  --n_drop ${ND} --n_crop ${NC} --just_cmp --loss_target ${TGT} \
  --max_train_samples 1000 --comment 1000
```

