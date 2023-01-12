## Tiny Models

- Baseline

```shell
export CUDA_VISIBLE_DEVICES=0
export PLM=google/bert_uncased_L-2_H-128_A-2
export DS=squad
export NE=3
export LR=2e-4
export WR=0.1
python run_squad.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --dataset_name ${DS} --max_seq_length 512 --doc_stride 128 \
  --do_train --load_best_model_at_end \
  --max_answer_length 30 --n_best_size 20 \
  --logging_dir runs/${DS} --logging_steps 200 \
  --output_dir ckpts/${DS} --save_steps 1000 \
  --evaluation_strategy steps --eval_steps 1000 --save_total_limit 3 \
  --metric_for_best_model f1 --greater_is_better True \
  --per_device_train_batch_size 12 --per_device_eval_batch_size 24 \
  --num_train_epochs ${NE} --learning_rate ${LR} --warmup_ratio ${WR}
```

- Comparative Loss with 2 CmpDrop

```shell
export CUDA_VISIBLE_DEVICES=0
export PLM=google/bert_uncased_L-2_H-128_A-2
export DS=squad
export NE=3
export LR=2e-4
export WR=0.1
export ND=2
export NC=0
export TGT=ultra
python run_squad.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --dataset_name ${DS} --max_seq_length 512 --doc_stride 128 \
  --do_train --load_best_model_at_end \
  --max_answer_length 30 --n_best_size 20 \
  --logging_dir runs/${DS} --logging_steps 200 \
  --output_dir ckpts/${DS} --save_steps 1000 \
  --evaluation_strategy steps --eval_steps 1000 --save_total_limit 3 \
  --metric_for_best_model f1 --greater_is_better True \
  --per_device_train_batch_size 12 --per_device_eval_batch_size 24 \
  --num_train_epochs ${NE} --learning_rate ${LR} --warmup_ratio ${WR} \
  --n_drop ${ND} --n_crop ${NC} --just_cmp --loss_target ${TGT}
```

## Small Models

Switch `PLM` between `google/bert_uncased_L-12_H-256_A-4` and `google/electra-small-discriminator`.

- Baseline

```shell
export CUDA_VISIBLE_DEVICES=0
export PLM=google/bert_uncased_L-12_H-256_A-4
export DS=squad
export NE=2
export LR=1e-4
export WR=0.1
python run_squad.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --dataset_name ${DS} --max_seq_length 512 --doc_stride 128 \
  --do_train --load_best_model_at_end \
  --max_answer_length 30 --n_best_size 20 \
  --logging_dir runs/${DS} --logging_steps 200 \
  --output_dir ckpts/${DS} --save_steps 1000 \
  --evaluation_strategy steps --eval_steps 1000 --save_total_limit 3 \
  --metric_for_best_model f1 --greater_is_better True \
  --per_device_train_batch_size 12 --per_device_eval_batch_size 24 \
  --num_train_epochs ${NE} --learning_rate ${LR} --warmup_ratio ${WR}
```

- Comparative Loss with 2 CmpDrop

```shell
export CUDA_VISIBLE_DEVICES=0
export PLM=google/bert_uncased_L-12_H-256_A-4
export DS=squad
export NE=2
export LR=1e-4
export WR=0.1
export ND=2
export NC=0
export TGT=ultra
python run_squad.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --dataset_name ${DS} --max_seq_length 512 --doc_stride 128 \
  --do_train --load_best_model_at_end \
  --max_answer_length 30 --n_best_size 20 \
  --logging_dir runs/${DS} --logging_steps 200 \
  --output_dir ckpts/${DS} --save_steps 1000 \
  --evaluation_strategy steps --eval_steps 1000 --save_total_limit 3 \
  --metric_for_best_model f1 --greater_is_better True \
  --per_device_train_batch_size 12 --per_device_eval_batch_size 24 \
  --num_train_epochs ${NE} --learning_rate ${LR} --warmup_ratio ${WR} \
  --n_drop ${ND} --n_crop ${NC} --just_cmp --loss_target ${TGT}
```

## Medium Models

- Baseline

```shell
export CUDA_VISIBLE_DEVICES=0
export PLM=google/bert_uncased_L-8_H-512_A-8
export DS=squad
export NE=2
export LR=5e-5
export WR=0.1
python run_squad.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --dataset_name ${DS} --max_seq_length 512 --doc_stride 128 \
  --do_train --load_best_model_at_end \
  --max_answer_length 30 --n_best_size 20 \
  --logging_dir runs/${DS} --logging_steps 200 \
  --output_dir ckpts/${DS} --save_steps 1000 \
  --evaluation_strategy steps --eval_steps 1000 --save_total_limit 3 \
  --metric_for_best_model f1 --greater_is_better True \
  --per_device_train_batch_size 12 --per_device_eval_batch_size 24 \
  --num_train_epochs ${NE} --learning_rate ${LR} --warmup_ratio ${WR}
```

- Comparative Loss with 2 CmpDrop

```shell
export CUDA_VISIBLE_DEVICES=0
export PLM=google/bert_uncased_L-8_H-512_A-8
export DS=squad
export NE=2
export LR=5e-5
export WR=0.1
export ND=2
export NC=0
export TGT=ultra
python run_squad.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --dataset_name ${DS} --max_seq_length 512 --doc_stride 128 \
  --do_train --load_best_model_at_end \
  --max_answer_length 30 --n_best_size 20 \
  --logging_dir runs/${DS} --logging_steps 200 \
  --output_dir ckpts/${DS} --save_steps 1000 \
  --evaluation_strategy steps --eval_steps 1000 --save_total_limit 3 \
  --metric_for_best_model f1 --greater_is_better True \
  --per_device_train_batch_size 12 --per_device_eval_batch_size 24 \
  --num_train_epochs ${NE} --learning_rate ${LR} --warmup_ratio ${WR} \
  --n_drop ${ND} --n_crop ${NC} --just_cmp --loss_target ${TGT}
```

## Base Models

Switch `PLM` between `bert-base-uncased` and `google/electra-base-discriminator`.

- Baseline

```shell
export CUDA_VISIBLE_DEVICES=0
export PLM=bert-base-uncased
export DS=squad
export NE=2
export LR=3e-5
export WR=0.1
python run_squad.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --dataset_name ${DS} --max_seq_length 512 --doc_stride 128 \
  --do_train --load_best_model_at_end \
  --max_answer_length 30 --n_best_size 20 \
  --logging_dir runs/${DS} --logging_steps 200 \
  --output_dir ckpts/${DS} --save_steps 1000 \
  --evaluation_strategy steps --eval_steps 1000 --save_total_limit 3 \
  --metric_for_best_model f1 --greater_is_better True \
  --per_device_train_batch_size 12 --per_device_eval_batch_size 24 \
  --num_train_epochs ${NE} --learning_rate ${LR} --warmup_ratio ${WR}
```

- Comparative Loss with 2 CmpDrop

```shell
export CUDA_VISIBLE_DEVICES=0
export PLM=bert-base-uncased
export DS=squad
export NE=2
export LR=3e-5
export WR=0.1
export ND=2
export NC=0
export TGT=ultra
python run_squad.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --dataset_name ${DS} --max_seq_length 512 --doc_stride 128 \
  --do_train --load_best_model_at_end \
  --max_answer_length 30 --n_best_size 20 \
  --logging_dir runs/${DS} --logging_steps 200 \
  --output_dir ckpts/${DS} --save_steps 1000 \
  --evaluation_strategy steps --eval_steps 1000 --save_total_limit 3 \
  --metric_for_best_model f1 --greater_is_better True \
  --per_device_train_batch_size 12 --per_device_eval_batch_size 24 \
  --num_train_epochs ${NE} --learning_rate ${LR} --warmup_ratio ${WR} \
  --n_drop ${ND} --n_crop ${NC} --just_cmp --loss_target ${TGT}
```

## Large Models

### BERT-large

- Baseline

```shell
export CUDA_VISIBLE_DEVICES=0,1,2
export PLM=bert-large-uncased
export DS=squad
export NE=2
export LR=3e-5
export WR=0.1
python run_squad.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --dataset_name ${DS} --max_seq_length 384 --doc_stride 128 \
  --do_train --load_best_model_at_end \
  --max_answer_length 30 --n_best_size 20 \
  --logging_dir runs/${DS} --logging_steps 200 \
  --output_dir ckpts/${DS} --save_steps 1000 \
  --evaluation_strategy steps --eval_steps 1000 --save_total_limit 3 \
  --metric_for_best_model f1 --greater_is_better True \
  --per_device_train_batch_size 8 --per_device_eval_batch_size 8 \
  --num_train_epochs ${NE} --learning_rate ${LR} --warmup_ratio ${WR}
```

- Comparative Loss with 2 CmpDrop

```shell
export CUDA_VISIBLE_DEVICES=0,1,2
export PLM=bert-large-uncased
export DS=squad
export NE=2
export LR=3e-5
export WR=0.1
export ND=2
export NC=0
export TGT=first
python run_squad.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --dataset_name ${DS} --max_seq_length 384 --doc_stride 128 \
  --do_train --load_best_model_at_end \
  --max_answer_length 30 --n_best_size 20 \
  --logging_dir runs/${DS} --logging_steps 200 \
  --output_dir ckpts/${DS} --save_steps 1000 \
  --evaluation_strategy steps --eval_steps 1000 --save_total_limit 3 \
  --metric_for_best_model f1 --greater_is_better True \
  --per_device_train_batch_size 8 --per_device_eval_batch_size 8 \
  --num_train_epochs ${NE} --learning_rate ${LR} --warmup_ratio ${WR} \
  --n_drop ${ND} --n_crop ${NC} --just_cmp --loss_target ${TGT}
```

### ELECTRA-large

- Baseline

```shell
export CUDA_VISIBLE_DEVICES=0,1,2
export PLM=google/electra-large-discriminator
export DS=squad
export NE=2
export LR=1e-5
export WR=0.1
python run_squad.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --dataset_name ${DS} --max_seq_length 512 --doc_stride 128 \
  --do_train --load_best_model_at_end \
  --max_answer_length 30 --n_best_size 20 \
  --logging_dir runs/${DS} --logging_steps 200 \
  --output_dir ckpts/${DS} --save_steps 1000 \
  --evaluation_strategy steps --eval_steps 1000 --save_total_limit 3 \
  --metric_for_best_model f1 --greater_is_better True \
  --per_device_train_batch_size 4 --per_device_eval_batch_size 8 \
  --num_train_epochs ${NE} --learning_rate ${LR} --warmup_ratio ${WR}
```

- Comparative Loss with 2 CmpDrop

```shell
export CUDA_VISIBLE_DEVICES=0,1,2
export PLM=google/electra-large-discriminator
export DS=squad
export NE=2
export LR=1e-5
export WR=0.1
export ND=2
export NC=0
export TGT=first
python run_squad.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --dataset_name ${DS} --max_seq_length 512 --doc_stride 128 \
  --do_train --load_best_model_at_end \
  --max_answer_length 30 --n_best_size 20 \
  --logging_dir runs/${DS} --logging_steps 200 \
  --output_dir ckpts/${DS} --save_steps 1000 \
  --evaluation_strategy steps --eval_steps 1000 --save_total_limit 3 \
  --metric_for_best_model f1 --greater_is_better True \
  --per_device_train_batch_size 4 --per_device_eval_batch_size 8 \
  --num_train_epochs ${NE} --learning_rate ${LR} --warmup_ratio ${WR} \
  --n_drop ${ND} --n_crop ${NC} --just_cmp --loss_target ${TGT}
```