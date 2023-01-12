Common variables:
```shell
export CUDA_VISIBLE_DEVICES=0
export PLM=bert-base-cased
export LOG_STEPS=50
export EVAL_STEPS=100
```

## CoLA

```shell
export TASK=cola
```

- Baseline

```shell
export NE=10
export LR=1e-5
export WR=0.06
python run_glue.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --task_name ${TASK} --max_seq_length 128 --pad_to_max_length \
  --do_train --do_eval --load_best_model_at_end \
  --logging_dir runs/glue/${TASK} --logging_steps ${LOG_STEPS} \
  --output_dir ckpts/glue/${TASK} --save_steps ${EVAL_STEPS} \
  --metric_for_best_model accuracy --greater_is_better True \
  --evaluation_strategy steps --eval_steps ${EVAL_STEPS} --save_total_limit 2 \
  --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --num_train_epochs ${NE} --max_steps 5336 \
  --warmup_ratio ${WR} --warmup_steps 320 \
  --learning_rate ${LR} --weight_decay 0.01 --fp16
```

- Comparative Loss

```shell
export NE=10
export LR=1e-5
export WR=0.06
export ND=4
export TGT=ultra
python run_glue.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --task_name ${TASK} --max_seq_length 128 --pad_to_max_length \
  --do_train --do_eval --load_best_model_at_end \
  --logging_dir runs/glue/${TASK} --logging_steps ${LOG_STEPS} \
  --output_dir ckpts/glue/${TASK} --save_steps ${EVAL_STEPS} \
  --metric_for_best_model accuracy --greater_is_better True \
  --evaluation_strategy steps --eval_steps ${EVAL_STEPS} --save_total_limit 2 \
  --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --num_train_epochs ${NE} --max_steps 5336 \
  --warmup_ratio ${WR} --warmup_steps 320 \
  --learning_rate ${LR} --weight_decay 0.01 --fp16 \
  --n_drop ${ND} --just_cmp --loss_target ${TGT}
```

## SST-2

```shell
export TASK=sst2
```

- Baseline

```shell
export NE=10
export LR=1e-5
export WR=0.06
python run_glue.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --task_name ${TASK} --max_seq_length 128 --pad_to_max_length \
  --do_train --do_eval --load_best_model_at_end \
  --logging_dir runs/glue/${TASK} --logging_steps ${LOG_STEPS} \
  --output_dir ckpts/glue/${TASK} --save_steps ${EVAL_STEPS} \
  --metric_for_best_model accuracy --greater_is_better True \
  --evaluation_strategy steps --eval_steps ${EVAL_STEPS} --save_total_limit 2 \
  --per_device_train_batch_size 32 --per_device_eval_batch_size 32 \
  --num_train_epochs ${NE} --max_steps 20935 \
  --warmup_ratio ${WR} --warmup_steps 1256 \
  --learning_rate ${LR} --weight_decay 0.01 --fp16
```

- Comparative Loss

```shell
export NE=3
export LR=2e-5
export WR=0.0
export ND=2
export TGT=ultra
python run_glue.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --task_name ${TASK} --max_seq_length 128 --pad_to_max_length \
  --do_train --do_eval --load_best_model_at_end \
  --logging_dir runs/glue/${TASK} --logging_steps ${LOG_STEPS} \
  --output_dir ckpts/glue/${TASK} --save_steps ${EVAL_STEPS} \
  --metric_for_best_model accuracy --greater_is_better True \
  --evaluation_strategy steps --eval_steps ${EVAL_STEPS} --save_total_limit 2 \
  --per_device_train_batch_size 32 --per_device_eval_batch_size 64 \
  --num_train_epochs ${NE} --learning_rate ${LR} --warmup_ratio ${WR} \
  --n_drop ${ND} --just_cmp --loss_target ${TGT}
```

## MRPC

```shell
export TASK=mrpc
```

- Baseline

```shell
export NE=10
export LR=1e-5
export WR=0.06
python run_glue.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --task_name ${TASK} --max_seq_length 128 --pad_to_max_length \
  --do_train --do_eval --load_best_model_at_end \
  --logging_dir runs/glue/${TASK} --logging_steps ${LOG_STEPS} \
  --output_dir ckpts/glue/${TASK} --save_steps ${EVAL_STEPS} \
  --metric_for_best_model accuracy --greater_is_better True \
  --evaluation_strategy steps --eval_steps ${EVAL_STEPS} --save_total_limit 2 \
  --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --num_train_epochs ${NE} --max_steps 2296 \
  --warmup_ratio ${WR} --warmup_steps 137 \
  --learning_rate ${LR} --weight_decay 0.01 --fp16
```

- Comparative Loss

```shell
export NE=10
export LR=2e-5
export WR=0.06
export ND=1
export TGT=ultra
python run_glue.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --task_name ${TASK} --max_seq_length 128 --pad_to_max_length \
  --do_train --do_eval --load_best_model_at_end \
  --logging_dir runs/glue/${TASK} --logging_steps ${LOG_STEPS} \
  --output_dir ckpts/glue/${TASK} --save_steps ${EVAL_STEPS} \
  --metric_for_best_model accuracy --greater_is_better True \
  --evaluation_strategy steps --eval_steps ${EVAL_STEPS} --save_total_limit 2 \
  --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --num_train_epochs ${NE} --max_steps 2296 \
  --warmup_ratio ${WR} --warmup_steps 137 \
  --learning_rate ${LR} --weight_decay 0.01 --fp16 \
  --n_drop ${ND} --just_cmp --loss_target ${TGT}
```

## STS-B

```shell
export TASK=stsb
```

- Baseline

```shell
export NE=10
export LR=1e-5
export WR=0.06
python run_glue.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --task_name ${TASK} --max_seq_length 128 --pad_to_max_length \
  --do_train --do_eval --load_best_model_at_end \
  --logging_dir runs/glue/${TASK} --logging_steps ${LOG_STEPS} \
  --output_dir ckpts/glue/${TASK} --save_steps ${EVAL_STEPS} \
  --metric_for_best_model accuracy --greater_is_better True \
  --evaluation_strategy steps --eval_steps ${EVAL_STEPS} --save_total_limit 2 \
  --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --num_train_epochs ${NE} --max_steps 3598 \
  --warmup_ratio ${WR} --warmup_steps 214 \
  --learning_rate ${LR} --weight_decay 0.01 --fp16
```

- Comparative Loss

```shell
export NE=10
export LR=1e-5
export WR=0.06
export ND=4
export TGT=first
python run_glue.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --task_name ${TASK} --max_seq_length 128 --pad_to_max_length \
  --do_train --do_eval --load_best_model_at_end \
  --logging_dir runs/glue/${TASK} --logging_steps ${LOG_STEPS} \
  --output_dir ckpts/glue/${TASK} --save_steps ${EVAL_STEPS} \
  --metric_for_best_model accuracy --greater_is_better True \
  --evaluation_strategy steps --eval_steps ${EVAL_STEPS} --save_total_limit 2 \
  --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --num_train_epochs ${NE} --max_steps 3598 \
  --warmup_ratio ${WR} --warmup_steps 214 \
  --learning_rate ${LR} --weight_decay 0.01 --fp16 \
  --n_drop ${ND} --just_cmp --loss_target ${TGT}
```

## QQP

```shell
export TASK=qqp
```

- Baseline

```shell
export NE=10
export LR=1e-5
export WR=0.25
python run_glue.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --task_name ${TASK} --max_seq_length 128 --pad_to_max_length \
  --do_train --do_eval --load_best_model_at_end \
  --logging_dir runs/glue/${TASK} --logging_steps ${LOG_STEPS} \
  --output_dir ckpts/glue/${TASK} --save_steps ${EVAL_STEPS} \
  --metric_for_best_model accuracy --greater_is_better True \
  --evaluation_strategy steps --eval_steps ${EVAL_STEPS} --save_total_limit 2 \
  --per_device_train_batch_size 32 --per_device_eval_batch_size 32 \
  --num_train_epochs ${NE} --max_steps 113272 \
  --warmup_ratio ${WR} --warmup_steps 28318 \
  --learning_rate ${LR} --weight_decay 0.01 --fp16
```

- Comparative Loss

```shell
export NE=3
export LR=2e-5
export WR=0.0
export ND=2
export TGT=first
python run_glue.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --task_name ${TASK} --max_seq_length 128 --pad_to_max_length \
  --do_train --do_eval --load_best_model_at_end \
  --logging_dir runs/glue/${TASK} --logging_steps ${LOG_STEPS} \
  --output_dir ckpts/glue/${TASK} --save_steps ${EVAL_STEPS} \
  --metric_for_best_model accuracy --greater_is_better True \
  --evaluation_strategy steps --eval_steps ${EVAL_STEPS} --save_total_limit 2 \
  --per_device_train_batch_size 32 --per_device_eval_batch_size 64 \
  --num_train_epochs ${NE} --learning_rate ${LR} --warmup_ratio ${WR} \
  --n_drop ${ND} --just_cmp --loss_target ${TGT}
```

## MNLI

```shell
export TASK=mnli
```

- Baseline

```shell
export NE=3
export LR=2e-5
export WR=0.0
python run_glue.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --task_name ${TASK} --max_seq_length 128 --pad_to_max_length \
  --do_train --do_eval --load_best_model_at_end \
  --logging_dir runs/glue/${TASK} --logging_steps ${LOG_STEPS} \
  --output_dir ckpts/glue/${TASK} --save_steps ${EVAL_STEPS} \
  --metric_for_best_model accuracy --greater_is_better True \
  --evaluation_strategy steps --eval_steps ${EVAL_STEPS} --save_total_limit 2 \
  --per_device_train_batch_size 32 --per_device_eval_batch_size 64 \
  --num_train_epochs ${NE} --learning_rate ${LR} --warmup_ratio ${WR}
```

- Comparative Loss

```shell
export NE=3
export LR=3e-5
export WR=0.0
export ND=3
export TGT=first
python run_glue.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --task_name ${TASK} --max_seq_length 128 --pad_to_max_length \
  --do_train --do_eval --load_best_model_at_end \
  --logging_dir runs/glue/${TASK} --logging_steps ${LOG_STEPS} \
  --output_dir ckpts/glue/${TASK} --save_steps ${EVAL_STEPS} \
  --metric_for_best_model accuracy --greater_is_better True \
  --evaluation_strategy steps --eval_steps ${EVAL_STEPS} --save_total_limit 2 \
  --per_device_train_batch_size 32 --per_device_eval_batch_size 64 \
  --num_train_epochs ${NE} --learning_rate ${LR} --warmup_ratio ${WR} \
  --n_drop ${ND} --just_cmp --loss_target ${TGT}
```

## QNLI

```shell
export TASK=qnli
```

- Baseline

```shell
export NE=3
export LR=2e-5
export WR=0.1
python run_glue.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --task_name ${TASK} --max_seq_length 128 --pad_to_max_length \
  --do_train --do_eval --load_best_model_at_end \
  --logging_dir runs/glue/${TASK} --logging_steps ${LOG_STEPS} \
  --output_dir ckpts/glue/${TASK} --save_steps ${EVAL_STEPS} \
  --metric_for_best_model accuracy --greater_is_better True \
  --evaluation_strategy steps --eval_steps ${EVAL_STEPS} --save_total_limit 2 \
  --per_device_train_batch_size 32 --per_device_eval_batch_size 64 \
  --num_train_epochs ${NE} --learning_rate ${LR} --warmup_ratio ${WR}
```

- Comparative Loss

```shell
export NE=3
export LR=3e-5
export WR=0.1
export ND=4
export TGT=ultra
python run_glue.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --task_name ${TASK} --max_seq_length 128 --pad_to_max_length \
  --do_train --do_eval --load_best_model_at_end \
  --logging_dir runs/glue/${TASK} --logging_steps ${LOG_STEPS} \
  --output_dir ckpts/glue/${TASK} --save_steps ${EVAL_STEPS} \
  --metric_for_best_model accuracy --greater_is_better True \
  --evaluation_strategy steps --eval_steps ${EVAL_STEPS} --save_total_limit 2 \
  --per_device_train_batch_size 32 --per_device_eval_batch_size 64 \
  --num_train_epochs ${NE} --learning_rate ${LR} --warmup_ratio ${WR} \
  --n_drop ${ND} --just_cmp --loss_target ${TGT}
```

## RTE

```shell
export TASK=rte
```

- Baseline

```shell
export NE=7
export LR=1e-5
export WR=0.06
python run_glue.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --task_name ${TASK} --max_seq_length 128 --pad_to_max_length \
  --do_train --do_eval --load_best_model_at_end \
  --logging_dir runs/glue/${TASK} --logging_steps ${LOG_STEPS} \
  --output_dir ckpts/glue/${TASK} --save_steps ${EVAL_STEPS} \
  --metric_for_best_model accuracy --greater_is_better True \
  --evaluation_strategy steps --eval_steps ${EVAL_STEPS} --save_total_limit 2 \
  --per_device_train_batch_size 8 --per_device_eval_batch_size 8 \
  --num_train_epochs ${NE} --max_steps 2036 \
  --warmup_ratio ${WR} --warmup_steps 122 \
  --learning_rate ${LR} --weight_decay 0.01 --fp16
```

- Comparative Loss

```shell
export NE=7
export LR=1e-5
export WR=0.06
export ND=1
export TGT=first
python run_glue.py \
  --tokenizer_name ${PLM} --model_name_or_path ${PLM} \
  --task_name ${TASK} --max_seq_length 128 --pad_to_max_length \
  --do_train --do_eval --load_best_model_at_end \
  --logging_dir runs/glue/${TASK} --logging_steps ${LOG_STEPS} \
  --output_dir ckpts/glue/${TASK} --save_steps ${EVAL_STEPS} \
  --metric_for_best_model accuracy --greater_is_better True \
  --evaluation_strategy steps --eval_steps ${EVAL_STEPS} --save_total_limit 2 \
  --per_device_train_batch_size 8 --per_device_eval_batch_size 8 \
  --num_train_epochs ${NE} --max_steps 2036 \
  --warmup_ratio ${WR} --warmup_steps 122 \
  --learning_rate ${LR} --weight_decay 0.01 --fp16 \
  --n_drop ${ND} --just_cmp --loss_target ${TGT}
```