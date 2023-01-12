#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for question answering using a slightly adapted version of the 🤗 Trainer.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import json
import logging
import os
import sys

import datasets
import transformers
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from arguments import CmpTrainingArguments, DataArguments, ModelArguments, DATA_ARGS_NAME
from modeling import CmpQA
from trainer import QATrainer
from utils.qa import postprocess_squad_predictions
from utils.data import prepare_squad_features

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.19.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

logger = logging.getLogger(__name__)


# noinspection PyArgumentList
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataArguments, CmpTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: CmpTrainingArguments

    # Update some training arguments
    if training_args.do_train:
        if 'large' in model_args.model_name_or_path:
            # For large models, try not to optimize the full model directly to avoid overfitting
            training_args.loss_target = 'first'
        training_args.run_name = f"{model_args.abs}#{data_args.abs}#{training_args.abs}"
        training_args.output_dir = os.path.join(training_args.output_dir, training_args.run_name)
        training_args.logging_dir = os.path.join(training_args.logging_dir, training_args.run_name)

    # Setup logging
    log_level = training_args.get_process_log_level()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level,
    )
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Data parameters {data_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(set(os.listdir(training_args.output_dir)) - {DATA_ARGS_NAME}) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = datasets.load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        extension = None
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = datasets.load_dataset(
            extension,
            data_files=data_files,
            field="data",
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if training_args.disable_dropout:
        config.attention_probs_dropout_prob = 0.
        config.hidden_dropout_prob = 0.
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if training_args.n_crop + training_args.n_drop > 0:
        model = CmpQA.from_pretrained(
            model_args,
            training_args,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models at"
            " https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet"
            " this requirement"
        )

    # Preprocessing the datasets.
    # Preprocessing is slightly different for training and evaluation.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    else:
        column_names = raw_datasets["test"].column_names

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Training preprocessing
    train_dataset = None
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            # Select samples from Dataset. This will help to decrease processing time
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        # Create training features
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                function=prepare_squad_features,
                fn_kwargs={"tokenizer": tokenizer, "max_seq_len": max_seq_length,
                           "doc_stride": data_args.doc_stride, "pad_to_max_length": data_args.pad_to_max_length},
                batched=True,
                remove_columns=column_names,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                new_fingerprint=(f"cr-{tokenizer.name_or_path.replace('/', '_')}-{data_args.doc_stride}-"
                                 f"{max_seq_length}{'padded' if data_args.pad_to_max_length else ''}-train"),
                desc="Running tokenizer on train dataset",
            )
        if data_args.max_train_samples is not None:
            # Select samples from dataset again since Feature Creation might increase number of features
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    # Validation preprocessing
    eval_examples, eval_dataset = None, None
    if training_args.do_eval or training_args.do_train:
        if "validation" not in raw_datasets:
            raise ValueError("Evaluation requires a validation dataset")
        eval_examples = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            # Select eval samples from dataset
            max_eval_samples = min(len(eval_examples), data_args.max_eval_samples)
            eval_examples = eval_examples.select(range(max_eval_samples))
        # Create validation features
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_examples.map(
                function=prepare_squad_features,
                fn_kwargs={"tokenizer": tokenizer, "max_seq_len": max_seq_length,
                           "doc_stride": data_args.doc_stride, "pad_to_max_length": data_args.pad_to_max_length},
                batched=True,
                remove_columns=column_names,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                new_fingerprint=(f"cr-{tokenizer.name_or_path.replace('/', '_')}-{data_args.doc_stride}-"
                                 f"{max_seq_length}{'padded' if data_args.pad_to_max_length else ''}-val"),
                desc="Running tokenizer on validation dataset",
            )
        if data_args.max_eval_samples is not None:
            # Select Samples from Dataset again since Feature Creation might increase samples size
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    # Prediction preprocessing
    predict_examples, predict_dataset = None, None
    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            # We will select sample from whole data
            predict_examples = predict_examples.select(range(data_args.max_predict_samples))
        # Create prediction features
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = eval_examples.map(
                function=prepare_squad_features,
                fn_kwargs={"tokenizer": tokenizer, "max_seq_len": max_seq_length,
                           "doc_stride": data_args.doc_stride, "pad_to_max_length": data_args.pad_to_max_length},
                batched=True,
                remove_columns=column_names,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                new_fingerprint=(f"cr-{tokenizer.name_or_path.replace('/', '_')}-{data_args.doc_stride}-"
                                 f"{max_seq_length}{'padded' if data_args.pad_to_max_length else ''}-test"),
                desc="Running tokenizer on prediction dataset",
            )
        if data_args.max_predict_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Data collator
    # We have already padded to max length if the corresponding flag is True,
    # otherwise we need to pad in the data collator.
    data_collator = (
        default_data_collator if data_args.pad_to_max_length
        else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    )

    # Post-processing:
    def post_processing_function(examples, features, predictions, output_dir=None, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_squad_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=data_args.version_2_with_negative,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            null_score_diff_threshold=data_args.null_score_diff_threshold,
            output_dir=output_dir,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        if data_args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
                for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric = datasets.load_metric("squad_v2" if data_args.version_2_with_negative else "squad")

    # Initialize our Trainer
    trainer = QATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_examples=eval_examples,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=lambda p: metric.compute(predictions=p.predictions, references=p.label_ids),
    )

    # Training
    if training_args.do_train:
        with open(os.path.join(training_args.output_dir, DATA_ARGS_NAME), 'w') as f:
            json.dump(data_args.to_dict(), f, ensure_ascii=False, indent=2)

        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = len(train_dataset) if data_args.max_train_samples is None else data_args.max_train_samples
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        max_eval_samples = len(eval_dataset) if data_args.max_eval_samples is None else data_args.max_eval_samples
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        results = trainer.predict(predict_dataset, predict_examples)
        metrics = results.metrics

        max_predict_samples = (
            len(predict_dataset) if data_args.max_predict_samples is None else data_args.max_predict_samples
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "question-answering"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
