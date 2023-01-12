from dataclasses import asdict, dataclass, field
from enum import Enum
import logging
from typing import Optional

from transformers import TrainingArguments

logger = logging.getLogger(__name__)

DATA_ARGS_NAME = 'data_args.json'
MODEL_ARGS_NAME = 'model_args.json'
TRAINING_ARGS_NAME = 'training_args.json'

task2keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

task2criterion = {
    "cola": "matthews_correlation",
    "mnli": "accuracy",
    "mrpc": "accuracy",  # f1 accuracy
    "qnli": "accuracy",
    "qqp": "accuracy",  # f1 accuracy
    "rte": "accuracy",
    "sst2": "accuracy",
    "stsb": "pearson",  # pearson spearmanr
    "wnli": "accuracy",
}


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    tokenizer_name: str = field(
        metadata={"help": "The name of pretrained LM and tokenizer"}
    )
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )

    tag: Optional[str] = field(default=None)

    def __post_init__(self):
        cfg = [self.tokenizer_name]
        if self.tag:
            cfg.append(self.tag)
        self.abs = '_'.join(cfg)

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
        return d


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the task to train on: " + ", ".join(task2keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."},
    )
    corpus_file: Optional[str] = field(
        default=None, metadata={"help": "The common corpus file."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file."},
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "The input validation data file."},
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "The input testing data file."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. If False, will pad the samples dynamically when"
                " batching to the maximum length in the batch (which can be faster on GPU but will be slower on TPU)."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, some of the examples do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": (
                "The threshold used to select the null answer: if the best answer has a score that is less than "
                "the score of the null answer minus this threshold, the null answer is selected for this example. "
                "Only useful when `version_2_with_negative=True`."
            )
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": (
                "The maximum length of an answer that can be generated. This is needed because the start "
                "and end predictions are not conditioned on one another."
            )
        },
    )
    max_q_len: int = field(default=128)
    max_p_len: int = field(default=256)
    max_p_num: Optional[int] = field(default=None)

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task2keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task2keys.keys()))
        elif all(arg is None for arg in [self.dataset_name, self.train_file, self.validation_file, self.test_file]):
            raise ValueError("Need either a dataset name or a training/validation/test file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "tsv"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "tsv"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json", "tsv"], "`test_file` should be a csv or a json file."

        self.abs = f"l{self.max_seq_length}{'p' if self.pad_to_max_length else ''}"

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
        return d


@dataclass
class CmpTrainingArguments(TrainingArguments):
    logging_dir: Optional[str] = field(default='runs', metadata={"help": "Tensorboard log dir."})
    logging_nan_inf_filter: bool = field(default=False, metadata={"help": "Filter nan and inf losses for logging."})

    n_drop: int = field(default=0, metadata={"help": "The total times of CmpDrop for each example."})
    n_crop: int = field(default=0, metadata={"help": "The total times of CmpCrop for each example."})
    listwise: bool = field(default=False)
    just_cmp: bool = field(default=False)
    loss_target: Optional[str] = field(default='ultra', metadata={"help": "The target loss for all model variants"})
    loss_primary: Optional[str] = field(default='max', metadata={"help": "The primary loss of all model variants"})
    cr_temp: float = field(default=1.0, metadata={"help": "The temperature in comparative loss."})
    cr_weight: float = field(default=0.0, metadata={"help": "The weight of comparative regularization."})
    cr_schedule: bool = field(default=False)

    max_ans_len: int = field(default=30)
    n_top_span: int = field(default=30)

    disable_dropout: bool = field(default=False)
    force_dropout: int = field(default=0)
    cmp_in_eval: bool = field(default=True)

    comment: Optional[str] = field(default=None)

    pdb: bool = field(default=False)

    def __post_init__(self):
        super().__post_init__()
        self.n_drop = max(0, self.n_drop)
        self.n_crop = max(0, self.n_crop)
        self.just_cmp = self.just_cmp and self.n_drop + self.n_crop > 0
        if self.loss_target not in {'ultra', 'first', 'best'}:
            self.loss_target = 'ultra'
        if self.loss_primary not in {'max', 'avg', 'fst', 'lst'}:
            self.loss_primary = 'max'
        self.cr_weight = max(0., self.cr_weight) if self.n_drop + self.n_crop > 0 else 0.
        if self.cr_weight == 0:
            self.cr_schedule = False
        self.force_dropout = max(0, self.force_dropout)

    @property
    def abs(self) -> str:
        cfg = [
            f"b{self.train_batch_size}{f'x{self.gradient_accumulation_steps}' if self.gradient_accumulation_steps > 1 else ''}",
            f"e{self.num_train_epochs:.0f}",
            f"lr{self.learning_rate}" + (f"w{self.warmup_ratio}" if self.warmup_ratio > 0 else '')
        ]
        if self.n_drop + self.n_crop > 0:
            cmp_pattern = f"{'l' if self.listwise else 'p'}"
            if self.n_drop > 0:
                cmp_pattern += f"{self.n_drop}"
            if self.n_crop > 0:
                cmp_pattern += f"+{self.n_crop}"
            cfg.append(cmp_pattern)
            if self.cr_temp != 1:
                cfg.append(f"t{self.cr_temp}")
            if self.just_cmp:
                cfg.append(self.loss_target)
            else:
                cfg.append(f"{self.loss_primary}{self.cr_weight}{'s' if self.cr_schedule else ''}")
        if self.disable_dropout:
            cfg.append("ddo")
        if self.fp16:
            cfg.append("fp16")
        if self.comment:
            cfg.append(self.comment)
        return '_'.join(cfg)

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = super().to_dict()
        d = {**d, **{"train_batch_size": int(self.train_batch_size), "eval_batch_size": int(self.eval_batch_size)}}
        return d
