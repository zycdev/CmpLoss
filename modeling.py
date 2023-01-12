from dataclasses import dataclass
import json
import logging
import os
from typing import Any, Dict, Optional, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import AutoModel, LongformerModel, RobertaModel, PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
)

from arguments import CmpTrainingArguments, ModelArguments, MODEL_ARGS_NAME
from utils.ranking import list_mle, pairwise_hinge
from utils.tensor import mask_where0

logger = logging.getLogger(__name__)

HEAD_WEIGHTS_NAME = 'head.bin'


def get_objective(effects: torch.tensor, strategy: str = 'ultra') -> torch.tensor:
    with torch.no_grad():
        if strategy == 'first':
            first_effect = effects[:, 0:1].detach()
            best_effect = effects[:, 1:].max(dim=1, keepdim=True)[0].detach().clone()
            return torch.where(first_effect > best_effect, (first_effect + best_effect) / 2, best_effect)
        elif strategy == 'best':
            return effects.max(dim=1, keepdim=True)[0].detach().clone()
        else:
            return effects.new_zeros((effects.size(0), 1))


def get_primary(losses: torch.tensor, strategy: str = 'max') -> torch.tensor:
    if strategy == 'avg':
        return losses.mean(dim=1)
    elif strategy == 'fst':
        return losses[:, 0]
    elif strategy == 'lst':
        return losses[:, -1]
    else:
        return losses.max(dim=1)[0]


def get_rng_states():
    torch_rng_state = torch.get_rng_state()
    random_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state()
    return torch_rng_state, random_rng_state, cuda_rng_state


def set_rng_states(torch_rng_state, random_rng_state, cuda_rng_state):
    torch.set_rng_state(torch_rng_state)
    torch.random.set_rng_state(random_rng_state)
    torch.cuda.set_rng_state(cuda_rng_state)


@dataclass
class QAModelOutput(QuestionAnsweringModelOutput):
    pred_starts: Optional[torch.LongTensor] = None
    pred_ends: Optional[torch.LongTensor] = None


class CmpBase(nn.Module):

    def __init__(
            self,
            encoder: PreTrainedModel,
            model_args: ModelArguments,
            training_args: Optional[CmpTrainingArguments] = None
    ):
        super(CmpBase, self).__init__()
        self.args = model_args
        self.training_args = training_args
        self.encoder = encoder

        self.config = self.encoder.config
        self.dropout_names = []

    def _init_weights(self, module):
        # copied from transformers/models/bert/modeling_bert.py
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @classmethod
    def from_pretrained(
            cls,
            model_args: Optional[ModelArguments] = None,
            training_args: Optional[CmpTrainingArguments] = None,
            *args,
            **kwargs
    ):
        model_name_or_path = model_args.model_name_or_path
        encoder = AutoModel.from_pretrained(model_name_or_path, *args, **kwargs)

        model_args_path = os.path.join(model_name_or_path, MODEL_ARGS_NAME)
        if model_args is None and os.path.exists(model_args_path):
            with open(model_args_path) as f:
                model_args_dict = json.load(f)
            model_args = ModelArguments(**model_args_dict)

        model = cls(encoder, model_args, training_args)

        head_weights_path = os.path.join(model_name_or_path, HEAD_WEIGHTS_NAME)
        if os.path.exists(head_weights_path):
            logger.info(f"loading extra weights from {head_weights_path}")
            model_dict = torch.load(head_weights_path, map_location="cpu")
            model.load_state_dict(model_dict, strict=False)

        return model

    def save_pretrained(self, save_directory: Union[str, os.PathLike], *args, **kwargs):
        self.encoder.save_pretrained(save_directory, *args, **kwargs)

        model_dict = self.state_dict()
        encoder_parameter_keys = [k for k in model_dict.keys() if k.startswith('encoder')]
        for k in encoder_parameter_keys:
            model_dict.pop(k)
        if len(model_dict) > 0:
            torch.save(model_dict, os.path.join(save_directory, HEAD_WEIGHTS_NAME))

        with open(os.path.join(save_directory, MODEL_ARGS_NAME), 'w') as f:
            json.dump(self.args.to_dict(), f, ensure_ascii=False, indent=2)

    def floating_point_ops(self, input_dict: Dict[str, Union[torch.Tensor, Any]],
                           exclude_embeddings: bool = True) -> int:
        return self.encoder.floating_point_ops(input_dict, exclude_embeddings)

    def activate_dropout(self):
        for name in self.dropout_names:
            module = self.get_submodule(name)
            module.train()

    def deactivate_dropout(self):
        for name in self.dropout_names:
            module = self.get_submodule(name)
            module.eval()

    def power_dropout(self, power: int = 1):
        for name in self.dropout_names:
            module = self.get_submodule(name)
            module.__setattr__('orig_p', module.p)
            if power != 1:  # and not name.endswith('.self.dropout'):
                module.p = 1 - (1 - module.p) ** power
            module.__setattr__('orig_mode', module.training)
            module.train(True)

    def close_dropout(self):
        for name in self.dropout_names:
            module = self.get_submodule(name)
            module.__setattr__('orig_p', module.p)
            module.p = 0
            module.__setattr__('orig_mode', module.training)

    def restore_dropout(self):
        for name in self.dropout_names:
            module = self.get_submodule(name)
            module.p = module.orig_p
            module.__delattr__('orig_p')
            module.train(module.orig_mode)
            module.__delattr__('orig_mode')

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            global_attention_mask: Optional[torch.Tensor] = None,
            **kwargs
    ) -> BaseModelOutputWithPoolingAndCrossAttentions:
        if isinstance(self.encoder, RobertaModel):
            outputs = self.encoder(input_ids, attention_mask, **kwargs)
        elif isinstance(self.encoder, LongformerModel):
            outputs = self.encoder(input_ids, attention_mask, global_attention_mask, **kwargs)
        else:
            outputs = self.encoder(input_ids, attention_mask, token_type_ids, **kwargs)

        return outputs


class CmpQA(CmpBase):

    def __init__(
            self,
            encoder: PreTrainedModel,
            model_args: ModelArguments,
            training_args: Optional[CmpTrainingArguments] = None
    ):
        super(CmpQA, self).__init__(encoder, model_args, training_args)

        self.answerer = nn.Linear(self.config.hidden_size, 2)
        # self.qa_dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.span_mask = None

        self.dropout_names = [name for name, module in self.named_modules() if isinstance(module, nn.Dropout)]

        self._init_weights(self.answerer)

    def get_span_mask(self, seq_len: int, device) -> torch.Tensor:
        if self.span_mask is not None and seq_len <= self.span_mask.size(0):
            return self.span_mask[:seq_len, :seq_len].to(device)
        self.span_mask = torch.tril(
            torch.triu(torch.ones((seq_len, seq_len), device=device), 0), self.training_args.max_ans_len - 1
        )
        self.span_mask[:4, :] = 0
        self.span_mask[1, 1] = 1
        self.span_mask[2, 2] = 1
        # self.span_mask[3, 3] = 1
        return self.span_mask

    def _forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            global_attention_mask: Optional[torch.Tensor] = None,
            answer_mask: Optional[torch.Tensor] = None,
            **kwargs
    ):
        extra_padding_shape = None
        if attention_mask is not None:
            max_seq_len = attention_mask.sum(dim=1).max().item()
            if isinstance(self.encoder, LongformerModel) and max_seq_len % 512 != 0:
                max_seq_len = (max_seq_len // 512 + 1) * 512
            if max_seq_len < attention_mask.size(1):
                extra_padding_shape = attention_mask[:, max_seq_len:].shape
                attention_mask = attention_mask[:, :max_seq_len]
                if input_ids is not None:
                    input_ids = input_ids[:, :max_seq_len]
                if token_type_ids is not None:
                    token_type_ids = token_type_ids[:, :max_seq_len]
                if global_attention_mask is not None:
                    global_attention_mask = global_attention_mask[:, :max_seq_len]
                if answer_mask is not None:
                    answer_mask = answer_mask[:, :max_seq_len]

        encoder_outputs = super().forward(input_ids, attention_mask, token_type_ids, global_attention_mask,
                                          return_dict=True, **kwargs)
        # (B, L, H)
        seq_hidden_states = encoder_outputs.last_hidden_state
        # seq_hidden_states = self.qa_dropout(seq_hidden_states)
        # (B, L)
        start_logits, end_logits = self.answerer(seq_hidden_states).split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        if answer_mask is not None:
            start_logits = mask_where0(start_logits, answer_mask)
            end_logits = mask_where0(end_logits, answer_mask)

        if extra_padding_shape is not None:
            start_logits = torch.cat([start_logits, start_logits.new_full(extra_padding_shape, -1000)], dim=-1)
            end_logits = torch.cat([end_logits, end_logits.new_full(extra_padding_shape, -1000)], dim=-1)

        return start_logits, end_logits, encoder_outputs

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            global_attention_mask: Optional[torch.Tensor] = None,
            answer_mask: Optional[torch.Tensor] = None,
            start_positions: Optional[torch.Tensor] = None,
            end_positions: Optional[torch.Tensor] = None,
            **kwargs
    ) -> QuestionAnsweringModelOutput:
        n_drop, n_crop = self.training_args.n_drop, self.training_args.n_crop
        if n_crop == 0:
            if input_ids is not None and input_ids.ndim == 2:
                input_ids.unsqueeze_(1)
            if attention_mask is not None and attention_mask.ndim == 2:
                attention_mask.unsqueeze_(1)
            if token_type_ids is not None and token_type_ids.ndim == 2:
                token_type_ids.unsqueeze_(1)
            if global_attention_mask is not None and global_attention_mask.ndim == 2:
                global_attention_mask.unsqueeze_(1)
            if answer_mask is not None and answer_mask.ndim == 2:
                answer_mask.unsqueeze_(1)
        assert input_ids is None or input_ids.size(1) == 1 + n_crop

        if n_drop > 0 and self.training:
            self.close_dropout()
        elif self.eval() and self.training_args.force_dropout > 0:
            self.activate_dropout()
            self.power_dropout(self.training_args.force_dropout)
        rng_states = get_rng_states()
        start_logits, end_logits, encoder_outputs = self._forward(
            input_ids[:, 0], attention_mask[:, 0], token_type_ids[:, 0],
            global_attention_mask[:, 0] if global_attention_mask is not None else None,
            answer_mask[:, 0] if answer_mask is not None else None,
            **kwargs
        )
        if n_drop > 0 and self.training:
            self.restore_dropout()
        elif self.eval() and self.training_args.force_dropout > 0:
            self.restore_dropout()
            self.deactivate_dropout()

        bsz, seq_len = start_logits.shape
        # (B, L, L)
        span_scores = start_logits[:, :, None] + end_logits[:, None]
        span_scores = mask_where0(span_scores, self.get_span_mask(seq_len, start_logits.device).unsqueeze(0))
        # (B, K)
        top_spans = span_scores.view(bsz, -1).argsort(dim=-1,
                                                      descending=True)[:, :self.training_args.n_top_span].squeeze(-1)
        pred_starts = torch.div(top_spans, seq_len, rounding_mode='floor')  # top_spans // seq_len
        pred_ends = top_spans % seq_len

        loss = None
        if start_positions is not None and end_positions is not None:
            if n_crop == 0 and start_positions.ndim == 1:
                start_positions.unsqueeze_(1)
            if n_crop == 0 and end_positions.ndim == 1:
                end_positions.unsqueeze_(1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            start_positions[start_positions > seq_len] = -100
            end_positions[end_positions > seq_len] = -100

            # reading comprehension losses
            loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
            # (1 + CMP, B)
            all_start_losses = [loss_fct(start_logits, start_positions[:, 0])]
            all_end_losses = [loss_fct(end_logits, end_positions[:, 0])]
            n_cmp = n_drop + n_crop if self.training or self.training_args.cmp_in_eval else 0
            dc, cc = 0, 0
            for i in range(n_cmp):
                if cc < n_crop and (i % 2 == 0 or dc >= n_drop):  # crop first
                    cc += 1
                else:
                    dc += 1

                if n_drop > 0:
                    self.power_dropout(dc)
                set_rng_states(*rng_states)
                _start_logits, _end_logits, _ = self._forward(
                    input_ids[:, cc], attention_mask[:, cc], token_type_ids[:, cc],
                    global_attention_mask[:, cc] if global_attention_mask is not None else None,
                    answer_mask[:, cc] if answer_mask is not None else None,
                    **kwargs
                )
                if n_drop > 0:
                    self.restore_dropout()

                all_start_losses.append(loss_fct(_start_logits, start_positions[:, cc]))
                all_end_losses.append(loss_fct(_end_logits, end_positions[:, cc]))
            assert n_cmp == 0 or dc == n_drop and cc == n_crop
            # (B, 1 + CMP)
            all_start_losses = torch.stack(all_start_losses, dim=1)
            all_end_losses = torch.stack(all_end_losses, dim=1)
            all_rc_losses: torch.Tensor = (all_start_losses + all_end_losses) / 2

            loss = {
                f"rc_{i}": all_rc_losses[:, i]  # (B,)
                for i in range(1 + n_cmp)
            }
            loss['rc'] = get_primary(all_rc_losses, self.training_args.loss_primary)  # (B,)

            # comparative loss
            if n_cmp > 0:
                # (B, 1 + CMP) log likelihood
                start_effects = -all_start_losses / self.training_args.cr_temp  # .mean(dim=0, keepdim=True)
                end_effects = -all_end_losses / self.training_args.cr_temp  # .mean(dim=0, keepdim=True)
                effect_labels = torch.arange(
                    end_effects.size(-1), 0, -1, device=end_effects.device
                )[None, :].expand_as(end_effects)
                # (B, 1 + CMP + 1) log likelihood
                start_effects_ = torch.cat(
                    [start_effects, get_objective(start_effects, self.training_args.loss_target)],
                    dim=1
                )
                end_effects_ = torch.cat(
                    [end_effects, get_objective(end_effects, self.training_args.loss_target)],
                    dim=1
                )
                effect_labels_ = torch.arange(
                    end_effects_.size(-1), 0, -1, device=end_effects_.device
                )[None, :].expand_as(end_effects_)

                cmp_fct = list_mle if self.training_args.listwise else pairwise_hinge

                # (B,)
                loss['cmp'] = (cmp_fct(start_effects_, effect_labels_, reduction='none') +
                               cmp_fct(end_effects_, effect_labels_, reduction='none')) / 2
                loss['pair_reg'] = (pairwise_hinge(start_effects, effect_labels, reduction='none') +
                                    pairwise_hinge(end_effects, effect_labels, reduction='none')) / 2
                loss['list_reg'] = (list_mle(start_effects, effect_labels, reduction='none') +
                                    list_mle(end_effects, effect_labels, reduction='none')) / 2

                if self.training_args.just_cmp:
                    loss['overall'] = loss['cmp']
                else:
                    loss['overall'] = loss['rc']
                    if not (self.training_args.cr_schedule and loss['rc'].mean() > 1.1):
                        loss['overall'] += (loss['list_reg'] if self.training_args.listwise else
                                            loss['pair_reg']) * self.training_args.cr_weight
            else:
                loss['overall'] = loss['rc']

            # average in batch
            for k, v in loss.items():
                loss[k] = v.mean()

        return QAModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            pred_starts=pred_starts,
            pred_ends=pred_ends,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CmpCls(CmpBase):
    """
    Cmp Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """

    def __init__(
            self,
            encoder: PreTrainedModel,
            model_args: ModelArguments,
            training_args: Optional[CmpTrainingArguments] = None
    ):
        super(CmpCls, self).__init__(encoder, model_args, training_args)

        classifier_dropout = (
            self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

        self.dropout_names = [name for name, module in self.named_modules() if isinstance(module, nn.Dropout)]

        self._init_weights(self.classifier)

    def _forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            **kwargs
    ):
        encoder_outputs = super().forward(input_ids, attention_mask, token_type_ids, return_dict=True, **kwargs)
        # (B, H)
        pooled_output = encoder_outputs.pooler_output
        # (B, NL)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits, encoder_outputs

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            **kwargs
    ) -> SequenceClassifierOutput:
        if self.training_args.n_drop > 0 and self.training:
            self.close_dropout()
        rng_states = get_rng_states()
        logits, encoder_outputs = self._forward(input_ids, attention_mask, token_type_ids, **kwargs)
        if self.training_args.n_drop > 0 and self.training:
            self.restore_dropout()

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # (1 + C, B, NL)
            cmp_logits = [logits]
            for i in range(self.training_args.n_drop):
                self.power_dropout(1 + i)
                set_rng_states(*rng_states)
                _logits, _ = self._forward(input_ids, attention_mask, token_type_ids, **kwargs)
                self.restore_dropout()
                cmp_logits.append(_logits)

            # classification losses
            cls_losses: torch.Tensor = None  # (B, NL)
            if self.config.problem_type == "regression":
                loss_fct = MSELoss(reduction='none')
                if self.config.num_labels == 1:
                    cls_losses = torch.stack(
                        [loss_fct(_logits.squeeze(), labels.squeeze()) for _logits in cmp_logits],
                        dim=1
                    )
                else:
                    cls_losses = torch.stack(
                        [loss_fct(_logits, labels) for _logits in cmp_logits],
                        dim=1
                    )
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss(reduction='none')
                cls_losses = torch.stack(
                    [loss_fct(_logits.view(-1, self.config.num_labels), labels.view(-1)) for _logits in cmp_logits],
                    dim=1
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss(reduction='none')
                cls_losses = torch.stack(
                    [loss_fct(_logits, labels) for _logits in cmp_logits],
                    dim=1
                )

            loss = {
                f"cls_{i}": cls_losses[:, i]  # (B,)
                for i in range(1 + self.training_args.n_drop)
            }
            loss['cls'] = get_primary(cls_losses, self.training_args.loss_primary)  # (B,)

            # comparative loss
            if self.training_args.n_drop > 0:
                # (B, 1 + C) log likelihood
                cls_effects = -cls_losses / self.training_args.cr_temp  # .mean(dim=0, keepdim=True)
                effect_labels = torch.arange(
                    cls_effects.size(-1), 0, -1, device=cls_effects.device
                )[None, :].expand_as(cls_effects)
                # (B, 1 + C + 1) log likelihood
                cls_effects_ = torch.cat(
                    [cls_effects, get_objective(cls_effects, self.training_args.loss_target)],
                    dim=1
                )
                effect_labels_ = torch.arange(
                    cls_effects_.size(-1), 0, -1, device=cls_effects_.device
                )[None, :].expand_as(cls_effects_)

                cmp_fct = list_mle if self.training_args.listwise else pairwise_hinge

                # (B,)
                loss['cmp'] = cmp_fct(cls_effects_, effect_labels_, reduction='none')
                loss['pair_reg'] = pairwise_hinge(cls_effects, effect_labels, reduction='none')
                loss['list_reg'] = list_mle(cls_effects, effect_labels, reduction='none')

                if self.training_args.just_cmp:
                    loss['overall'] = loss['cmp']
                else:
                    loss['overall'] = loss['cls']
                    if not (self.training_args.cr_schedule and loss['cls'].mean() > 1.1):
                        loss['overall'] += (loss['list_reg'] if self.training_args.listwise else
                                            loss['pair_reg']) * self.training_args.cr_weight
            else:
                loss['overall'] = loss['cls']

            # average in batch
            for k, v in loss.items():
                loss[k] = v.mean()

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


def check_deterministic_dropout(use_cuda=True, dp=False):
    class MultiDropout(nn.Module):
        """Used to check the determinism of two runs of dropout"""

        def __init__(self):
            super(MultiDropout, self).__init__()
            self.dropout1 = nn.Dropout(p=0.2)
            self.dropout2 = nn.Dropout(p=0.2)
            self.dropout3 = nn.Dropout(p=0.2)

            self.dropout_names = [name for name, module in self.named_modules() if isinstance(module, nn.Dropout)]

        def power_dropout(self, power: int = 1):
            for name in self.dropout_names:
                module = self.get_submodule(name)
                module.__setattr__('orig_p', module.p)
                if power != 1:
                    module.p = 1 - (1 - module.p) ** power
                module.__setattr__('orig_mode', module.training)
                module.train(True)

        def close_dropout(self):
            for name in self.dropout_names:
                module = self.get_submodule(name)
                module.__setattr__('orig_p', module.p)
                module.p = 0
                module.__setattr__('orig_mode', module.training)

        def restore_dropout(self):
            for name in self.dropout_names:
                module = self.get_submodule(name)
                module.p = module.orig_p
                module.__delattr__('orig_p')
                module.train(module.orig_mode)
                module.__delattr__('orig_mode')

        def _forward(self, x: torch.Tensor) -> torch.Tensor:
            x1 = self.dropout1(x)
            # print(x1)
            x2 = self.dropout2(x1)
            # print(x2)
            x3 = self.dropout3(x2)
            # print(x3)

            return x3

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            rng_states = get_rng_states()
            self.close_dropout()
            x0 = self._forward(x)
            self.restore_dropout()

            set_rng_states(*rng_states)
            x1 = self._forward(x)
            assert len(set(tuple(idx) for idx in x1.nonzero().squeeze(-1).tolist()) -
                       set(tuple(idx) for idx in x0.nonzero().squeeze(-1).tolist())) == 0, f"{x0}\n{x1}"

            self.power_dropout(2)
            set_rng_states(*rng_states)
            x2 = self._forward(x)
            self.restore_dropout()
            assert len(set(tuple(idx) for idx in x2.nonzero().squeeze(-1).tolist()) -
                       set(tuple(idx) for idx in x1.nonzero().squeeze(-1).tolist())) == 0, f"{x1}\n{x2}"

            self.power_dropout(1)
            set_rng_states(*rng_states)
            x3 = self._forward(x)
            assert torch.all(x1 == x3), f"{x1}\n{x3}"

            return x1

    from tqdm.auto import tqdm

    model = MultiDropout()
    xx = torch.ones(16, 1000)
    if use_cuda:
        model = model.cuda()
        xx = xx.cuda()
    if dp and torch.cuda.device_count() > 1:
        print(torch.cuda.device_count())
        model = nn.DataParallel(model)

    for _ in tqdm(range(1000)):
        model(xx)

    print('OK')
