from collections import OrderedDict
from dataclasses import dataclass
import json
import logging
import os
import random
from tqdm.auto import tqdm
from typing import Any, List, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import LongformerTokenizerFast, PreTrainedTokenizerFast

from utils.tensor import pad_tensors

logger = logging.getLogger(__name__)


def load_corpus(corpus_path: str, tokenizer: PreTrainedTokenizerFast = None, recache: bool = False) -> Dict:
    corpus = dict()
    logger.info(f"Loading corpus from {corpus_path} ...")
    with open(corpus_path) as f:
        col_names = None
        for line in f:
            segs = [field.strip() for field in line.strip().split('\t')]
            if not col_names:
                col_names = segs
                continue
            if len(segs) != len(col_names) or segs[0] == 'id':
                logger.warning(f'Wrong line format: {segs[0]}')
                continue
            fields = {k: v for k, v in zip(col_names, segs)}

            p_id, text = fields['id'], fields['text']
            title = fields['title'] if 'title' in fields else ''
            if text == '' and title == '':
                logger.warning(f"empty passage: {p_id}")
                continue
            sentence_spans = [tuple(span) for span in
                              eval(fields['sentence_spans'])] if 'sentence_spans' in fields else [(0, len(text))]

            if title in corpus:
                logger.warning(f"Duplicate passage: {p_id} ({title})")
            corpus[title] = {
                "id": p_id,
                "text": text,
                "sentence_spans": sentence_spans
            }
    logger.info(f"{len(corpus):,d} passages Loaded")
    if not tokenizer:
        return corpus

    corpus_dir, corpus_file = os.path.split(corpus_path)
    cache_dir = os.path.join(corpus_dir, '.cache')
    cache_file = f"{corpus_file.rsplit('.', 1)[0]}.{tokenizer.name_or_path.replace('/', '_')}.tsv"
    cache_path = os.path.join(cache_dir, cache_file)
    if recache or not os.path.exists(cache_path):
        logger.info(f"Tokenizing and caching {corpus_path} into {cache_path}")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        with open(cache_path, 'w') as f:
            for title, psg in tqdm(corpus.items(), total=len(corpus)):
                title_codes = tokenizer(title, add_special_tokens=False,
                                        return_attention_mask=False, return_offsets_mapping=True)
                text_codes = tokenizer(psg['text'], add_special_tokens=False,
                                       return_attention_mask=False, return_offsets_mapping=True)
                f.write(f"{title}\t{json.dumps(dict(title_codes))}\t{json.dumps(dict(text_codes))}\n")
    with open(cache_path) as f:
        for line in f:
            title, title_codes, text_codes = [field.strip() for field in line.strip().split('\t')]
            corpus[title]['title_codes'] = json.loads(title_codes)
            corpus[title]['text_codes'] = json.loads(text_codes)
    logger.info(f"Tokenized cache is loaded from {cache_path}")

    return corpus


def construct_char2token(token2spans, char_num):
    char2token = [-1] * char_num
    for tok_idx, (char_start, char_end) in enumerate(token2spans):
        for char_idx in range(char_start, char_end):
            char2token[char_idx] = tok_idx
    return char2token


def char2token_span(index_map: List[int], span: Tuple[int, int]) -> Tuple[int, int]:
    s, e = span  # [s, e]
    assert s <= e, f"[{s}, {e}]"
    if not 0 <= s <= e < len(index_map):
        return -1, -1

    while index_map[s] < 0 and s + 1 <= e:
        s += 1
    ns = index_map[s]
    if ns < 0:
        return -1, -1

    while index_map[e] < 0 and e - 1 >= s:
        e -= 1
    ne = index_map[e]  # include
    assert ns <= ne
    return ns, ne


def prepare_squad_features(examples: Dict[str, List], tokenizer: PreTrainedTokenizerFast,
                           max_seq_len: int = 512, doc_stride: int = 128, pad_to_max_length: bool = False):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples['question'] = [q.lstrip() for q in examples['question']]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == 'right'
    context_idx = 1 if pad_on_right else 0

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples['question' if pad_on_right else 'context'],
        examples['context' if pad_on_right else 'question'],
        truncation='only_second' if pad_on_right else 'only_first',
        max_length=max_seq_len,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding='max_length' if pad_to_max_length else False,
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop('overflow_to_sample_mapping')
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples['offset_mapping']

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples['example_id'] = []
    if 'answers' in examples:
        # Let's label those examples!
        tokenized_examples['start_positions'] = []
        tokenized_examples['end_positions'] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples['input_ids'][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples['example_id'].append(examples['id'][sample_index])

        if 'answers' in examples:
            answers = examples['answers'][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers['answer_start']) == 0:
                tokenized_examples['start_positions'].append(cls_index)
                tokenized_examples['end_positions'].append(cls_index)
            else:
                # XXX: only the first answer is considered
                # Start/end character index of the answer in the text.
                start_char = answers['answer_start'][0]
                end_char = start_char + len(answers['text'][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != context_idx:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != context_idx:
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples['start_positions'].append(cls_index)
                    tokenized_examples['end_positions'].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples['start_positions'].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples['end_positions'].append(token_end_index + 1)

        # Set to None the offset_mapping that are not part of the context,
        # so it's easy to determine if a token position is part of the context or not.
        offset_mapping[i] = [o if sequence_ids[k] == context_idx else None for k, o in enumerate(offset_mapping[i])]

    return tokenized_examples


@dataclass
class MultiDocQACollator:
    pad_token_id: Optional[int] = 0
    pad_to_multiple_of: Optional[int] = 1

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        bsz = len(samples)
        if bsz == 0:
            return dict()

        nn_keys = ["input_ids", "attention_mask", "token_type_ids", "global_attention_mask", "answer_mask"]
        if "start_positions" in samples[0]:
            nn_keys.extend(["start_positions", "end_positions"])
        if torch.is_tensor(samples[0]['input_ids']):
            n_view = 1
            nn_input: Dict[str, List[torch.Tensor]] = {
                k: [sample[k] for sample in samples] for k in nn_keys
            }
        else:
            n_view = len(samples[0]['input_ids'])
            nn_input: Dict[str, List[torch.Tensor]] = {
                k: [sample[k][i] for sample in samples for i in range(n_view)] for k in nn_keys
            }

        batch: Dict[str, torch.Tensor] = dict()
        for k, v in nn_input.items():
            if k in ["start_positions", "end_positions"]:
                batch[k] = torch.stack(v).view(bsz, n_view)
            elif k == 'input_ids':
                batch[k] = pad_tensors(v, self.pad_token_id, self.pad_to_multiple_of).view(bsz, n_view, -1)
            else:
                batch[k] = pad_tensors(v, 0, self.pad_to_multiple_of).view(bsz, n_view, -1)
            batch[k].squeeze_(1)

        batch.update({
            k: [sample[k][0] if n_view > 1 else sample[k] for sample in samples] for k in samples[0] if k not in batch
        })
        return batch


class MultiDocQADataset(Dataset):

    def __init__(self, data_path: str, tokenizer: LongformerTokenizerFast, corpus: Dict,
                 max_seq_len: int = 4096, max_q_len: int = 128, max_p_len: int = 256, max_p_num: int = None,
                 mode: str = 'test', crop_times: int = 0, recache: bool = False):
        self.tokenizer = tokenizer
        self.corpus = corpus
        self.max_seq_len = max_seq_len
        self.max_q_len = max_q_len
        self.max_p_len = max_p_len
        self.max_p_num = max_p_num
        self.mode = mode
        self.crop_times = crop_times

        self.q_ids = []
        self.examples = dict()
        with open(data_path) as f:
            for line in f:
                q_id, question, context, answers, sp_facts = [field.strip() for field in line.strip().split('\t')]
                if q_id == 'id':
                    continue
                context, answers, sp_facts = json.loads(context), json.loads(answers), json.loads(sp_facts)
                answers['positions'] = [tuple(pos) for pos in answers['positions']]
                example = {
                    "id": q_id,
                    "question": question,
                    "context": {title: corpus[title] for title in context},
                    "answers": answers,
                    "sp_facts": sp_facts
                }
                if max_p_num is not None:
                    for title in list(example['context'].keys()):
                        if len(example['context']) <= max_p_num:
                            break
                        if title not in example['sp_facts']:
                            example['context'].pop(title)
                self.q_ids.append(q_id)
                self.examples[q_id] = example

        data_dir, data_file = os.path.split(data_path)
        cache_dir = os.path.join(data_dir, '.cache')
        cache_file = f"{data_file.rsplit('.', 1)[0]}.{self.tokenizer.name_or_path.replace('/', '_')}.tsv"
        cache_path = os.path.join(cache_dir, cache_file)
        if recache or not os.path.exists(cache_path):
            logger.info(f"Tokenizing and caching questions of {data_path} into {cache_path}")
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            with open(cache_path, 'w') as f:
                for q_id, example in tqdm(self.examples.items(), total=len(self.examples)):
                    ques_codes = self.tokenizer(example['question'], add_special_tokens=False,
                                                return_attention_mask=False, return_offsets_mapping=True)
                    f.write(f"{q_id}\t{json.dumps(dict(ques_codes))}\n")
        with open(cache_path) as f:
            for line in f:
                q_id, ques_codes = [field.strip() for field in line.strip().split('\t')]
                self.examples[q_id]['ques_codes'] = json.loads(ques_codes)

    def __len__(self):
        return len(self.examples)

    def _construct_feature(self, example, doc_seq: List[str],
                           ans_pos: Tuple[str, int, int, int] = None) -> Dict[str, Any]:
        yes = 'ĠYES'
        no = 'ĠNO'
        soq = '????????'
        sod = 'madeupword0000'
        sop = 'madeupword0001'
        # sos = 'madeupword0002'
        yes_id = self.tokenizer.convert_tokens_to_ids(yes)
        no_id = self.tokenizer.convert_tokens_to_ids(no)
        soq_id = self.tokenizer.convert_tokens_to_ids(soq)
        sod_id = self.tokenizer.convert_tokens_to_ids(sod)
        sop_id = self.tokenizer.convert_tokens_to_ids(sop)
        # sos_id = self.tokenizer.convert_tokens_to_ids(sos)

        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id

        if not ans_pos:
            answer_start, answer_end = None, None
        elif ans_pos[:2] == ("012Q", -1):
            answer_start, answer_end = 1 + ans_pos[2], 1 + ans_pos[3]
        else:
            answer_start, answer_end = -100, -100

        '''
        <s> [YES] [NO] [Q] q </s> [T] t1 [P] p1 [T] t2 [P] p2 </s>
        1 + 3 + |Q| + 1 + np * (2 + |P|) + 1
        '''
        input_ids = [cls_id, yes_id, no_id, soq_id]
        token_type_ids = [0] * len(input_ids)
        global_attention_mask = [1] * len(input_ids)
        answer_mask = [0] + [1] * (len(input_ids) - 1)

        # concatenate question
        question_codes = example['ques_codes']
        input_ids += question_codes['input_ids'][:self.max_q_len]
        global_attention_mask += [1] * (len(input_ids) - len(global_attention_mask))
        input_ids.append(sep_id)
        global_attention_mask.append(0)
        token_type_ids += [0] * (len(input_ids) - len(token_type_ids))
        answer_mask += [0] * (len(input_ids) - len(answer_mask))
        assert len(input_ids) == len(token_type_ids) == len(global_attention_mask) == len(answer_mask)

        range2para: Dict[Tuple[int, int], Tuple[str, int]] = OrderedDict()  # closed interval -> paragraph position
        for title in doc_seq:
            if len(input_ids) >= self.max_seq_len:
                break

            doc = example['context'][title]
            if self.mode == 'train' and ans_pos[:2] == (title, 1):
                max_p_len = max(self.max_p_len, ans_pos[3] + 12)  # should at least include the answer
            else:
                max_p_len = self.max_p_len

            # concatenate title
            input_ids.append(sod_id)
            token_type_ids.append(1)
            global_attention_mask.append(1)
            answer_mask.append(0)
            title_offset = len(input_ids)
            input_ids += doc['title_codes']['input_ids']
            range2para[(title_offset, len(input_ids) - 1)] = (title, 0)
            token_type_ids += [1] * (len(input_ids) - len(token_type_ids))
            global_attention_mask += [0] * (len(input_ids) - len(global_attention_mask))
            answer_mask += [1] * (len(input_ids) - len(answer_mask))
            # label answer span in title tokens
            if ans_pos and ans_pos[:2] == (title, 0):
                char2token = construct_char2token(doc['title_codes']['offset_mapping'], len(title))
                start_char, end_char = ans_pos[2:]
                start_token, end_token = char2token_span(char2token, (start_char, end_char - 1))
                if start_token >= 0:
                    answer_start = title_offset + start_token
                    answer_end = title_offset + end_token

            # concatenate text
            input_ids.append(sop_id)
            token_type_ids.append(1)
            global_attention_mask.append(1)
            answer_mask.append(0)
            text_offset = len(input_ids)
            input_ids += doc['text_codes']['input_ids'][:max_p_len]
            range2para[(text_offset, len(input_ids) - 1)] = (title, 1)
            token_type_ids += [1] * (len(input_ids) - len(token_type_ids))
            global_attention_mask += [0] * (len(input_ids) - len(global_attention_mask))
            answer_mask += [1] * (len(input_ids) - len(answer_mask))
            # label answer span in text tokens
            if ans_pos and ans_pos[:2] == (title, 1):
                char2token = construct_char2token(doc['text_codes']['offset_mapping'], len(doc['text']))
                start_char, end_char = ans_pos[2:]
                start_token, end_token = char2token_span(char2token, (start_char, end_char - 1))
                if start_token >= 0:
                    answer_start = text_offset + start_token
                    answer_end = text_offset + end_token

            assert len(input_ids) == len(token_type_ids) == len(global_attention_mask) == len(answer_mask)

        if len(input_ids) >= self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len - 1]
            token_type_ids = token_type_ids[:len(input_ids)]
            global_attention_mask = global_attention_mask[:len(input_ids)]
            answer_mask = answer_mask[:len(input_ids)]
            if answer_end is not None and answer_end >= self.max_seq_len - 1:
                answer_start, answer_end = -100, -100
        input_ids.append(sep_id)
        token_type_ids.append(1)
        global_attention_mask.append(0)
        answer_mask.append(0)

        feature = {
            "q_id": example['id'],
            "range2para": range2para,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor([1] * len(input_ids), dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "global_attention_mask": torch.tensor(global_attention_mask, dtype=torch.long),
            "answer_mask": torch.tensor(answer_mask, dtype=torch.float),
        }

        if self.mode == 'test' or None in (answer_start, answer_end):
            return feature

        feature.update({
            "start_positions": torch.tensor(answer_start, dtype=torch.long),
            "end_positions": torch.tensor(answer_end, dtype=torch.long),
        })
        return feature

    def __getitem__(self, index) -> Dict[str, Any]:
        q_id = self.q_ids[index]
        example = self.examples[q_id]

        titles = list(example['context'].keys())
        if self.mode == 'test':
            return self._construct_feature(example, titles)

        if self.mode == 'train':
            random.shuffle(titles)
            ans_pos = random.choice(example['answers']['positions'])
        else:
            ans_pos = example['answers']['positions'][0]
        feature = self._construct_feature(example, titles, ans_pos)
        if self.mode != 'train':  # cache token offsets for prediction
            example['range2para'] = feature['range2para']
        if self.crop_times == 0:
            return feature

        features = [feature]
        doc2rank = {d: i for i, d in enumerate(titles)}
        support_docs = list(example['sp_facts'].keys())
        distractor_docs = [d for d in titles if d not in example['sp_facts']]
        if self.mode == 'train':
            distractor_docs = np.random.permutation(distractor_docs).tolist()
            if len(distractor_docs) > self.crop_times:
                distractor_nums = sorted(
                    np.random.choice(range(len(distractor_docs)), size=self.crop_times, replace=False), reverse=True
                )
            else:
                distractor_nums = list(range(len(distractor_docs) - 1, -1, -1))
                distractor_nums += [0] * (self.crop_times - len(distractor_nums))
        else:
            distractor_nums = [6, 4, 2, 0][:self.crop_times]
        for nd in distractor_nums:
            doc_seq = sorted(support_docs + distractor_docs[:nd], key=lambda d: doc2rank[d])
            features.append(self._construct_feature(example, doc_seq, ans_pos))
        return {k: [f[k] for f in features] for k in feature}
