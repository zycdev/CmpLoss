# -*- coding: UTF-8 -*-

# from html import unescape
import logging
# import re
from typing import List, Tuple
import unicodedata
# from urllib.parse import unquote

from Levenshtein import distance, ratio

from utils.basic_tokenizer import Tokenizer, SimpleTokenizer

logger = logging.getLogger(__name__)


def atof(s):
    try:
        return float(s)
    except ValueError:
        pass

    try:
        import locale
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
        return locale.atof(s)
    except ValueError:
        pass

    return None


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import locale
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
        locale.atof(s)
        return True
    except ValueError:
        pass

    try:
        for c in s:
            unicodedata.numeric(c)
        return True
    except (TypeError, ValueError):
        pass
    return False


def is_startswith(a: str, b: str) -> bool:
    short, long = sorted([a, b], key=lambda x: len(x))
    return long.startswith(short)


def is_endswith(a: str, b: str) -> bool:
    short, long = sorted([a, b], key=lambda x: len(x))
    return long.endswith(short)


def start_or_end_with(a: str, b: str) -> bool:
    return is_startswith(a, b) or is_endswith(a, b)


def is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char in [" ", "\t", "\n", "\r"]:
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace characters.
    if char in ["\t", "\n", "\r"]:
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    if char in ["～", "￥", "×"]:
        return True
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def is_chinese_char(char):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    cp = ord(char)
    if (0x4E00 <= cp <= 0x9FFF or
            0x3400 <= cp <= 0x4DBF or
            0x20000 <= cp <= 0x2A6DF or
            0x2A700 <= cp <= 0x2B73F or
            0x2B740 <= cp <= 0x2B81F or
            0x2B820 <= cp <= 0x2CEAF or
            0xF900 <= cp <= 0xFAFF or
            0x2F800 <= cp <= 0x2FA1F):
        return True

    return False


def is_word_boundary(char):
    return is_whitespace(char) or is_punctuation(char) or is_chinese_char(char)


def clean_text(text):
    # unescaped_text = unescape(text)
    # unquoted_text = unquote(unescaped_text, 'utf-8')
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or is_control(char):
            continue
        if is_whitespace(char):
            output.append(" ")
        # elif char in ["–"]:
        #     output.append("-")
        else:
            output.append(char)
    output_text = ''.join(output)
    # output_text = re.sub(r' {2,}', ' ', output_text).strip()
    return output_text


def norm_text(s):
    return ' '.join(clean_text(s).strip().split())


def normalize_unicode(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


def fuzzy_find_all(
        source: str, patterns: List[str], tokenizer: Tokenizer = None, ignore_case: bool = False,
        max_l_dist: int = 0, min_ratio: float = 1.0, granularity: str = 'word'
) -> Tuple[List[Tuple[int, int]], List[str]]:
    assert granularity in {'word', 'char'}
    if tokenizer is None:
        tokenizer = SimpleTokenizer()
    sep = '' if granularity == 'char' else ' '
    src_tokens = tokenizer.tokenize(source)
    src_words = src_tokens.words(uncased=ignore_case)
    src_offsets = src_tokens.offsets()
    _src_words = [normalize_unicode(w) for w in src_words]
    spans, matches = [], []
    for pattern in patterns:
        pat_tokens = tokenizer.tokenize(pattern)
        pat_words = pat_tokens.words(uncased=ignore_case)
        _pat_words = [normalize_unicode(w) for w in pat_words]
        if len(_pat_words) == 0:
            continue
        _pattern = sep.join(_pat_words)
        for s in range(0, len(_src_words) - len(_pat_words) + 1):
            e = min(s + len(_pat_words), len(_src_words))
            while e + 1 <= len(_src_words) and _pattern.startswith(sep.join(_src_words[s:e + 1])):
                e += 1
            _candidate = sep.join(_src_words[s:e])
            if (
                    start_or_end_with(_pattern, _candidate) and
                    distance(_pattern, _candidate) <= min(max_l_dist, len(_pattern), len(_candidate)) and
                    ratio(_pattern, _candidate) >= min_ratio
            ):
                match = src_tokens.slice(s, e).untokenize()
                span = (src_offsets[s][0], src_offsets[e - 1][1])
                assert source[span[0]:span[1]] == match
                spans.append(span)
                matches.append(match)
    return spans, matches


def fuzzy_find_spans(source: str, targets: List[str], max_dis: int = 4) -> Tuple[List[Tuple[int, int]], int]:
    spans, _ = fuzzy_find_all(source, targets, ignore_case=False)
    if spans or 1 > max_dis:
        return spans, 0

    spans, _ = fuzzy_find_all(source, targets, ignore_case=True)
    if spans or 2 > max_dis:
        return spans, 1

    spans, matches = fuzzy_find_all(source, targets, ignore_case=True,
                                    max_l_dist=3, min_ratio=0.75)
    spans = [(start, end) for (start, end), match in zip(spans, matches)
             if not is_number(match) or atof(match) in [atof(ans) for ans in targets]]
    if spans or 3 > max_dis:
        return spans, 2

    spans, matches = fuzzy_find_all(source, targets, ignore_case=True,
                                    max_l_dist=3, min_ratio=0.75, granularity='char')
    spans = [(start, end) for (start, end), match in zip(spans, matches)
             if not is_number(match) or atof(match) in [atof(ans) for ans in targets]]
    if spans or 4 > max_dis:
        return spans, 3

    return [], 4
