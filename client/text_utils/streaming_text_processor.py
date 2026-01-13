import re
import string
from typing import List, Optional, Tuple

from .numbers_to_text import NumberConverter, strip_punctuation, restore_punctuation


_REMOVAL_INFLECTIONS = ["-ს", "-ად"]
_ADDITION_INFLECTIONS = ["-მა", "-ო"]
_KEEP_INFLECTIONS = ["-ის", "-ით"]
_ALL_INFLECTIONS = _REMOVAL_INFLECTIONS + _ADDITION_INFLECTIONS + _KEEP_INFLECTIONS

_DASH_ONLY_RE = re.compile(r"^[-–—]+$")
_NUM_TOKEN_RE = re.compile(r"^(-?\d+(?:[.,]\d+)?)(%?)(.*)$")

_PUNCT_STRIP_CHARS = set(string.punctuation) - {"%", "-"}
_PUNCT_STRIP_CHARS.update({"“", "”", "‘", "’", "…"})


def _split_punctuation(word: str) -> Tuple[str, str, str]:
    start = 0
    end = len(word)

    while start < end and word[start] in _PUNCT_STRIP_CHARS:
        start += 1

    while end > start and word[end - 1] in _PUNCT_STRIP_CHARS:
        end -= 1

    return word[:start], word[start:end], word[end:]


class StreamingTextProcessor:
    def __init__(
        self,
        num_converter: Optional[NumberConverter] = None,
        comma_multiplier: int = 3,
        dash_to_comma: bool = True,
        remove_asterisks: bool = True,
    ) -> None:
        self.num_converter = num_converter or NumberConverter()
        self.comma_multiplier = max(1, comma_multiplier)
        self.dash_to_comma = dash_to_comma
        self.remove_asterisks = remove_asterisks
        self._pending_word: Optional[str] = None

    def push_word(self, word: str) -> List[str]:
        cleaned = self._clean_word(word)
        if not cleaned:
            return []

        expanded = self._expand_word(cleaned)
        emitted: List[str] = []
        for item in expanded:
            emitted.extend(self._process_inflections(item))
        return emitted

    def flush(self) -> List[str]:
        if self._pending_word is None:
            return []
        word = self._pending_word
        self._pending_word = None
        return [self._apply_punctuation_multiplier(word)]

    def _clean_word(self, word: str) -> str:
        if self.remove_asterisks and "*" in word:
            word = word.replace("*", "")
        if not word:
            return ""
        if self.dash_to_comma and _DASH_ONLY_RE.match(word):
            return ","
        return word

    def _expand_word(self, word: str) -> List[str]:
        prefix, core, suffix = _split_punctuation(word)
        if not core:
            return [word]

        match = _NUM_TOKEN_RE.match(core)
        if not match:
            return [word]

        number_part, percent_part, tail = match.groups()

        if percent_part:
            if tail and tail not in _ALL_INFLECTIONS:
                return [word]
            number_words = self.num_converter.georgian_number(number_part)
            percent_word = self.num_converter.percent_string
            if tail:
                percent_word = percent_word + tail
            words = number_words.split()
            if not words:
                return [word]
            words.append(percent_word)
            words[0] = prefix + words[0]
            words[-1] = words[-1] + suffix
            return words

        if tail:
            return [word]

        number_words = self.num_converter.georgian_number(number_part)
        words = number_words.split()
        if not words:
            return [word]
        words[0] = prefix + words[0]
        words[-1] = words[-1] + suffix
        return words

    def _process_inflections(self, word: str) -> List[str]:
        if self._pending_word is None:
            self._pending_word = word
            return []

        prev_word = self._pending_word
        new_prev, new_curr = self._apply_inflection(prev_word, word)

        if not new_curr:
            self._pending_word = None
            return [self._apply_punctuation_multiplier(new_prev)]

        self._pending_word = new_curr
        return [self._apply_punctuation_multiplier(new_prev)]

    def _apply_inflection(self, prev_word: str, current_word: str) -> Tuple[str, str]:
        prev_base, prev_punct = strip_punctuation(prev_word)
        cur_base, cur_punct = strip_punctuation(current_word)

        if cur_base in _ALL_INFLECTIONS:
            new_prev = prev_base.removesuffix("ი") + cur_base[1:]
            return restore_punctuation(new_prev, prev_punct), ""

        for inflection in _REMOVAL_INFLECTIONS:
            if inflection in cur_base:
                new_prev = prev_base.removesuffix("ი")
                new_cur = cur_base.removesuffix(inflection).removesuffix("ი") + inflection[1:]
                return (
                    restore_punctuation(new_prev, prev_punct),
                    restore_punctuation(new_cur, cur_punct),
                )

        for inflection in _ADDITION_INFLECTIONS:
            if inflection in cur_base:
                new_prev = prev_base.removesuffix("ი") + inflection[1:]
                new_cur = cur_base.removesuffix(inflection).removesuffix("ი") + inflection[1:]
                return (
                    restore_punctuation(new_prev, prev_punct),
                    restore_punctuation(new_cur, cur_punct),
                )

        for inflection in _KEEP_INFLECTIONS:
            if inflection in cur_base:
                new_prev = prev_base
                new_cur = cur_base.removesuffix(inflection).removesuffix("ი") + inflection[1:]
                return (
                    restore_punctuation(new_prev, prev_punct),
                    restore_punctuation(new_cur, cur_punct),
                )

        return prev_word, current_word

    def _apply_punctuation_multiplier(self, word: str) -> str:
        if self.comma_multiplier <= 1:
            return word
        return word.replace(",", "," * self.comma_multiplier).replace(".", "." * self.comma_multiplier)
