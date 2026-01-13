import re
import string
from typing import List, Optional, Tuple

from .numbers_to_text import NumberConverter, strip_punctuation, restore_punctuation


_REMOVAL_INFLECTIONS = ["-ს", "-ად"]
_ADDITION_INFLECTIONS = ["-მა", "-ო"]
_KEEP_INFLECTIONS = ["-ის", "-ით"]
_LOCATIVE_INFLECTIONS = ["-დან", "-მდე"]
_ALL_INFLECTIONS = (
    _REMOVAL_INFLECTIONS
    + _ADDITION_INFLECTIONS
    + _KEEP_INFLECTIONS
    + _LOCATIVE_INFLECTIONS
)

_DASH_ONLY_RE = re.compile(r"^[-–—]+$")
_NUM_TOKEN_RE = re.compile(r"^(-?\d+(?:[.,]\d+)?)(%?)(.*)$")
_LATIN_LETTER_RE = re.compile(r"[A-Za-z]")

_PUNCT_STRIP_CHARS = set(string.punctuation) - {"%", "-"}
_PUNCT_STRIP_CHARS.update({"“", "”", "‘", "’", "…"})

_KNOWN_WORD_MAP = {
    "visa": "ვიზა",
    "mastercard": "მასტერქარდი",
    "card": "ქარდი",
}

_LETTER_NAME_MAP = {
    "a": "ეი",
    "b": "ბი",
    "c": "სი",
    "d": "დი",
    "e": "ი",
    "f": "ეფ",
    "g": "ჯი",
    "h": "ეიჩ",
    "i": "აი",
    "j": "ჯეი",
    "k": "კეი",
    "l": "ელ",
    "m": "ემ",
    "n": "ენ",
    "o": "ო",
    "p": "პი",
    "q": "ქიუ",
    "r": "არ",
    "s": "ესი",
    "t": "ტი",
    "u": "იუ",
    "v": "ვი",
    "w": "დაბლიუ",
    "x": "ექს",
    "y": "უაი",
    "z": "ზი",
}

_LATIN_DIGRAPHS = [
    ("tch", "ჩ"),
    ("sch", "შ"),
    ("sh", "შ"),
    ("ch", "ჩ"),
    ("zh", "ჟ"),
    ("ts", "ც"),
    ("dz", "ძ"),
    ("gh", "ღ"),
    ("kh", "ხ"),
    ("ph", "ფ"),
    ("th", "თ"),
    ("qu", "ქვ"),
    ("ck", "კ"),
]

_LATIN_LETTER_MAP = {
    "a": "ა",
    "b": "ბ",
    "c": "კ",
    "d": "დ",
    "e": "ე",
    "f": "ფ",
    "g": "გ",
    "h": "ჰ",
    "i": "ი",
    "j": "ჯ",
    "k": "კ",
    "l": "ლ",
    "m": "მ",
    "n": "ნ",
    "o": "ო",
    "p": "პ",
    "q": "ქ",
    "r": "რ",
    "s": "ს",
    "t": "ტ",
    "u": "უ",
    "v": "ვ",
    "w": "ვ",
    "x": "ქს",
    "y": "ი",
    "z": "ზ",
}


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
        comma_multiplier: int = 1,
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
            return self._expand_latin_word(word, prefix, core, suffix)

        number_part, percent_part, tail = match.groups()

        if percent_part:
            if tail and tail not in _ALL_INFLECTIONS:
                return [word]
            number_words = self.num_converter.georgian_number(number_part)
            percent_word = self.num_converter.percent_string
            if tail:
                percent_word = self._apply_inflection_to_word(percent_word, tail)
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

    def _expand_latin_word(self, word: str, prefix: str, core: str, suffix: str) -> List[str]:
        if not _LATIN_LETTER_RE.search(core):
            return [word]

        latin_core, inflection = self._split_inflection(core)
        if not latin_core:
            return [word]

        letters_only = re.sub(r"[^A-Za-z]", "", latin_core)
        if not letters_only:
            return [word]

        lowered = letters_only.lower()

        mapped = _KNOWN_WORD_MAP.get(lowered)
        if mapped:
            words = mapped.split()
            if inflection:
                words[-1] = self._apply_inflection_to_word(words[-1], inflection)
            words[0] = prefix + words[0]
            words[-1] = words[-1] + suffix
            return words

        if self._is_abbreviation(latin_core, letters_only):
            words = self._expand_abbreviation(letters_only)
            if not words:
                return [word]
            if inflection:
                words[-1] = self._apply_inflection_to_word(words[-1], inflection)
            words[0] = prefix + words[0]
            words[-1] = words[-1] + suffix
            return words

        transliterated = self._latin_to_georgian(latin_core)
        if not transliterated:
            return [word]
        if inflection:
            transliterated = self._apply_inflection_to_word(transliterated, inflection)
        return [prefix + transliterated + suffix]

    def _is_abbreviation(self, core: str, letters_only: str) -> bool:
        if letters_only.lower() in _KNOWN_WORD_MAP:
            return False
        if letters_only.isupper() and len(letters_only) <= 6:
            return True
        if "." in core and letters_only.isalpha() and len(letters_only) <= 6:
            return True
        return False

    def _expand_abbreviation(self, letters_only: str) -> List[str]:
        words = []
        for letter in letters_only.lower():
            name = _LETTER_NAME_MAP.get(letter)
            if not name:
                return []
            words.append(name)
        return words

    def _apply_inflection_to_word(self, word: str, inflection: str) -> str:
        if inflection == "-დან":
            if word.endswith("ი"):
                return word[:-1] + "იდან"
            return word + "დან"
        if inflection == "-მდე":
            if word.endswith("ი"):
                return word[:-1] + "ამდე"
            return word + "მდე"
        if inflection in _ALL_INFLECTIONS:
            return word.removesuffix("ი") + inflection[1:]
        return word + inflection.replace("-", "")

    def _split_inflection(self, core: str) -> Tuple[str, str]:
        for inflection in _ALL_INFLECTIONS:
            if core.endswith(inflection):
                return core[: -len(inflection)], inflection
        return core, ""

    def _latin_to_georgian(self, core: str) -> str:
        text = core.lower()
        result: List[str] = []
        i = 0
        while i < len(text):
            matched = False
            for latin, geo in _LATIN_DIGRAPHS:
                if text.startswith(latin, i):
                    result.append(geo)
                    i += len(latin)
                    matched = True
                    break
            if matched:
                continue

            char = text[i]
            mapped = _LATIN_LETTER_MAP.get(char)
            if mapped is None:
                result.append(char)
            else:
                result.append(mapped)
            i += 1
        return "".join(result)

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

        for inflection in _LOCATIVE_INFLECTIONS:
            if inflection in cur_base:
                base = cur_base.removesuffix(inflection)
                new_cur = self._apply_inflection_to_word(base, inflection)
                return (
                    restore_punctuation(prev_base, prev_punct),
                    restore_punctuation(new_cur, cur_punct),
                )

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
