"""
Streaming chunker for LLM tokens -> TTS word chunks.
"""

from __future__ import annotations

import re
from typing import List


_TOKEN_REPLACEMENTS = {
    "\u200b": "",
    "\u200c": "",
    "\u200d": "",
    "\ufeff": "",
    "\u2581": " ",
}

_PUNCT_ONLY_RE = re.compile(r"^[\.\,\!\?\:\;\-\u2014\"'“”‘’…]+$")


def split_text_for_streaming(text: str) -> List[str]:
    text = text.replace("\n", " ").strip()
    words = text.split()

    if not words:
        return ["", ""]

    if len(words) <= 3:
        return [text, "", ""]

    chunks = [" ".join(words[:3])]
    for word in words[3:]:
        chunks.append(" " + word)
    chunks.extend(["", ""])
    return chunks


class StreamingWordBuffer:
    def __init__(self) -> None:
        self._buffer = ""

    def push(self, token: str) -> List[str]:
        if not token:
            return []

        self._buffer += token
        parts = re.split(r"\s+", self._buffer)

        if self._buffer and not self._buffer[-1].isspace():
            self._buffer = parts.pop()
        else:
            self._buffer = ""

        return [part for part in parts if part]

    def flush(self) -> List[str]:
        if self._buffer.strip():
            word = self._buffer.strip()
            self._buffer = ""
            return [word]
        self._buffer = ""
        return []


class StreamingTTSChunker:
    def __init__(self) -> None:
        self._word_buffer = StreamingWordBuffer()
        self._words: List[str] = []
        self._started = False
        self._sent_words = 0

    def push_token(self, token: str) -> List[str]:
        if token:
            for src, dst in _TOKEN_REPLACEMENTS.items():
                if src in token:
                    token = token.replace(src, dst)
        new_words = self._word_buffer.push(token)
        if new_words:
            for word in new_words:
                self._append_word(word)
        return self._emit_ready_chunks()

    def finalize(self) -> List[str]:
        new_words = self._word_buffer.flush()
        if new_words:
            for word in new_words:
                self._append_word(word)

        chunks = self._emit_ready_chunks()

        if self._started:
            chunks.extend(["", ""])
            return chunks

        if self._words:
            return split_text_for_streaming(" ".join(self._words))

        return []

    def _emit_ready_chunks(self) -> List[str]:
        chunks: List[str] = []

        if not self._started and len(self._words) >= 3:
            chunks.append(" ".join(self._words[:3]))
            self._started = True
            self._sent_words = 3

        if self._started and len(self._words) > self._sent_words:
            for word in self._words[self._sent_words:]:
                chunks.append(" " + word)
            self._sent_words = len(self._words)

        return chunks

    def _append_word(self, word: str) -> None:
        if not word:
            return
        if self._words and _PUNCT_ONLY_RE.match(word):
            self._words[-1] += word
            return
        self._words.append(word)
