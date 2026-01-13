import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from streaming_chunker import StreamingTTSChunker
from text_utils.numbers_to_text import NumberConverter
from text_utils.streaming_text_processor import StreamingTextProcessor


def collect_streamed_text(tokens):
    processor = StreamingTextProcessor(num_converter=NumberConverter(), comma_multiplier=1)
    chunker = StreamingTTSChunker(text_processor=processor)
    emitted = []
    for token in tokens:
        emitted.extend(chunker.push_token(token))
    emitted.extend(chunker.finalize())
    return "".join([chunk for chunk in emitted if chunk]).strip()


class TestStreamingTextProcessor(unittest.TestCase):
    def test_percent_inflections_streamed(self):
        tokens = [
            "იპოთეკური ",
            "სესხის ",
            "საპროცენტო ",
            "განაკვეთი ",
            "თიბისი ",
            "ბანკში ",
            "მერყეობს ",
            "8",
            "%",
            "-დან ",
            "12",
            "%",
            "-მდე,",
            " დამოკიდებულია ",
            "სხვადასხვა ",
            "ფაქტორზე.",
        ]
        expected = (
            "იპოთეკური სესხის საპროცენტო განაკვეთი თიბისი ბანკში მერყეობს "
            "რვა პროცენტიდან თორმეტი პროცენტამდე, დამოკიდებულია სხვადასხვა ფაქტორზე."
        )
        self.assertEqual(collect_streamed_text(tokens), expected)


if __name__ == "__main__":
    unittest.main()
