import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from streaming_chunker import StreamingTTSChunker, split_text_for_streaming


def collect_chunks(tokens):
    chunker = StreamingTTSChunker()
    emitted = []
    for token in tokens:
        emitted.extend(chunker.push_token(token))
    emitted.extend(chunker.finalize())
    return emitted


class TestLLMStreamChunker(unittest.TestCase):
    def test_partial_tokens_no_early_emit(self):
        tokens = ["Hel", "lo ", "wor", "ld "]
        self.assertEqual(collect_chunks(tokens), ["Hello world", "", ""])

    def test_streaming_four_words_emits_incrementally(self):
        tokens = ["Hello ", "world ", "from ", "AI"]
        self.assertEqual(collect_chunks(tokens), ["Hello world from", " AI", "", ""])

    def test_multiword_tokens_with_leading_spaces(self):
        tokens = ["Hello world ", " from", " AI"]
        self.assertEqual(collect_chunks(tokens), ["Hello world from", " AI", "", ""])

    def test_exact_three_words(self):
        tokens = ["Hello ", "world ", "from "]
        self.assertEqual(collect_chunks(tokens), ["Hello world from", "", ""])

    def test_georgian_two_words_from_partial_tokens(self):
        tokens = ["გამ", "არჯ", "ობა ", "კე", "თი", "ლი "]
        self.assertEqual(collect_chunks(tokens), ["გამარჯობა კეთილი", "", ""])

    def test_georgian_exact_three_words_with_punctuation(self):
        tokens = ["გამარჯობა", "! ", "როგორ ", "შე", "გიძლ", "იათ "]
        self.assertEqual(collect_chunks(tokens), ["გამარჯობა! როგორ შეგიძლიათ", "", ""])

    def test_georgian_four_words_streaming(self):
        tokens = ["გამარჯობა! ", "როგორ ", "შეგიძლ", "იათ ", "დაგე", "ხმარ", "ოთ?"]
        self.assertEqual(
            collect_chunks(tokens),
            ["გამარჯობა! როგორ შეგიძლიათ", " დაგეხმაროთ?", "", ""],
        )

    def test_georgian_with_commas_and_spaces(self):
        tokens = [" არ", " ვიცი", ", ", "მაგ", "რამ ", "ვცდი "]
        self.assertEqual(
            collect_chunks(tokens),
            ["არ ვიცი, მაგრამ", " ვცდი", "", ""],
        )

    def test_georgian_long_sentence(self):
        tokens = [
            "დღეს ", "თბილის", "ში ", "ძა", "ლიან ", "ცხე", "ლია ",
            "და ", "მგო", "ნი ", "წვიმ", "ს ",
        ]
        self.assertEqual(
            collect_chunks(tokens),
            ["დღეს თბილისში ძალიან", " ცხელია", " და", " მგონი", " წვიმს", "", ""],
        )

    def test_llm_style_georgian_tokens_match_full_split(self):
        tokens = [
            "გამარჯობა! ", "როგორ ", "შემიძლია ", "დაგეხმაროთ? ", "თუ ",
            "გაქვთ ", "შეკითხვები ", "თიბისი ", "ბანკთან ", "დაკავშირებით, ",
            "მე ", "მზად ", "ვარ ", "გიპასუხოთ. ", "რა ", "გაინტერესებთ?",
        ]
        expected = split_text_for_streaming("".join(tokens).strip())
        self.assertEqual(collect_chunks(tokens), expected)


if __name__ == "__main__":
    unittest.main()
