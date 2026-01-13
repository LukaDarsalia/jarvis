import string
from typing import Callable, List, Tuple


def strip_punctuation(word: str) -> Tuple[str, str]:
    """Separates trailing punctuation from a word."""
    for i in range(len(word) - 1, -1, -1):
        if word[i] not in string.punctuation:
            return word[:i + 1], word[i + 1:]
    return word, ""


def restore_punctuation(word: str, punctuation: str) -> str:
    return word + punctuation


class NumberConverter:
    def __init__(self) -> None:
        self.units = [
            "ნული",
            "ერთი",
            "ორი",
            "სამი",
            "ოთხი",
            "ხუთი",
            "ექვსი",
            "შვიდი",
            "რვა",
            "ცხრა",
        ]
        self.teens = [
            "ათი",
            "თერთმეტი",
            "თორმეტი",
            "ცამეტი",
            "თოთხმეტი",
            "თხუთმეტი",
            "თექვსმეტი",
            "ჩვიდმეტი",
            "თვრამეტი",
            "ცხრამეტი",
        ]
        self.tens = ["", "", "ოცი", "ოცდაათი", "ორმოცი", "ორმოცდაათი", "სამოცი", "სამოცდაათი", "ოთხმოცი", "ოთხმოცდაათი"]
        self.hundreds = ["", "ასი", "ორასი", "სამასი", "ოთხასი", "ხუთასი", "ექვსასი", "შვიდასი", "რვაასი", "ცხრაასი"]
        self.point = "მთელი"
        self.percent_string = "პროცენტი"

    def convert_hundreds(self, n: int) -> str:
        if n == 0:
            return ""
        if n <= 9:
            return self.units[n]
        if n <= 19:
            return self.teens[n - 10]
        if n <= 99:
            if n == 20:
                return self.tens[2]
            tens, units = divmod(n, 20)
            tens_word = ["ოცი", "ორმოცი", "სამოცი", "ოთხმოცი"][tens - 1]
            if units == 0:
                return tens_word
            return f"{tens_word[:-1]}და{self.convert_hundreds(units)}"

        hundreds, remainder = divmod(n, 100)
        if remainder == 0:
            return self.hundreds[hundreds]
        return f"{self.hundreds[hundreds][:-1]} {self.convert_hundreds(remainder)}"

    def convert_large(self, n: int) -> str:
        if n < 1000:
            return self.convert_hundreds(n)
        if n // 1000 == 1:
            thousands, rest = divmod(n, 1000)
            return "ათასი" if rest == 0 else f"ათას {self.convert_large(rest)}"
        if n < 1_000_000:
            thousands, rest = divmod(n, 1000)
            prefix = self.convert_hundreds(thousands)
            prefix += " ათას" if rest != 0 else " ათასი"

            rest_str = self.convert_hundreds(rest) if rest else ""
            return f"{prefix} {rest_str}".strip()

        millions, rest = divmod(n, 1_000_000)
        million_part = "მილიონი" if rest == 0 else "მილიონ"
        prefix = f"{self.convert_hundreds(millions)} {million_part}"
        rest_str = self.convert_large(rest) if rest else ""
        return f"{prefix} {rest_str}".strip()

    def check_and_remove_trailing_zeroes(self, num: str) -> str:
        if "." in num or "," in num:
            if "." in num:
                int_part, frac_part = num.split(".")
            else:
                int_part, frac_part = num.split(",")

            try:
                if int(frac_part) == 0:
                    return int_part
            except ValueError:
                return num

        return num

    def georgian_number(self, num: str) -> str:
        try:
            if num.startswith("-"):
                return "მინუს " + self.georgian_number(num=num[1:])
            num = self.check_and_remove_trailing_zeroes(num=num)

            if num.startswith("0") and len(num) > 1:
                return " ".join(self.units[int(digit)] for digit in num)

            if "." in num or "," in num:
                if "." in num:
                    int_part, frac_part = num.split(".")
                else:
                    int_part, frac_part = num.split(",")

                int_str = self.convert_large(abs(int(int_part)))
                frac_str = self.convert_large(int(frac_part))
                return int_str + " " + self.point + " " + frac_str

            n = int(num)
            if n == 0:
                return "ნული"
            if n < 0:
                return self.convert_large(abs(n))
            return self.convert_large(n)
        except ValueError:
            return num
        except Exception:
            return num

    def __check_and_replace_inflections(
        self,
        possible_inflections: List[str],
        text: List[str],
        removal_rule: Callable[[str, str], str],
    ) -> List[str]:
        res = text.copy()
        for index, word in enumerate(res):
            word_base, word_punct = strip_punctuation(word)

            for inflection in possible_inflections:
                if inflection in word_base:
                    if index - 1 >= 0:
                        prev_word_base, prev_punct = strip_punctuation(res[index - 1])
                        new_prev_word = removal_rule(prev_word_base, inflection)
                        res[index - 1] = restore_punctuation(new_prev_word, prev_punct)

                        modified_word = word_base.removesuffix(inflection).removesuffix("ი") + inflection[1:]
                        res[index] = restore_punctuation(modified_word, word_punct)

        return res

    def check_for_inflections(self, text: List[str]) -> List[str]:
        removal_inflections = ["-ს", "-ად"]
        addition_inflections = ["-მა", "-ო"]
        keep_inflections = ["-ის", "-ით"]

        text = self.__check_and_replace_inflections(
            removal_inflections,
            text=text,
            removal_rule=lambda word, _: word.removesuffix("ი"),
        )

        text = self.__check_and_replace_inflections(
            addition_inflections,
            text=text,
            removal_rule=lambda word, inflection: word.removesuffix("ი") + inflection[1:],
        )

        text = self.__check_and_replace_inflections(
            keep_inflections,
            text=text,
            removal_rule=lambda word, _: word,
        )

        return text

    def convert_numbers(self, text: str) -> str:
        text = text.replace("%", f" {self.percent_string}")
        converted_texts = [self.georgian_number(word) for word in text.split(" ")]

        result = self.check_for_inflections(converted_texts)
        return " ".join(result)
