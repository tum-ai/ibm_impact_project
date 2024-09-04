import os
import re
from typing import List

from src.encoding_decoding.numerical_encodings import encoding_to_number
from src.tokenizer.abstract_tokenizer import NumberEncodingTokenizer, NUMBER_REGEX


class RtTokenizer(NumberEncodingTokenizer):
    def __init__(self, embedding_dim=256, **kwargs):
        super().__init__(**kwargs)

        with open(os.path.join(os.path.dirname(__file__), "..", "..", "regression_transformer_number_tokens.txt"), 'r') as file:
            lines = file.readlines()

        num_tokens = [line.strip() for line in lines]

        self.add_tokens(num_tokens)
        self.num_tokens = num_tokens
        self.num_token_ids = [self.convert_tokens_to_ids(num_token) for num_token in num_tokens]
        self.embedding_dim = embedding_dim

    def get_num_token_ids(self):
        return self.num_token_ids

    def get_num_tokens(self):
        return self.num_tokens

    def decode_number_token(self, token):
        return encoding_to_number(token)

    def tokenize(self, text: str, add_special_tokens=False, **kwargs) -> List[str]:
        nonum_text, number_tokens = extract(text)
        out = super().tokenize(
            nonum_text, **kwargs
        )
        return out

    def t5_tokenize(self, text: str, **kwargs) -> List[str]:
        return super().tokenize(
            text, **kwargs
        )


def extract(text):
    numbers = []

    def replace(match):
        number = match.group(0).strip()
        tokens = []
        
        #Remove plus as previously we treated it like a digit
        number = number.lstrip('+-')

        if "." in number:
            integer_part, _, fractional_part = number.partition('.')
        else:
            integer_part = number
            fractional_part = []

        z = len(integer_part) - 1
        for digit in integer_part:
            tokens.append(f"_{digit}_{z}_")
            numbers.append(f"_{digit}_{z}_")
            z -= 1
        for i, digit in enumerate(fractional_part):
            tokens.append(f"_{digit}_{-i - 1}_")
            numbers.append(f"_{digit}_{-i - 1}_")

        return " ".join(tokens)

    text = separate_space_seperated_numbers_by_comma(text)
    nonum_text = re.sub(NUMBER_REGEX, replace, text)
    return nonum_text, numbers


def separate_space_seperated_numbers_by_comma(text):
    # If two numbers are just seperated by whitespace or linebreak, we must clearly split them by comma for
    # the tokenizer to recognize them as two different numbers
    regex = r"(\d+\.?\d*)\s+(\d+\.?\d*)"
    return re.sub(regex, r"\1, \2", text)


if __name__ == "__main__":
    tokenizer = RtTokenizer.from_pretrained("t5-small")
    print(tokenizer.convert_tokens_to_string(tokenizer.tokenize("(3/2)x=54\n3x=108")))
    # print(separate_space_seperated_numbers_by_comma("5.34\n4.45"))
    # print(separate_space_seperated_numbers_by_comma("5.34 4.45"))
    # print(separate_space_seperated_numbers_by_comma("5  4"))
    # print(separate_space_seperated_numbers_by_comma("54 4"))

