import re
import tiktoken


class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        # break into words and punctuation
        toks = re.split(r'([,.?_!"()\']|--|\s)', text)
        # remove whitespace, so it only contain words
        toks = [item.strip() for item in toks if item.strip()]
        toks = [item if item in self.str_to_int else "<|unk|>" for item in toks]
        # convert into toks
        ids = [self.str_to_int[s] for s in toks]
        return ids

    def decode(self, ids):
        # convert token_ids to tokens and join then with whitespace
        text = " ".join(self.int_to_str[i] for i in ids)
        # remove whitespace before punctuation, such as "Hello , world !" into "Hello, world!"
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text


# TODO: using tiktoken for now. Implement from scratch later
class Tokenizer:
    def __init__(self, encoding: str = "gpt2"):
        self.tokenizer = tiktoken.get_encoding(encoding)

    def encode(self, text) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids) -> str:
        return self.tokenizer.decode(ids)
