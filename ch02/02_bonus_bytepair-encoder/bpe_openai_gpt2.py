# Source: https://github.com/openai/gpt-2/blob/master/src/encoder.py
# License:
# Modified MIT License

# Software Copyright (c) 2019 OpenAI

# We don’t claim ownership of the content you create with GPT-2, so it is yours to do with as you please.
# We only ask that you use GPT-2 responsibly and clearly indicate your content was created using GPT-2.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# The above copyright notice and this permission notice need not be included
# with content created by the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.

import os
import json
import regex as re
from functools import lru_cache

import requests
from tqdm import tqdm


@lru_cache()
def bytes_to_unicode() -> dict[int, str]:
    """
    Returns list of utf-8 byte and a corresponding list of Unicode strings.
    The reversible bpe codes work on Unicode strings.
    This means you need a large # of Unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and Unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word) -> set[tuple]:
    """
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Encoder:
    def __init__(self, encoder: dict[str, int], bpe_merges: list[tuple[str, ...]], errors: str = 'replace'):
        self.encoder: dict[str, int] = encoder
        self.decoder: dict[int, str] = {v: k for k, v in self.encoder.items()}
        self.errors: str = errors  # how to handle errors in decoding
        self.byte_encoder: dict[int, str] = bytes_to_unicode()
        self.byte_decoder: dict[str, int] = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))  # vocab.bpe tuple and its number mapping
        self.cache = {}

        # Should have added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token: str):
        """
        ensure thw token is correctly processed, even it's not in pre-defined vocabulary.
        """
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)  # this split the string into individual characters
        pairs: set[tuple] = get_pairs(word)

        if not pairs:
            # usually for syntax
            print(f'{token} totally not processed.')
            return token
        print(f'word ready: {word}')
        print(f'pairs ready: {pairs}')

        while True:
            # take the bpe with smaller index value
            # before this while-loop, a word is first split into individual chars, and then
            bigram: tuple | float = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            print(f'processing {pairs}, min is {bigram}')
            if bigram not in self.bpe_ranks:
                # usually happens when encounter a word not in vocabulary
                # by this time, this word is already split in a suitable form, and is ready-to-use
                print(f'{pairs} break\n')
                break
            first, second = bigram
            new_word = []
            i = 0

            # split word, and put sub-words in new_word
            while i < len(word):
                try:
                    # split by sort of punctuation
                    j = word.index(first, i)
                    new_word.extend(word[i:j])  # be aware this is a list operation, not concatenate or join
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                # if found that first and second pair are continuous in this word
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            print(f'new word: {new_word}')

            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                # this means the sub-words are become one, this is the final result
                # this usually happens with a word already in our vocabulary
                print(f'{word} break with len 1\n')
                break
            else:
                pairs = get_pairs(word)

        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            # use regex to split the text to sub-words
            # for every sub-word, encode every single char of them, and concatenate them again
            token: str = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            print(token)
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text


def get_encoder(model_name, models_dir):
    with open(os.path.join(models_dir, model_name, 'encoder.json'), 'r') as f:
        encoder = json.load(f)
    with open(os.path.join(models_dir, model_name, 'vocab.bpe'), 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    return Encoder(encoder=encoder, bpe_merges=bpe_merges)


def download_vocab():
    # Modified code from
    subdir = 'gpt2_model'
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    subdir = subdir.replace('\\', '/')  # needed for Windows

    for filename in ['encoder.json', 'vocab.bpe']:
        r = requests.get("https://openaipublic.blob.core.windows.net/gpt-2/models/117M/" + filename, stream=True)

        with open(os.path.join(subdir, filename), 'wb') as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)
