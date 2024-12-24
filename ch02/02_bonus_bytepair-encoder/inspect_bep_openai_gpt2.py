"""
some printer to inspect ./bpe_openai_gpt2.py
"""
from pathlib import Path
import json

from bpe_openai_gpt2 import bytes_to_unicode

ONE_BYTE_ROOM: int = 2 ** 8


def inspect_get_word_encoder() -> dict[str, int]:
    word_encoder_path = Path('./gpt2_model/encoder.json')
    with word_encoder_path.open(mode='r') as f:
        # encoder.json basically is a key-value map,
        # mapping every syntax to a unique number
        encoder: dict[str, int] = json.load(f)
    print(f'encoder len: {len(encoder)}')

    # reverse the encoder to get the decoder
    decoder: dict[int, str] = {v: k for k, v in encoder.items()}
    print(f'decoder len: {len(decoder)}')

    word = r'hello'
    print(f'text: {word}')
    text_encoded = encoder[word]
    print(f'encoder: {text_encoded}')
    text_decoded = decoder[text_encoded]
    print(f'decoder: {text_decoded}')
    return encoder


def inspect_get_bpe_encoder() -> list[tuple[str, ...]]:
    bpe_encoder_path = Path('./gpt2_model/vocab.bpe')
    with bpe_encoder_path.open(mode='r', encoding='utf-8') as f:
        bpe_data = f.read()
    bpe: list[tuple[str, ...]] = [tuple(x.split()) for x in bpe_data.split('\n')[1:-1]]
    return bpe


def inspect_bytes_to_unicode() -> dict[int, str]:
    # https://web.itu.edu.tr/sgunduz/courses/mikroisl/ascii.html
    # printable ascii characters, ranging from 33('!') to 126('~')
    # noted that 32(space character) and 127(delete character) are also printable, but omitted here
    # as for 0~31, they are un-printable, used for control characters
    ascii_printable_range = range(ord('!'), ord('~') + 1)
    ascii_printable_list: list[int] = list(ascii_printable_range)
    # print(f'ascii_printable_list len: {len(ascii_printable_list)}')

    # for 128~255, they are extended ascii characters
    # https://en.wikipedia.org/wiki/Latin-1_Supplement
    # here we take from 161 to 172, and 174 to 255
    latin_1_supplement_range_1: list[int] = list(range(ord('¡'), ord('¬') + 1))  # U+00A0(Non-breaking space) not included
    latin_1_supplement_range_2: list[int] = list(range(ord('®'), ord('ÿ') + 1))  # U+00AD(Soft hyphen) not included
    ascii_extended = latin_1_supplement_range_1 + latin_1_supplement_range_2

    # all bytes' indices mapping
    ascii_main: list[int] = ascii_printable_list + ascii_extended
    unicodes_with_chars_beyond_one_byte = ascii_main[:]

    # for the rest spaces within one byte room(256),
    # "overwrite" them with characters whose unicodes are beyond 255
    offset: int = 0
    for b in range(ONE_BYTE_ROOM):
        if b not in ascii_main:
            ascii_main.append(b)
            unicodes_with_chars_beyond_one_byte.append(ONE_BYTE_ROOM + offset)
            offset += 1
    # print(unicodes_with_chars_beyond_one_byte)
    unicodes_main = [chr(x) for x in unicodes_with_chars_beyond_one_byte]
    return dict(zip(ascii_main, unicodes_main))


def my_bytes_to_unicode() -> dict[int, str]:
    # printable
    ascii_printable_range = range(ord('!'), ord('~') + 1)
    ascii_printable_list: list[int] = list(ascii_printable_range)

    # extended
    latin_1_supplement_range_1: list[int] = list(range(ord('¡'), ord('¬') + 1))  # U+00A0(Non-breaking space) not included
    latin_1_supplement_range_2: list[int] = list(range(ord('®'), ord('ÿ') + 1))  # U+00AD(Soft hyphen) not included
    ascii_extended: list[int] = latin_1_supplement_range_1 + latin_1_supplement_range_2

    ascii_main: list[int] = ascii_printable_list + ascii_extended
    ascii_main_table: dict[int, str] = {index: chr(index) for index in ascii_main}

    unhandled_ascii_codes: list[int] = [x for x in range(ONE_BYTE_ROOM) if x not in ascii_main]
    characters_beyond_one_byte: dict[int, str] = {x: chr(ONE_BYTE_ROOM+offset) for offset, x in enumerate(unhandled_ascii_codes)}

    # merge
    ascii_main_table.update(characters_beyond_one_byte)
    return ascii_main_table


def check_bytes_implementation() -> None:
    original_ver = bytes_to_unicode()
    inspect_ver = inspect_bytes_to_unicode()
    my_ver = my_bytes_to_unicode()
    assert len(original_ver) == len(inspect_ver)
    assert len(original_ver) == len(my_ver)
    for i in range(ONE_BYTE_ROOM):
        assert original_ver[i] == inspect_ver[i], f'Not equal({i}): {original_ver[i]} != {inspect_ver[i]}'
        assert original_ver[i] == my_ver[i], f'Not equal({i}): {original_ver[i]} != {my_ver[i]}'
    print('All bytes were correct')


if __name__ == '__main__':
    check_bytes_implementation()
