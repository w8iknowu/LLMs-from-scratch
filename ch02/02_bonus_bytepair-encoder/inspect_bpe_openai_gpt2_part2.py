from bpe_openai_gpt2 import get_encoder, Encoder, get_pairs


def inspect_encode():
    text = r"Don't go gentle into that good night."
    text = r"Don't go gentle into that good subnight."
    encoder: Encoder = get_encoder(model_name="gpt2_model", models_dir=".")
    encoder.encode(text)
    """
    Don
    't
    Ġgo
    Ġgentle
    Ġinto
    Ġthat
    Ġgood
    Ġnight
    .
    """
    # z = tuple('Ġnight')
    # print(z)
    # x = get_pairs(z)
    # print(x)
    print('========\n')
    # print(encoder.bpe_ranks.get(('Ġn', 'igh')))
    # print(encoder.bpe_ranks.get(('igh', 't')))


def inspect_get_pairs() -> None:
    input_text = tuple('word')
    input_text = ('w', 'or', 'd')

    pairs = set()
    prev_char = input_text[0]
    for char in input_text[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    print(pairs)


if __name__ == "__main__":
    # inspect_encode()
    inspect_get_pairs()
