"""
A naive script exercise referring to dataloader.ipynb
"""
from pathlib import Path
from dataclasses import dataclass

from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import tiktoken
import torch


def get_full_text() -> str:
    input_file_path = r'./the-verdict.txt'
    input_text = Path(input_file_path)
    if not input_text.exists():
        raise FileNotFoundError(f'File not found on: {input_text}')
    return input_text.read_text()


@dataclass
class GPTDatasetV1(Dataset):
    input_ids: list[Tensor]
    label_ids: list[Tensor]
    tokenizer: tiktoken.core.Encoding

    @classmethod
    def get_instance(cls, full_text: str,
                     tokenizer: tiktoken.core.Encoding,
                     sequence_size: int, stride: int) -> 'GPTDatasetV1':
        """build the dataset instance by using a sliding window way.

        Parameters
        ----------
        full_text : str
            pass
        tokenizer : tiktoken.core.Encoding
            pass
        sequence_size : int
            for every processed sample, how many tokens are included in it.
            here we use this parameter to define the sizes of both input and label.
        stride : int
            the step of this sliding window when it moves forward.
            be aware of the values of strid and sequence_size, if strid is larger than sequence_size,
            this will cause the chunked data not continous.

        Returns
        -------
        GPTDatasetV1
            pass
        """
        inputs: list[Tensor] = []
        labels: list[Tensor] = []

        token_ids: list[int] = tokenizer.encode(full_text, allowed_special={'<|endoftext|>'})
        for index in range(0, len(token_ids) - sequence_size, stride):
            input_token = token_ids[index:index + sequence_size]
            label_token = token_ids[index + 1:index + sequence_size + 1]
            inputs.append(torch.tensor(input_token))
            labels.append(torch.tensor(label_token))

        return cls(inputs, labels, tokenizer)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index) -> tuple[list[int], list[int]]:
        return self.input_ids[index], self.label_ids[index]


def create_dataloder_v1(txt: str, batch_size: int = 4,
                        sequence_size: int = 256, stride: int = 128,
                        shuffle: bool = True, drop_last: bool = True, num_workers: int = 0) -> DataLoader:
    """to enhance dataset through dataloader
    """
    tokenizer = tiktoken.get_encoding('gpt2')

    my_dataset: GPTDatasetV1 = GPTDatasetV1.get_instance(txt, tokenizer, sequence_size, stride)

    return DataLoader(my_dataset, batch_size, shuffle, drop_last=drop_last, num_workers=num_workers)


def embedding_layer(x: Tensor):
    raise NotImplementedError


@dataclass
class EmbeddingMethods:
    vocab_size: int
    content_length: int
    output_dim: int
    token_embedding_layer: torch.nn.modules.sparse.Embedding
    pos_embedding_layer: torch.nn.modules.sparse.Embedding

    @classmethod
    def init_embedding_methods(cls, vocab_size: int, content_length: int, output_dim: int):
        token_embedding = torch.nn.Embedding(vocab_size, output_dim)

        # this content_length, or context_length, is can been seen as a max-length for model to understand,
        # you know, just like what we saw in many LLM's feature description today
        # just as this name, "position", if we set length to 1024,
        # that indicates we grant the continous 1024 tokens' order in a text got their own meaning
        # for example, the text "i love you", if we dont add a postion embeddng, which means word's order in text not matters,
        # this sentence may be seen as the same meaning of "you love i", which should not.
        pos_embedding = torch.nn.Embedding(content_length, output_dim)

        return cls(vocab_size, content_length, output_dim, token_embedding, pos_embedding)

    def embed_token_layer(self, x: torch.Tensor):
        return self.token_embedding_layer(x)

    def embed_pos_layer(self, max_length: int):
        return self.pos_embedding_layer(torch.arange(max_length))


def dataset_test() -> None:
    full_text = get_full_text()
    tokenizer = tiktoken.get_encoding('gpt2')

    my_dataset = GPTDatasetV1.get_instance(full_text, tokenizer, 4, 4)
    print(f'dataset size: {len(my_dataset)}\n')
    first_pair = my_dataset[0]
    print(f'train: [{tokenizer.decode(first_pair[0])}]--> label: [{tokenizer.decode(first_pair[1])}]\n')


def main() -> None:
    # some parameters
    torch.manual_seed(0)  # comment this will produce different result every time
    vocab_size = 50327
    output_dim = 256
    context_length = 1024
    max_length = 4

    full_text: str = get_full_text()

    my_dataloader: DataLoader = create_dataloder_v1(full_text, batch_size=8, sequence_size=4, stride=4)

    for index, (input_sequence, label_sequence) in enumerate(my_dataloader):
        print(type(input_sequence))
        print(label_sequence)
        embedding = EmbeddingMethods.init_embedding_methods(vocab_size, context_length, output_dim)

        token_embeddings: Tensor = embedding.embed_token_layer(input_sequence)
        pos_embeddings: Tensor = embedding.embed_pos_layer(max_length)

        input_embeddings: Tensor = token_embeddings + pos_embeddings
        print(type(input_embeddings))
        print(input_embeddings.shape)

        break


if __name__ == '__main__':
    main()
