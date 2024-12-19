"""
A naive script exercise referring to dataloader.ipynb
"""
from pathlib import Path
from dataclasses import dataclass
import itertools

from torch.utils.data import Dataset, DataLoader
import tiktoken
from torch import Tensor


def get_full_text() -> str:
    input_file_path = r'./the-verdict.txt'
    input_text = Path(input_file_path)
    if not input_text.exists():
        raise FileNotFoundError(f'File not found on: {input_text}')
    return input_text.read_text()


@dataclass
class GPTDatasetV1(Dataset):
    input_ids: list[list[int]]
    label_ids: list[list[int]]
    tokenizer: tiktoken.core.Encoding

    @staticmethod
    def get_instance(full_text: str,
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
        inputs: list[list[int]] = []
        labels: list[list[int]] = []

        token_ids: list[int] = tokenizer.encode(full_text, allowed_special={'<|endoftext|>'})
        for index in range(0, len(token_ids) - sequence_size, stride):
            input_token = token_ids[index:index + sequence_size]
            label_token = token_ids[index + 1:index + sequence_size + 1]
            inputs.append(input_token)
            labels.append(label_token)

        return GPTDatasetV1(inputs, labels, tokenizer)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index) -> tuple[list[int], list[int]]:
        return self.input_ids[index], self.label_ids[index]

    def inspect(self) -> None:
        sequences: list[list[int]] = self.input_ids[:10]
        print(sequences)
        print(len(sequences))
        flatten_sequence = list(itertools.chain.from_iterable(self.input_ids[:10]))
        flatten_sequence_decoded = self.tokenizer.decode(flatten_sequence)
        print(flatten_sequence)
        print(len(flatten_sequence))
        print(flatten_sequence_decoded)
        print(len(flatten_sequence_decoded))
        for x in self.input_ids[:10]:
            first = self.tokenizer.decode([x[0]])
            second = self.tokenizer.decode([x[1]])
            third = self.tokenizer.decode([x[2]])
            fourth = self.tokenizer.decode([x[3]])
            print(f'{x}-->[{first}]+[{second}]+[{third}]+[{fourth}]-->[{self.tokenizer.decode(x)}]')


def create_dataloder_v1(txt: str, batch_size: int = 4,
                        sequence_size: int = 256, stride: int = 128,
                        shuffle: bool = True, drop_last: bool = True, num_workers: int = 0) -> DataLoader:
    tokenizer = tiktoken.get_encoding('gpt2')

    my_dataset: GPTDatasetV1 = GPTDatasetV1.get_instance(txt, tokenizer, sequence_size, stride)

    return DataLoader(my_dataset, batch_size, shuffle, drop_last=drop_last, num_workers=num_workers)


def embedding_layer(x: Tensor):
    raise NotImplementedError


def dataset_test() -> None:
    full_text = get_full_text()
    tokenizer = tiktoken.get_encoding('gpt2')

    my_dataset = GPTDatasetV1.get_instance(full_text, tokenizer, 4, 4)
    print(f'dataset size: {len(my_dataset)}\n')
    first_pair = my_dataset[0]
    print(f'train: [{tokenizer.decode(first_pair[0])}]--> label: [{tokenizer.decode(first_pair[1])}]\n')


if __name__ == '__main__':
    full_text = get_full_text()

    my_dataloader: DataLoader = create_dataloder_v1(full_text, batch_size=8, sequence_size=4, stride=4)

    for index, (input_sequence, label_sequence) in enumerate(my_dataloader):
        # list[tensor]
        print(f'input({len(input_sequence)}--{type(input_sequence[0])}): {input_sequence}')
        print(f'label({len(label_sequence)}--{type(label_sequence[0])}): {label_sequence}')

        break
