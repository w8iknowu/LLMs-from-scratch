import torch


def inspect_embedding_layer() -> None:
    torch.manual_seed(123)

    embedding = torch.nn.Embedding(4, 1)
    print(embedding.weight)
    # print(embedding(torch.tensor([3])))
    # print(embedding(torch.tensor([3, 1, 2])))


def inspect_linear_layer() -> None:
    torch.manual_seed(123)
    linear_layer = torch.nn.Linear(4, 5, bias=False)

    x = torch.rand((3, 4))
    print(f'===>input({x.shape}):\n{x}\n')
    print(f'===>weight size: {linear_layer.weight.shape}\n')
    y = linear_layer(x)
    print(f'===>output({y.shape}):\n{y}\n')


if __name__ == '__main__':
    inspect_linear_layer()

