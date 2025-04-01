"""
A temp test file, may be deleted in the future.
"""
import torch


torch.manual_seed(0)


def matmul_1() -> None:
    arr_1 = [
        # science, math, english
        [1, 2, 3],
        [3, 4, 5],
        [5, 6, 7],
        [7, 8, 9]
    ]
    x = torch.tensor(arr_1, dtype=torch.float)

    print(f'x=\n{x}\n')
    print(x.shape)
    print(type(x))

    print()

    arr_2 = [
        [1, 1, 1],  # all matter
        [8, 1, 1],  # only care science
        [3, 3, 4],  # avg, but english a little bit more
    ]
    y = torch.tensor(arr_2, dtype=torch.float)

    print(f'y=\n{y}\n')
    print(y.shape)
    print(type(y))

    print()

    z = torch.matmul(x, y)
    print(f'z=\n{z}\n')
    print(z.shape)
    print(type(z))


if __name__ == '__main__':
    matmul_1()
