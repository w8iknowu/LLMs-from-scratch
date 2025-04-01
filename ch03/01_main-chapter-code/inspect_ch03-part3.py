"""
<This part is still in process.>

My code for inspection coressponding to ch03.
It's focusing on chapter 3.4.2, Implementing a compact SelfAttention class

"""
import torch
from torch import Tensor
from torch.nn.modules.linear import Linear


class SelfAttention_v1(torch.nn.Module):
    """
    A compact implementation of self-attention with trainable weights
    """
    def __init__(self, d_in: int = 3, d_out: int = 2):
        super().__init__()
        # trainable weights initialization
        # default shape: (3,2)
        self.w_query: Tensor = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)
        self.w_key: Tensor = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)
        self.w_value: Tensor = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)

    def forward(self, input_embed: Tensor) -> Tensor:
        # three transformations
        # unlike code in part2, here we perform full transformation of query vector
        query_embed: Tensor = torch.matmul(input_embed, self.w_query)
        keys_embed: Tensor = torch.matmul(input_embed, self.w_key)
        values_embed: Tensor = torch.matmul(input_embed, self.w_value)

        # thus the attention scores become a batch process
        # the results become every word's corresponding attention score
        attention_scores: Tensor = torch.matmul(query_embed, keys_embed.T)
        attention_weights: Tensor = torch.softmax(attention_scores / keys_embed.shape[-1]**0.5, dim=-1)

        # then here we get the final results, of every word's context vector
        return torch.matmul(attention_weights, values_embed)


class SelfAttention_v2(torch.nn.Module):
    """
    implementation using pytorch's linear layer
    """
    def __init__(self, d_in: int = 3, d_out: int = 2, use_bias: bool = False):
        super().__init__()
        # Linear method have a initialized weight, better than simply using 'torch.rand'
        # and this leads to a better training process.
        self.w_query: Linear = torch.nn.Linear(d_in, d_out, bias=use_bias)
        self.w_key: Linear = torch.nn.Linear(d_in, d_out, bias=use_bias)
        self.w_value: Linear = torch.nn.Linear(d_in, d_out, bias=use_bias)

    def forward(self, input_embed: Tensor) -> Tensor:
        query_embed: Tensor = self.w_query(input_embed)
        keys_embed: Tensor = self.w_key(input_embed)
        values_embed: Tensor = self.w_value(input_embed)

        attention_scores: Tensor = torch.matmul(query_embed, keys_embed.T)
        attention_weights: Tensor = torch.softmax(attention_scores / keys_embed.shape[-1]**0.5, dim=-1)

        return torch.matmul(attention_weights, values_embed)


def test_v1(input_tensor: Tensor) -> None:
    torch.manual_seed(123)
    sa_v1 = SelfAttention_v1()
    print(sa_v1(input_tensor))


def test_v2(input_tensor: Tensor) -> None:
    torch.manual_seed(789)
    sa_v2 = SelfAttention_v2()
    print(sa_v2(input_tensor))


if __name__ == '__main__':
    # shape: (6, 3)
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your     (x^1)
         [0.55, 0.87, 0.66],  # journey  (x^2)
         [0.57, 0.85, 0.64],  # starts   (x^3)
         [0.22, 0.58, 0.33],  # with     (x^4)
         [0.77, 0.25, 0.10],  # one      (x^5)
         [0.05, 0.80, 0.55]]  # step     (x^6)
    )
    test_v1(inputs)
    test_v2(inputs)
