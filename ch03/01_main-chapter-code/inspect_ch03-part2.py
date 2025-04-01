"""
My code for inspection coressponding to ch03.
It's focusing on chapter 3.4, self-attention with trainable weights implementation.

From this part, the familiar "q(uery), k(ey), v(alue)" come out

"""
from typing import Optional

import torch


torch.manual_seed(123)


def inspect_self_attention_weights_naive() -> None:
    """
    Corresponding to Chapter 3.4.1
    About self attention with trainable weights.

    Example text:
    "Your journey starts with one step."
    """
    # shape: (6, 3)
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your     (x^1)
         [0.55, 0.87, 0.66],  # journey  (x^2)
         [0.57, 0.85, 0.64],  # starts   (x^3)
         [0.22, 0.58, 0.33],  # with     (x^4)
         [0.77, 0.25, 0.10],  # one      (x^5)
         [0.05, 0.80, 0.55]]  # step     (x^6)
    )

    # Step-1. here still, we take x2 as the example
    # but be aware that here x2 is as the input, not the "weight" in previous part
    x2 = inputs[1]  # (3,)

    # define the input and output dimension of embedding
    d_in = inputs.shape[1]  # 3
    d_out = 2

    # Step-2. initialize three different weights(embeddings)
    # "query" weights, shape: (3, 2)
    w_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    show_tensor_info(w_query, 'query weight')

    # "key" weights, shape: (3, 2)
    w_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    show_tensor_info(w_key, 'key weight')

    # "value" weights, shape: (3, 2)
    w_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    show_tensor_info(w_value, 'value weight')

    # Step-3. compute x2's query vector
    # this is just transforming the x2, from its original three-dimension vector,
    # to a new two-dimension vector
    query_x2 = torch.matmul(x2, w_query)  # (3, )*(3, 2) ==> (1, 2)
    show_tensor_info(query_x2, 'query_x2')

    # Step-4. all key and value weights computation
    # same as last step, you can see them just as two transformations,
    # to turn the inputs into two different representaions
    # here by matrix product, every word is expressed with a new two-dimension vector,
    # comparing to original three-dimension vector.
    keys = torch.matmul(inputs, w_key)  # (6,3)*(3,2) ==> (6,2)
    values = torch.matmul(inputs, w_value)  # (6,3)*(3,2) ==> (6,2)
    show_tensor_info(keys, 'keys vector')
    show_tensor_info(values, 'values vector')

    # Step-5. computes attention score of x2
    # To perform query_x2 on the whole keys(transpose)
    # here query_x2 can be seen as a weight(or impact),
    # the weighted result reflects how query_x2 affects(contributes to) the final socre
    #
    # And this how "query" expresses:
    # we want to know x2(query_x2)'s relation with other words in matrix,
    # this is like x2 knocks every word's door and ask, "hi, I just want to know how well we are getting with".
    # And this "asking" is expressed with a mathematical computation,
    # which is to use query_x2 to perform dot product with every vector in keys.
    #
    # This is how "attention" explains itself. From the final score,
    # we get a overrall view of how x2's relations with other words.
    # noted that since we've performed a dot prodcut, you can see it as a simplified similarity calculation.
    # for example, here the result is: [1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440]
    # that indicates x2 is most similari with itself, and x3("starts") is the second most relative one to it.
    attention_score_x2 = torch.matmul(query_x2, keys.T)  # (1,2)*(2,6) ==> (1,6)
    show_tensor_info(attention_score_x2, 'attention x2')

    # Step-6. normalize score
    # after this step, we then get a probability table, telling the relation in the view of x2
    # BUT keep in mind, this result is just a probability talble, NOT the actual representaion of our inputs.
    d_k = keys.shape[1]  # 2
    atten_weights_x2 = torch.softmax(attention_score_x2 / d_k**0.5, dim=-1)  # (6,)
    show_tensor_info(atten_weights_x2, 'nomalized attention x2')

    # Step-7(Final). context vector
    # Keep in mind that "values" is also one form of our inputs
    # So by applying such a matrix prodcut transformation, we get a new vector
    # This vector is still the embedding of our inputs, but in a view of x2
    context_vector_2 = torch.matmul(atten_weights_x2, values)  # (1,6)*(6,2) ==> (1,2)
    show_tensor_info(context_vector_2, "final x2's context vector")


def show_tensor_info(tensor_para: torch.Tensor, tensor_name: Optional[str] = None) -> None:
    from textwrap import dedent
    tensor_banner = f'=========={tensor_name}=============='
    banner_size = len(tensor_banner)
    banner_end = f'{banner_size*"="}'

    info: str = f"""
    {tensor_banner}
    Shape: {tensor_para.shape}
    Content:
        {tensor_para.data}
    {banner_end}
    """
    print(dedent(info).lstrip())


if __name__ == '__main__':
    inspect_self_attention_weights_naive()
