import torch


def inspect_navie_self_attention_single() -> None:
    """
    A simple self attention example. Calculating self attention for one query.

    Example text:
    "Your journey starts with one step."
    """
    # 1. A pseudo embeddings for text
    inputs = torch.tensor(
      [[0.43, 0.15, 0.89], # Your     (x^1)
       [0.55, 0.87, 0.66], # journey  (x^2)
       [0.57, 0.85, 0.64], # starts   (x^3)
       [0.22, 0.58, 0.33], # with     (x^4)
       [0.77, 0.25, 0.10], # one      (x^5)
       [0.05, 0.80, 0.55]] # step     (x^6)
    )
    print(f'\ninputs:\n{inputs}\n')

    # 2. Taking the second embedding("journey") as the example query
    query = inputs[1]
    print(f"inputs' shape: {inputs.shape}")
    print(f"query's shape: {query.shape}\n")
    # using the query to perform dot product with every embedding in inputs
    # thus you get attention score between every embedding and query
    attention_score_x2 = torch.matmul(inputs, query)

    # 3. normalize our attension_score, to make the sum of its elemnts to 1.
    # such normalization will imporve our training's performance.
    # here we apply softmax to do the normalization.
    normalized_attention_score_x2 = torch.softmax(attention_score_x2, dim=0)
    print(f'normalized_attention_score_x2 attention score:\n{normalized_attention_score_x2}')
    print(f'sum = {normalized_attention_score_x2.sum()}')

    # 4. compute context vector z2
    print(f"inputs' shape: {inputs.shape}")
    print(f"normalized_attention_score's shape: {normalized_attention_score_x2.shape}")
    context_vector = torch.matmul(normalized_attention_score_x2, inputs)
    print(context_vector)


if __name__ == '__main__':
    inspect_navie_self_attention_single()

