"""
My code for inspection coressponding to ch03.
Some matrix operation in original file is implemented with for-loop, easier to understand, but less effective, which is good for learning.
Here mine is implemented with a more vecotrization way, this is faster in speed, but may make it harder to understand.
"""
import torch


def inspect_navie_self_attention_single() -> None:
    """
    Corresponding to Chapter 3.3.1.

    A simple self attention example. Calculating self attention for one query.

    Example text:
    "Your journey starts with one step."
    """
    # 1. A pseudo embeddings for text
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your     (x^1)
         [0.55, 0.87, 0.66],  # journey  (x^2)
         [0.57, 0.85, 0.64],  # starts   (x^3)
         [0.22, 0.58, 0.33],  # with     (x^4)
         [0.77, 0.25, 0.10],  # one      (x^5)
         [0.05, 0.80, 0.55]]  # step     (x^6)
    )
    print(f'\n===>S1-inputs:\n{inputs}\n')

    # 2. Taking the second embedding("journey") as the example query
    query = inputs[1]
    print(f"inputs' shape: {inputs.shape}")
    print(f"query's shape: {query.shape}\n")
    """
    using the query to perform dot product with every embedding in inputs.
    thus you get attention score between every embedding and query
    by applying dot product, the relevance of "journey" with other words in sequence can be known.
    bigger value means bigger relevance.
    """
    attention_score_x2 = torch.matmul(inputs, query)
    print(f'===>S2-attention score:\n{attention_score_x2}\n')
    print(f"attention's shape: {attention_score_x2.shape}\n")

    # 3. normalize our attension_score, to make the sum of its elemnts to 1.
    """
    such normalization will imporve our training's performance.
    here we apply softmax to do the normalization.
    as been mentioned in the second part, the results of dot product means the relevance,
    by normalize their value, this can also be called as "weight" in a way,
    the weight of every word towards x2, including x2 self.
    so in the result, you will find the second value is the biggest, as of course,
    what is most related to x2 is x2 itself.
    """
    weights_to_x2 = torch.softmax(attention_score_x2, dim=0)
    print(f'===>S3-weights_to_x2 :\n{weights_to_x2}')
    print(f"normalized attention's shape: {weights_to_x2.shape}")
    print(f'sum = {weights_to_x2.sum()}')

    # 4. compute context vector z2
    print(f"inputs' shape: {inputs.shape}")
    print(f"normalized_attention_score's shape: {weights_to_x2.shape}\n")
    """
    after getting the weights from step3, we now multiply every weight to its corresponding input,
    as the following:
        y1 = w1 * x1 = (w1*a1, w1*b1, w1*c1)
        y2 = w2 * x2 = (w2*a2, w2*b2, w2*c2)
        ...
        y6 = w6 * x6 = (w6*a6, w6*b6, w6*c6)
        new weighted input = {y1, y2, y3, y4, y5, y6}^T

    for now, this is like we've got a weighted(or scaled) inputs in the perspective of x2.
    then, we add those inputs by row, thus we get the context vector z2:
        z21 = w1*a1 + w2*a2 + ... + w6*a6
        z22 = w1*b1 + w2*b2 + ... + w6*b6
        z23 = w1*b1 + w2*c2 + ... + w6*c6
        z2 = {z21, z22, z23}

    so here why do we add the weighted inputs by row? 
    this is beacause we need to get a "summary" of query(x2 here), this sum is including how every other words
    contributes thier relevance, for a sum, the bigger relevance word will surely contribute more for this value.
    and this final context vector is a new representaiton for x2, which is granted with more info like position,
    other words' relations with it.
    """
    context_vector = torch.matmul(weights_to_x2, inputs)
    print(f'===>S4-context_vector:\n{context_vector}\n')
    print(f"context_vector's shape: {context_vector.shape}\n")


def inspect_naive_self_attention() -> None:
    """
    Corresponding to 3.3.2

    A simple self attention example. Calculating self attention for all inputs.

    Example text:
    "Your journey starts with one step."
    """
    # 1. A pseudo embeddings for text
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your     (x^1)
         [0.55, 0.87, 0.66],  # journey  (x^2)
         [0.57, 0.85, 0.64],  # starts   (x^3)
         [0.22, 0.58, 0.33],  # with     (x^4)
         [0.77, 0.25, 0.10],  # one      (x^5)
         [0.05, 0.80, 0.55]]  # step     (x^6)
    )
    print(f'\n===>S1-inputs:\n{inputs}\n')

    # 2. Attention score calculation
    """
    Now the query become inputs it self,
    as we need to compute the attention score of every word towards other words in sequence(including itself, of course)
    """
    # Here I explictly assign a new parameter to represent it.
    # To get the matrix product, need to make it transposed.
    query = inputs.T
    print(f'\n===>S2-query:\n{query}\n')
    atten_scores = torch.matmul(inputs, query) 
    print(f'\n===>S2-attention_scores:\n{atten_scores}\n')

    # 3. Normalization
    weights = torch.softmax(atten_scores, dim=-1)
    print(f'\n===>S3-normalized attention scores:\n{weights}\n')
    # validation
    for weight in weights:
        print(weight.sum())

    # 4. Compute context vector of each word
    context_vectors = torch.matmul(weights, inputs)
    print(f'\n===>S4-final context vectors:\n{context_vectors}\n')


if __name__ == '__main__':
    # inspect_navie_self_attention_single()
    inspect_naive_self_attention()
