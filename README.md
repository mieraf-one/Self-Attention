```markdown
# Report Paper: Self-Attention

# Main Idea of the Transformer

The Transformer is based on an encoder-decoder architecture.

The encoder's job is to convert an input sequence into meaningful representations.

The decoder uses these representations to generate an output sequence.

The main innovation is that both parts use:

- Multi-Head Self-Attention
- Feed-Forward Networks
- Residual Connections
- Layer Normalization

Instead of processing words step-by-step like an RNN, the Transformer allows every token to directly communicate with other tokens through attention.


# Input Representation

The Transformer cannot understand words directly. The input text must first be converted into numbers.

The process is:

```

Text
↓
Tokenizer
↓
Token IDs
↓
Token Embedding
↓
Positional Encoding
↓
Transformer Encoder

```


# Token Embedding

The token IDs are converted into vectors using an embedding layer.

The embedding layer is a learned matrix:

```

Vocabulary size × Embedding dimension

```

For example, each token may become a 512-dimensional vector.

The embedding values are learned during training, allowing the model to represent relationships between words.

The original Transformer scales embeddings by:

\[
\sqrt{d_{model}}
\]

This helps keep the embedding values at a suitable scale during training.


# Positional Encoding

The Transformer does not process tokens in order like an RNN. Therefore, it needs information about token position.

The paper introduces positional encoding using sine and cosine functions.

The positional encoding is added to the token embeddings:

\[
Input = TokenEmbedding + PositionalEncoding
\]

This allows the model to understand the difference between:

```

"dog bites man"

```

and

```

"man bites dog"

````

because word order changes the meaning.


# Self-Attention Mechanism

The main idea of the Transformer is attention.

Attention allows each token to decide which other tokens are important.

Each embedding is transformed into three vectors:

- Query (Q)
- Key (K)
- Value (V)

These are created using learned linear layers.

The attention calculation is:

\[
Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V
\]

The steps are:

1. Compare Query with all Keys.
2. Calculate attention scores.
3. Apply softmax to convert scores into probabilities.
4. Multiply probabilities with Values.

The output becomes a new representation containing information from other tokens.


# Multi-Head Self-Attention

Instead of using one attention operation, the Transformer uses multiple attention heads.

Each head learns different relationships.

For example:

- One head may focus on grammar relationships.
- Another head may focus on word meaning.
- Another head may focus on long-distance dependencies.

The outputs from all heads are combined and passed through another linear layer.

In the implementation:

```python
self.Wq
self.Wk
self.Wv
````

create Query, Key, and Value.

Then the attention outputs are combined:

```
Multiple heads
       ↓
Concatenate
       ↓
Linear layer
```

# Feed-Forward Network (FFN)

After attention, every token representation passes through a Feed-Forward Network.

The FFN contains:

* Linear layer
* Activation function (ReLU)
* Linear layer

The purpose of FFN is to transform the attention output into a more useful representation.

The attention layer mixes information between tokens.

The FFN processes each token individually and improves its features.

The equation from the paper:

[
FFN(x)=max(0,xW_1+b_1)W_2+b_2
]

# Add & Norm

Each Transformer sub-layer uses:

* Residual connection
* Layer Normalization

The operation is:

[
LayerNorm(x + Sublayer(x))
]

The residual connection allows information from previous layers to flow easily.

Layer normalization keeps values stable during training.

In the implementation:

```python
x = self.norm1(x + attention_output)

x = self.norm2(x + ffn_output)
```

# Transformer Encoder Block

A single encoder layer contains:

```
Input
 |
Multi-Head Self-Attention
 |
Add & Norm
 |
Feed-Forward Network
 |
Add & Norm
 |
Output
```

The paper stacks multiple encoder layers.

The original Transformer uses:

```
6 Encoder Blocks
```

Each block gradually creates better representations of the input.

# Encoder Output

The encoder does not directly produce words.

Its output is a sequence of vectors:

[
z=(z_1,z_2,...,z_n)
]

These vectors contain contextual information about the input.

In classification tasks, we can take one vector (such as the CLS token representation) and pass it to a classifier.

The classifier is usually:

```python
nn.Linear()
```

which converts the vector into class scores.

Example:

```
Encoder output vector
        |
     Linear Layer
        |
     Logits
        |
     Softmax
        |
     Class prediction
```

# Conclusion

The Transformer changed sequence modeling by replacing recurrent processing with attention.

Its main components work together:

* Embedding converts tokens into vectors.
* Positional Encoding adds order information.
* Self-Attention allows tokens to communicate.
* Multi-Head Attention learns different relationships.
* FFN improves token representations.
* Add & Norm stabilizes training.
* Stacked Encoder Blocks create powerful representations.

The Transformer became the foundation for modern models such as BERT, GPT, and many other large language models.

```
```
