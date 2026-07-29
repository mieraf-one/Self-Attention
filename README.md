# Understanding the Transformer: Self-Attention and Encoder Architecture

## Overview

The Transformer is a deep learning architecture introduced in the paper **"Attention Is All You Need"** (Vaswani et al., 2017).

The main idea behind the Transformer is replacing sequential processing used by RNNs with an attention-based mechanism that allows every token to directly communicate with every other token.

Unlike RNNs, which process words one by one, the Transformer processes tokens in parallel and learns relationships between them using **self-attention**.

The Transformer is based on an **Encoder-Decoder architecture**:

- The **Encoder** converts an input sequence into meaningful numerical representations.
- The **Decoder** uses these representations to generate an output sequence.

The main components of the Transformer are:

- Multi-Head Self-Attention
- Feed-Forward Networks
- Residual Connections
- Layer Normalization
- Positional Encoding


---

# 1. Input Representation

The Transformer cannot understand raw text directly. Text must first be converted into numerical representations.

The input processing pipeline is:

```
Text
 |
Tokenizer
 |
Token IDs
 |
Token Embedding
 |
Positional Encoding
 |
Transformer Encoder
```

---

# 2. Token Embedding

## Purpose

Token embedding converts token IDs into dense vectors that the Transformer can process.

A tokenizer first converts words into numbers:

Example:

```
"cat is cute"

↓

[1523, 45, 893]
```

These numbers are token IDs.

The embedding layer converts each token ID into a vector:

```
Token ID
   |
Embedding Layer
   |
Vector Representation
```

The embedding layer is a learned matrix:

```
Vocabulary Size × Embedding Dimension
```

Example:

```
Vocabulary size = 30,000 tokens

Embedding dimension = 512

Embedding matrix:

30000 × 512
```

Each token becomes a 512-dimensional vector.

During training, these values are learned, allowing the model to represent relationships between words.

For example:

```
king → vector
queen → vector

cat → vector
dog → vector
```

Words with similar meanings learn similar representations.

The original Transformer scales embeddings by:

\[
\sqrt{d_{model}}
\]

This keeps the embedding values at a suitable scale during training.

---

# 3. Positional Encoding

## Why is it needed?

RNNs naturally understand word order because they process tokens step-by-step.

The Transformer processes all tokens at the same time, so it needs additional information about token positions.

Example:

```
Dog bites man

Man bites dog
```

The same words are used, but the meaning changes because the order changes.

Positional encoding provides this order information.

The Transformer adds positional encoding to embeddings:

\[
Input = TokenEmbedding + PositionalEncoding
\]

The paper uses sine and cosine functions to create position information:

\[
PE(pos,2i)=sin(pos/10000^{2i/d_{model}})
\]

\[
PE(pos,2i+1)=cos(pos/10000^{2i/d_{model}})
\]


The final representation contains:

- Word meaning
- Word position


---

# 4. Self-Attention Mechanism

Self-attention is the main idea of the Transformer.

It allows every token to look at other tokens and decide which ones are important.

Example:

Sentence:

```
The animal didn't cross the road because it was tired.
```

The word:

```
it
```

needs to understand that it refers to:

```
animal
```

Attention helps the model create this relationship.

---

## Query, Key, and Value

Each token embedding is transformed into three vectors:

```
Embedding

   |
   |---- Query (Q)
   |
   |---- Key (K)
   |
   |---- Value (V)
```

These vectors are created using learned linear layers.

### Query

"What information am I looking for?"

### Key

"What information do I contain?"

### Value

"What information should I provide?"

---

## Attention Calculation

The Transformer uses:

\[
Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V
\]


Steps:

1. Compare Query with all Keys.

2. Calculate attention scores.

3. Divide by:

\[
\sqrt{d_k}
\]

to prevent very large values.

4. Apply softmax.

5. Multiply attention weights with Values.

The output becomes a new representation containing information from other tokens.


---

# 5. Multi-Head Self-Attention

Instead of using only one attention operation, the Transformer uses multiple attention heads.

Example:

```
Input

 |
 |---- Attention Head 1
 |
 |---- Attention Head 2
 |
 |---- Attention Head 3
 |
 |---- Attention Head 4

        ...

 |
Combine outputs
 |
Linear Layer
```

Each head can learn different relationships.

Examples:

Head 1:

```
Grammar relationships
```

Head 2:

```
Word meaning
```

Head 3:

```
Long-distance dependencies
```

The outputs of all heads are concatenated and passed through a linear layer.

In implementation:

```python
self.Wq
self.Wk
self.Wv
```

create:

```
Query
Key
Value
```

The process:

```
Multiple Heads

      ↓

Concatenate

      ↓

Linear Layer
```

---

# 6. Feed-Forward Network (FFN)

After attention, every token goes through a Feed-Forward Network.

The FFN works on each token independently.

Structure:

```
Input

 |
Linear Layer

 |
ReLU Activation

 |
Linear Layer

 |
Output
```

The purpose:

- Attention mixes information between tokens.
- FFN improves each token representation individually.


The equation from the paper:

\[
FFN(x)=max(0,xW_1+b_1)W_2+b_2
\]


---

# 7. Add & Norm

Every Transformer sub-layer uses:

1. Residual connection
2. Layer normalization


The operation:

\[
LayerNorm(x + Sublayer(x))
\]


## Residual Connection

Allows information from previous layers to flow directly.

Example:

```
Input
 |
+
 |
Attention Output
 |
Next Layer
```


## Layer Normalization

Keeps values stable during training.

Implementation:

```python
x = self.norm1(x + attention_output)

x = self.norm2(x + ffn_output)
```

---

# 8. Transformer Encoder Block

A single Transformer encoder layer contains:

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


The original Transformer stacks:

```
6 Encoder Blocks
```

Each layer creates better and more meaningful representations.


---

# 9. Encoder Output

The encoder does not directly generate words.

The output is a sequence of vectors:

\[
z=(z_1,z_2,...,z_n)
\]


Each vector contains contextual information about the input.

Example:

Input:

```
"The movie was amazing"
```

The encoder creates representations where:

```
movie vector
+
amazing relationship
+
sentence context
```

are stored together.

---

# 10. Using Encoder for Classification

For classification tasks, we take a representation from the encoder.

Example:

```
Encoder Output

      |

CLS Token Vector

      |

Linear Layer

      |

Logits

      |

Softmax

      |

Prediction
```


The classifier is usually:

```python
nn.Linear()
```


Example:

```
Output:

[2.5, -1.3]
```

After softmax:

```
Positive probability: 0.95

Negative probability: 0.05
```

Prediction:

```
Positive
```

---

# 11. Summary

The Transformer works through several stages:

## 1. Embedding

Converts token IDs into vectors.

## 2. Positional Encoding

Adds information about word order.

## 3. Self-Attention

Allows tokens to communicate with each other.

## 4. Multi-Head Attention

Allows the model to learn different relationships.

## 5. Feed-Forward Network

Improves token features.

## 6. Add & Norm

Stabilizes training.

## 7. Encoder Stack

Creates powerful contextual representations.

## 8. Classification Head

Converts encoder vectors into predictions.

---

# Final Idea

The Transformer changed sequence modeling by replacing step-by-step recurrence with attention.

Its ability to understand relationships between all tokens made it the foundation for modern language models such as:

- BERT
- GPT
- Large Language Models (LLMs)

The key idea is:

> Instead of reading words one by one, the Transformer allows every word to look at every other word and learn which relationships matter.