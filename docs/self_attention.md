# Masked Self-Attention Head

## Objective

Implements one masked self-attention head.

Self-attention lets each token use information from other tokens in the same sequence. In a GPT-style model, each token is only allowed to look at itself and previous tokens.

It cannot look at future tokens.

The goal is:

```text
current token -> look back at previous tokens -> build a better token representation
```

## Why Self-Attention

The bigram model only used the current token to predict the next token.

That means it could learn patterns like:

```text
"New" -> "York"
```

but it could not properly use longer context.

Self-attention fixes this by allowing each token to decide which earlier tokens are useful.

## Input Shape

Self-attention does not take raw token IDs directly.

It takes token embeddings:

```text
x shape = (batch, time, n_embd)
```

Where:

```text
batch = number of examples
time = number of tokens in each example
n_embd = size of each token vector
```

Each token is represented by a vector of numbers.

## Query, Key, and Value

The attention head creates three versions of each token vector:

```text
query
key
value
```

A simple way to think about them:

```text
query = what this token is looking for
key   = what this token contains
value = the information this token can pass forward
```

Each token compares its query with the keys of other tokens.

This tells the model which tokens are relevant.

## Attention Scores

The model compares queries and keys using matrix multiplication:

```text
scores = query @ key_transpose
```

This creates a score for how much each token should pay attention to every other token.

The score matrix has shape:

```text
(batch, time, time)
```

This means every token has a score against every other token in the sequence.

## Scaling

The attention scores are scaled down by:

```text
sqrt(head_size)
```

This helps keep the scores stable.

Without scaling, the scores can become very large, which can make softmax too extreme.

## Causal Mask

Because this is a GPT-style model, the model should not look into the future.

For example:

```text
token 0 can see token 0
token 1 can see token 0 and token 1
token 2 can see token 0, token 1, and token 2
token 3 can see token 0, token 1, token 2, and token 3
```

The model should not allow this:

```text
token 1 looking at token 2
token 1 looking at token 3
```

The causal mask blocks future positions.

This prevents the model from cheating during training.

## Softmax

After masking, softmax turns the attention scores into probabilities. These probabilities are called attention weights.

The attention weights decide how much information to take from each previous token.

## Output

The attention weights are multiplied by the value vectors:

```text
output = attention_weights @ value
```

This creates a new vector for each token.

Each output token vector now contains information from the tokens it was allowed to attend to.

For one attention head, the output shape is:

```text
(batch, time, head_size)
```

This is smaller than the original embedding size because one attention head only handles part of the embedding.

Multi-head attention will combine multiple heads in the next commit.

## Main Idea

Masked self-attention does this:

```text
token embeddings
-> create query, key, value
-> compare query and key
-> mask future tokens
-> softmax into attention weights
-> mix value vectors
-> output context-aware token vectors
```

Self-attention is what allows the model to use context.

The mask is what makes it safe for next-token prediction.

## Example

Suppose:

```text
batch = 2
time = 8
n_embd = 32
n_head = 4
head_size = 8
```

The input is:

```text
x shape = (2, 8, 32)
```

This means:

```text
2 examples
8 tokens per example
32 numbers per token
```

The attention head creates:

```text
query shape = (2, 8, 8)
key shape   = (2, 8, 8)
value shape = (2, 8, 8)
```

The last dimension is `8` because:

```text
head_size = n_embd / n_head
head_size = 32 / 4
head_size = 8
```

Then the model calculates attention scores:

```text
scores shape = (2, 8, 8)
```

This means each of the 8 tokens gets a score against each of the 8 tokens.

After applying the causal mask and softmax, the attention weights are multiplied by the value vectors.

The final output is:

```text
output shape = (2, 8, 8)
```

So one masked self-attention head turns:

```text
input:  (2, 8, 32)
output: (2, 8, 8)
```