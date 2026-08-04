# Transformer Block and Language Model

## Objective

The Transformer block combines the main components built:

```text
LayerNorm
Multi-head attention
Feed-forward network
Residual connections
```

The language model then stacks multiple Transformer blocks together and adds the pieces needed for next-token prediction.

The goal is:

```text
token IDs -> embeddings -> Transformer blocks -> logits over vocabulary
```

## Transformer Block

A Transformer block is one reusable unit of the model.

It combines:

```text
attention = tokens communicate with each other
feed-forward = each token processes itself
LayerNorm = keeps values stable
residual connections = preserve original information
```

The block structure is:

```text
x = x + attention(layer_norm(x))
x = x + feed_forward(layer_norm(x))
```

This is called a pre-norm structure because LayerNorm happens before attention and before the feed-forward network.

## Why Transformer Blocks

A single attention layer is useful, but a full model needs multiple layers of processing.

Each Transformer block lets the model update the token representations.

Stacking many blocks allows the model to build more complex understanding over time.

A simple way to think about it:

```text
one block = one round of communication and processing
many blocks = many rounds of communication and processing
```

## Transformer Block Input and Output

The Transformer block takes token vectors as input:

```text
x shape = (batch, time, n_embd)
```

The output has the same shape:

```text
output shape = (batch, time, n_embd)
```

The shape stays the same because residual connections require the original input and transformed output to match.

## Transformer Block Flow

The first part of the block is attention:

```text
x -> LayerNorm -> Multi-Head Attention -> add original x back
```

This lets tokens gather context from previous tokens.

The second part is feed-forward:

```text
x -> LayerNorm -> Feed-Forward Network -> add original x back
```

This lets each token process its own updated information.

So the full block does:

```text
input token vectors
-> normalize
-> attention
-> residual add
-> normalize
-> feed-forward
-> residual add
-> output token vectors
```

## Language Model

The language model connects everything together into a full decoder-only language model.

It takes token IDs as input:

```text
idx shape = (batch, time)
```

Then it produces logits:

```text
logits shape = (batch, time, vocab_size)
```

The logits are prediction scores over the vocabulary. Each position predicts the next token.

## Token Embeddings

The model starts with token IDs.

Token IDs are just integers, so the model first converts them into vectors using a token embedding table.

```text
token IDs -> token embeddings
```

Shape:

```text
idx shape = (batch, time)
token embeddings shape = (batch, time, n_embd)
```

Each token now has a vector representation.

## Position Embeddings

A Transformer does not automatically know token order.

So the model also adds position embeddings.

Position embeddings tell the model where each token is in the sequence.

```text
token embeddings + position embeddings
```

The result still has shape:

```text
(batch, time, n_embd)
```

Now each token vector contains information about:

```text
what token it is
where it appears in the sequence
```

## Stacked Transformer Blocks

After token and position embeddings are added, the result is passed through multiple Transformer blocks.

```text
x -> block 1 -> block 2 -> block 3 -> ...
```

Each block keeps the same shape:

```text
(batch, time, n_embd)
```

The token vectors become more context-aware after each block.

## Final LayerNorm

After the stacked Transformer blocks, the model applies one final LayerNorm.

This helps stabilize the final token representations before prediction.

Shape stays the same:

```text
(batch, time, n_embd)
```

## Language Modeling Head

The final step is the language modeling head.

This is a linear layer that turns each token vector into scores over the vocabulary.

```text
n_embd -> vocab_size
```

Shape:

```text
(batch, time, n_embd) -> (batch, time, vocab_size)
```

These scores are called logits.

The model uses these logits to predict the next token at every position.

## Loss

If targets are provided, the model calculates cross-entropy loss.

The logits are flattened from:

```text
(batch, time, vocab_size)
```

into:

```text
(batch * time, vocab_size)
```

The targets are flattened from:

```text
(batch, time)
```

into:

```text
(batch * time)
```

This allows PyTorch cross-entropy to compare each prediction with the correct next token.

## Main Idea

The Transformer block does this:

```text
token vectors
-> attention with residual
-> feed-forward with residual
-> updated token vectors
```

The full model does this:

```text
token IDs
-> token embeddings
-> position embeddings
-> stacked Transformer blocks
-> final LayerNorm
-> language modeling head
-> logits
```

The Transformer block is the reusable processing unit.

The language model is the full system that stacks those blocks and predicts the next token.

## Example

Suppose:

```text
batch = 2
time = 8
n_embd = 32
vocab_size = 50257
n_layer = 2
```

The input token IDs have shape:

```text
idx shape = (2, 8)
```

The token embedding table turns them into vectors:

```text
token embeddings shape = (2, 8, 32)
```

Position embeddings are added:

```text
position embeddings shape = (8, 32)
combined shape = (2, 8, 32)
```

Then the model passes the vectors through Transformer blocks:

```text
block 1 input  = (2, 8, 32)
block 1 output = (2, 8, 32)

block 2 input  = (2, 8, 32)
block 2 output = (2, 8, 32)
```

The final LayerNorm keeps the same shape:

```text
final LayerNorm output = (2, 8, 32)
```

The language modeling head projects each token vector to vocabulary scores:

```text
logits shape = (2, 8, 50257)
```

This means:

```text
2 examples
8 token positions per example
50257 possible next-token scores per position
```

So the full model turns:

```text
input:  (2, 8)
output: (2, 8, 50257)
```

The input is token IDs.

The output is next-token prediction scores.