# Feed Forward Network

## Objective

Implements feed-forward network used inside a Transformer block

After attention lets tokens use context from other tokens, the feed-forward network processes each token individually.

Goal:

```text
context-aware token vector -> process it -> improved token representation
```

## Why feed-forward

Multi-head attention lets tokens communitcate with each other, but each token still needs to process its own updated information.

Feed-forward network allows each token to think for itself. Attention allows tokens to talk to each other.

The feed-forward network does not mix information between different tokens. It applies the same transformation to each token independently.

## Input Shape 

The feed-forward network takes token embeddings as input:

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

## How It Works

The feed-forward network usually has this structure:

```text
n_embd -> 4 * n_embd -> n_embd
```

This means the token vector is first expanded to a larger size, then projected back down to the original size.

In code, this looks like:

```text
Linear(n_embd, 4 * n_embd)
GELU()
Linear(4 * n_embd, n_embd)
Dropout()
```

The first linear layer expands the token vector. The GELU activation adds non-linearity, which helps the model learn more complex patterns. The second linear layer projects the vector back to the original embedding size. Dropout helps regularize the model during training.

## Why Expand Then Shrink

The feed-forward network temporarily gives each token more space to process information.

For example, if:

```text
n_embd = 32
```

then the feed-forward network does:

```text
32 -> 128 -> 32
```

The middle dimension is larger, which lets the model learn richer transformations.

The final output returns to `n_embd` so the shape stays compatible with the rest of the Transformer block.

## Output

The feed-forward network preserves the original shape:

```text
input shape  = (batch, time, n_embd)
output shape = (batch, time, n_embd)
```
## Main Idea

The feed-forward network does this:

```text
token vectors
-> expand each token vector
-> apply activation
-> project back down
-> output improved token vectors
```

Attention mixes information across tokens.

Feed-forward processes each token individually after that context has been added.

## Example

Suppose:

```text
batch = 2
time = 8
n_embd = 32
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

The first linear layer expands each token vector:

```text
(2, 8, 32) -> (2, 8, 128)
```

The last dimension becomes `128` because:

```text
4 * n_embd = 4 * 32 = 128
```

Then GELU keeps the same shape:

```text
(2, 8, 128) -> (2, 8, 128)
```

Then the second linear layer projects back down:

```text
(2, 8, 128) -> (2, 8, 32)
```

Dropout also keeps the same shape:

```text
(2, 8, 32) -> (2, 8, 32)
```

So the full feed-forward network turns:

```text
input:  (2, 8, 32)
output: (2, 8, 32)
```

The shape stays the same, but each token vector has been transformed and improved.
