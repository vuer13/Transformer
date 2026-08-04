# LayerNorm and Residual Connections

## Objective

LayerNorm helps keep token vectors stable during training.

Residual connections help preserve the original information by adding the input back after a transformation.

The goal is:

```text
normalize input -> transform input -> add original input back
```

## Why LayerNorm

As data moves through a neural network, the numbers inside the token vectors can become too large, too small, or unstable. LayerNorm helps by normalizing each token vector.

```text
LayerNorm = keeps each token vector in a stable range
```

It makes training more stable and helps the model learn better.

## Input Shape

LayerNorm takes token embeddings as input:

```text
x shape = (batch, time, n_embd)
```

Where:

```text
batch = number of examples
time = number of tokens in each example
n_embd = size of each token vector
```

LayerNorm does not change the shape.

```text
input shape  = (batch, time, n_embd)
output shape = (batch, time, n_embd)
```

## How LayerNorm Works

LayerNorm normalizes each token vector across the embedding dimension.

For each token, it calculates:

```text
mean
variance
```

Then it uses those values to normalize the token vector.

Conceptually:

```text
token vector -> normalize values -> stable token vector
```

For example, if one token has:

```text
32 numbers
```

LayerNorm normalizes those 32 numbers for that token. It does this independently for every token in every example.

## Learnable Weight and Bias

After normalization, LayerNorm uses learnable parameters:

```text
weight
bias
```

The `weight` lets the model scale the normalized values.

The `bias` lets the model shift the normalized values.

This means LayerNorm stabilizes the values, but still lets the model learn the best scale and shift.

## Why Residual Connections

A residual connection adds the original input back after a transformation.

Instead of only doing:

```text
output = transformation(x)
```

we do:

```text
output = x + transformation(x)
```

This helps because the model does not have to completely rewrite the token representation at every layer.

It can keep the original information and add new information on top.

A simple way to think about it:

```text
residual connection = keep the old information and add the new information
```

## Why Residuals Help

Residual connections help with two main things:

```text
preserving information
making training easier
```

Without residual connections, information can get changed too much as it passes through many layers.

With residual connections, the model can choose to make small updates instead of replacing everything.

## Transformer Residual Structure

In a Transformer block, residual connections are used around attention and feed-forward layers.

The structure is:

```text
x = x + attention(layer_norm(x))
x = x + feed_forward(layer_norm(x))
```

This is called a pre-norm structure because LayerNorm happens before the attention or feed-forward layer.

## Why Shapes Must Match

Residual connections use addition:

```text
x + transformed_x
```

For this to work, both tensors must have the same shape.

So if:

```text
x shape = (batch, time, n_embd)
```

then the transformed output must also be:

```text
transformed_x shape = (batch, time, n_embd)
```

This is why multi-head attention and feed-forward both output the same shape that they receive.

## Main Idea

LayerNorm and residual connections work together:

```text
LayerNorm = stabilize the token vector before transformation
Residual = add the original token vector back after transformation
```

Together, they help the Transformer train more reliably.

The pattern is:

```text
input
-> normalize
-> transform
-> add original input back
-> output
```

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

LayerNorm keeps the same shape:

```text
layer_norm(x) shape = (2, 8, 32)
```

Then attention or feed-forward also keeps the same shape:

```text
transformed_x shape = (2, 8, 32)
```

Now the residual connection can add them:

```text
x + transformed_x
```

Shape-wise:

```text
(2, 8, 32) + (2, 8, 32) = (2, 8, 32)
```

So the final output is:

```text
output shape = (2, 8, 32)
```

The shape stays the same, but the token vectors now contain both the original information and the new transformed information.
