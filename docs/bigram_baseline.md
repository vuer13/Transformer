# Bigram Baseline Model

## Objective

This model is a simple test model that predicts the next token using only the current token. It does not use attention yet.

The purpose of this model is to check that the basic training loop works before building the Transformer.

The model should confirm that this pipeline works:

```text
data -> model -> logits -> loss -> backward -> optimizer step
```

### Input Shape
`get_batch()` gives us:
```
x shape = (batch, time)
y shape = (batch, time)
```

`x` contains the input token IDs.

`y` contains the target token IDs, which are shifted one position forward.

So the model is learning:

```text
current token -> next token
```

### Outputs

The model turns every token ID into prediction scores over the whole vocabulary.

```
x shape = (batch, time)
logits shape = (batch, time, vocab_size)
```

Each position in `logits` contains one score for every possible next token.

The model does not directly output the next token. It outputs scores, and the highest-scoring tokens are the model's best guesses.

### Flattening loss

PyTorch cross-entropy wants:

```
logits shape  = (number_of_predictions, vocab_size)
targets shape = (number_of_predictions)
```

But our model gives:

```text
logits shape = (batch, time, vocab_size)
targets shape = (batch, time)
```

So we flatten the `batch` and `time` dimensions together before calculating loss.

This turns:

```text
(batch, time, vocab_size)
```

into:

```text
(batch * time, vocab_size)
```

and:

```text
(batch, time)
```

into:

```text
(batch * time)
```

## Main Idea
Bigram checks that this works:
```
data -> model -> logits -> loss -> backward -> optimizer step
```

If this works, then the data pipeline, model output, loss calculation, gradients, and optimizer are all connected correctly.

The bigram model is intentionally simple. It only uses the current token to predict the next token. It does not understand longer context.

## Example

Suppose:

```text
batch = 2
time = 4
vocab_size = 10
```

Then `get_batch()` might give:

```text
x =
[
  [10, 20, 30, 40],
  [50, 60, 70, 80]
]

y =
[
  [20, 30, 40, 50],
  [60, 70, 80, 90]
]
```

So the shapes are:

```text
x shape = (2, 4)
y shape = (2, 4)
```

The model outputs prediction scores:

```text
logits shape = (2, 4, 10)
```

This means:

```text
2 examples
4 token positions per example
10 possible next-token scores per position
```

Before calculating loss, we flatten:

```text
logits shape = (2, 4, 10)
targets shape = (2, 4)
```

into:

```text
logits shape = (8, 10)
targets shape = (8)
```

because:

```text
2 * 4 = 8 total predictions
```

So each row of `logits` is one next-token prediction, and each value in `targets` is the correct next token.
