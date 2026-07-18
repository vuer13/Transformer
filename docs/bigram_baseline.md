# Bigram Baseline Model

## Objective
This model is a simple test model that predicts the next token using only the current token. It does not use attention yet.

### Input Shape
`get_batch()` gives us:
```
x shape = (batch, time)
y shape = (batch, time)
```

### Outputs
The model turns every token ID into prediction scores over the whole vocabulary
```
x shape = (batch, time)
logits shape = (batch, time, vocab_size)
```

### Flattening loss
PyTorch cross-entropy wants:
```
logits shape  = (number_of_predictions, vocab_size)
targets shape = (number_of_predictions)
```

## Main Idea
Bigram checks that this works:
```
data -> model -> logits -> loss -> backward -> optimizer step
```
