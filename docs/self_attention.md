# Self Attention

## Masked Self-Attention

This is the first real transformer component. 
It does the following:
- creates K, Q, V
- computes attention weights
- applies casual mask, softmax, dropout
- multiplies by values

### Inputs
Attention does not take raw token IDs directly
When attention runs, tokens have already become embeddings
```
x shape = (batch, time, n_embd)
```

### K, Q, V Shapes
```
K = keys
Q = queries
V = values
```
Each one goes from `n_embd -> head_size`

### Attention Score Shape
```
Q @ K&T
```

Resulting scores shape: `(batch, time, time)`
ie. For each example, each token compares itself to every other token