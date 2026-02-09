# ML Helpers

The `shapeguard.ml` module provides pre-defined dimensions and shape utilities for common ML patterns.

## Pre-defined dimensions

```python
from shapeguard.ml import B, T, C, H, W, D
```

| Dim | Name | Typical use |
|-----|------|-------------|
| `B` | `"B"` | Batch size (`Batch` instance) |
| `T` | `"T"` | Sequence length / time steps |
| `C` | `"C"` | Channels |
| `H` | `"H"` | Height |
| `W` | `"W"` | Width |
| `D` | `"D"` | Feature / embedding dimension |

Use them directly in `@expects`:

```python
from shapeguard import expects
from shapeguard.ml import B, T, D

@expects(x=(B, T, D))
def transformer_layer(x):
    ...
```

!!! note
    These are module-level singletons. If you need independent dimensions with the same name, create your own `Dim` or `Batch` instances.

## Attention shapes

`attention_shapes` returns shape specs for multi-head attention Q, K, V tensors:

```python
from shapeguard import Dim, expects
from shapeguard.ml import B, attention_shapes

heads = Dim("heads")
seq_q = Dim("seq_q")
seq_k = Dim("seq_k")
d_k = Dim("d_k")

@expects(**attention_shapes(B, heads, seq_q, seq_k, d_k))
def attention(q, k, v):
    # q: (B, heads, seq_q, d_k)
    # k: (B, heads, seq_k, d_k)
    # v: (B, heads, seq_k, d_k)
    scores = q @ k.swapaxes(-2, -1)
    ...
```

The returned dict has keys `"q"`, `"k"`, `"v"`:

```python
attention_shapes(B, heads, seq_q, seq_k, d_k)
# {
#     "q": (B, heads, seq_q, d_k),
#     "k": (B, heads, seq_k, d_k),
#     "v": (B, heads, seq_k, d_k),
# }
```

You can also pass integers for fixed dimensions:

```python
@expects(**attention_shapes(B, 8, seq_q, seq_k, 64))
def attention(q, k, v):
    # 8 heads, head dim 64
    ...
```

## Convolution output shape

`conv_output_shape` computes the output shape of a convolution:

```python
from shapeguard.ml import conv_output_shape

# Conv2d: 32 batch, 3 channels, 28x28 input, 3x3 kernel
conv_output_shape((32, 3, 28, 28), kernel=3)
# (32, 3, 26, 26)

# With stride and padding
conv_output_shape((32, 3, 28, 28), kernel=3, stride=2, padding=1)
# (32, 3, 14, 14)
```

The formula applied per spatial dimension is:

```
output = floor((input + 2*padding - kernel) / stride) + 1
```

### Parameters

- `input`: Shape as `(batch, channels, *spatial)` — at least 3 dimensions
- `kernel`: Kernel size — an int (broadcast to all spatial dims) or a tuple
- `stride`: Stride — an int or tuple (default 1)
- `padding`: Padding — an int or tuple (default 0)

### Per-dimension kernel/stride/padding

For non-square kernels, pass tuples:

```python
conv_output_shape((1, 3, 28, 28), kernel=(3, 5), stride=(1, 2), padding=(1, 2))
# (1, 3, 28, 14)
```

### Validation

Raises `ValueError` if the output spatial size would be non-positive:

```python
conv_output_shape((1, 1, 3, 3), kernel=5)
# ValueError: non-positive output size -1 at spatial dim 0: ...
```
