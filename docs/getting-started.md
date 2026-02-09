# Getting Started

## Installation

=== "pip"

    ```bash
    pip install jax-shapeguard
    ```

=== "uv"

    ```bash
    uv add jax-shapeguard
    ```

For JAX support, install the optional JAX extra:

```bash
pip install jax-shapeguard[jax]
```

## Your first shape contract

Create symbolic dimensions with `Dim`, then use `@expects` to declare what shapes your function accepts:

```python
import numpy as np
from shapeguard import Dim, expects

n = Dim("n")
m = Dim("m")
k = Dim("k")

@expects(a=(n, m), b=(m, k))
def matmul(a, b):
    return a @ b

# Works — m=4 in both arguments
x = np.random.randn(3, 4)
y = np.random.randn(4, 5)
result = matmul(x, y)  # shape (3, 5)
```

If you pass mismatched shapes:

```python
y_bad = np.random.randn(5, 7)
matmul(x, y_bad)
# ShapeGuardError:
#   function: matmul
#   argument: b
#   expected: (m, k)
#   actual:   (5, 7)
#   reason:   dimension 'm' bound to 4 from a.shape[1], but got 5 from b.shape[0]
```

## Adding output contracts

Use `@ensures` to also validate the return value. Stack it *under* `@expects` so that dimension bindings carry through:

```python
from shapeguard import ensures

@expects(a=(n, m), b=(m, k))
@ensures(result=(n, k))
def matmul(a, b):
    return a @ b
```

Now if your function returns the wrong shape, you'll get an error at the return site rather than downstream.

## Using `check_shape` directly

For one-off checks outside of decorated functions, use `check_shape`:

```python
from shapeguard import check_shape, Dim

n = Dim("n")
ctx = check_shape(x, (n, 128), name="input")
check_shape(y, (n, 64), name="output", ctx=ctx)  # n must match
```

## Grouped checks with ShapeContext

When you need to validate multiple arrays share dimensions:

```python
from shapeguard import ShapeContext, Dim

n, m, k = Dim("n"), Dim("m"), Dim("k")

with ShapeContext() as ctx:
    ctx.check(x, (n, m), "x")
    ctx.check(y, (m, k), "y")
    ctx.check(z, (n, k), "z")
```

## What's next?

- [Shape Contracts](guide/shape-contracts.md) — wildcards, integer dims, PyTree specs
- [Batch & Ellipsis](guide/batch-and-ellipsis.md) — ML-friendly batch dimensions
- [Broadcasting](guide/broadcasting.md) — broadcast validation and debugging
- [ML Helpers](guide/ml-helpers.md) — pre-defined dims for transformers and CNNs
- [Unification](concepts/unification.md) — how dimension matching works under the hood
- [JIT Modes](concepts/jit-modes.md) — controlling validation under JAX JIT
