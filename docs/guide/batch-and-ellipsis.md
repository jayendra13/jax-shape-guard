# Batch Dimensions & Ellipsis

ML workflows commonly deal with variable batch sizes and functions that should work on inputs with any number of leading dimensions. ShapeGuard supports both patterns.

## Batch dimensions

`Batch` is a specialized `Dim` for batch sizes. It unifies across arguments within a single call, but each call can have a different batch size:

```python
from shapeguard import Batch, Dim, expects

B = Batch()
n = Dim("n")
m = Dim("m")

@expects(x=(B, n, m), y=(B, m))
def layer(x, y):
    return x @ y[..., None]

layer(x_32, y_32)  # B=32 for this call
layer(x_64, y_64)  # B=64 for this call — different, OK
```

`Batch` is just a `Dim` with a default name of `"batch"`. You can give it a custom name:

```python
B = Batch("B")
```

## Ellipsis for variable leading dims

Use `...` (Python's `Ellipsis`) in a shape spec to match any number of leading dimensions:

```python
@expects(x=(..., n, m))
def normalize(x):
    # Works with (n, m), (B, n, m), (B, T, n, m), etc.
    ...
```

The ellipsis matches zero or more dimensions at the front. The trailing dimensions after the ellipsis must match exactly.

### Rules

- At most **one** ellipsis per shape spec
- Ellipsis must be at the **start** of the spec
- Dimensions after the ellipsis are matched from the right

```python
@expects(x=(..., n, m))
def f(x):
    ...

f(np.zeros((3, 4)))        # ... matches nothing, n=3, m=4
f(np.zeros((2, 3, 4)))     # ... matches (2,), n=3, m=4
f(np.zeros((5, 2, 3, 4)))  # ... matches (5, 2), n=3, m=4
```

### Minimum rank

The spec `(..., n, m)` requires at least rank 2. Passing a 1D array will raise a `RankMismatchError`:

```python
f(np.zeros((4,)))
# RankMismatchError:
#   expected: (..., n, m)
#   actual:   (4,)
#   reason:   expected rank 2+, got rank 1
```

## Grouped checks with ShapeContext

`ShapeContext` provides imperative shape checking with a shared [unification](../concepts/unification.md) context. Use it when validating arrays outside a single decorated function:

```python
from shapeguard import ShapeContext, Dim

n, m, k = Dim("n"), Dim("m"), Dim("k")

with ShapeContext() as ctx:
    ctx.check(x, (n, m), "x")
    ctx.check(y, (m, k), "y")
    ctx.check(z, (n, k), "z")

# Inspect resolved bindings
print(ctx.bindings)  # {"n": 3, "m": 4, "k": 5}
```

### Method chaining

`check` returns `self`, so you can chain calls:

```python
ctx = ShapeContext()
ctx.check(x, (n, m), "x").check(y, (m, k), "y")
```

### Resolving dimensions

Look up the bound value for a specific `Dim`:

```python
ctx = ShapeContext()
ctx.check(x, (n, m), "x")
print(ctx.resolve(n))  # 3
print(ctx.resolve(k))  # None (not bound yet)
```

## Combining Batch with ellipsis

You can use `Batch` and ellipsis in the same spec:

```python
B = Batch()

@expects(x=(..., B, n, m))
def layer(x):
    # Accepts any leading dims, then batch, then (n, m)
    ...
```
