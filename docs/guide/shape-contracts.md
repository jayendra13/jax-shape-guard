# Shape Contracts

Shape contracts declare the shapes your functions expect and produce. ShapeGuard validates them at runtime and gives clear error messages when they don't hold.

## Symbolic dimensions with `Dim`

A `Dim` is a named placeholder for a dimension size. The same `Dim` object used in multiple positions means those dimensions must match:

```python
from shapeguard import Dim, expects

n = Dim("n")
m = Dim("m")

@expects(x=(n, m), y=(m,))
def forward(x, y):
    return x @ y
```

Here `m` appears in both `x.shape[1]` and `y.shape[0]`. ShapeGuard binds `m` to the actual value from `x` and verifies it matches in `y`. This process is called [unification](../concepts/unification.md).

!!! info "Identity-based equality"
    Two `Dim` objects unify only if they are the **same object** (same Python identity). Creating `Dim("n")` twice gives two independent dimensions — use a single variable to share.

## Integer dimensions

Use plain integers for fixed dimension sizes:

```python
@expects(x=(n, 128))
def embed(x):
    ...
```

This requires `x.shape[1]` to be exactly 128.

## Wildcard dimensions

Use `None` to accept any value at a position:

```python
@expects(x=(n, None))
def process(x):
    # x must be 2D, but second dimension can be anything
    ...
```

## Input contracts with `@expects`

`@expects` validates argument shapes on function entry:

```python
@expects(a=(n, m), b=(m, k))
def matmul(a, b):
    return a @ b
```

- Only arguments named in the decorator are checked
- Non-array arguments are silently skipped
- Works with both positional and keyword arguments

## Output contracts with `@ensures`

`@ensures` validates the return value shape:

```python
@expects(a=(n, m), b=(m, k))
@ensures(result=(n, k))
def matmul(a, b):
    return a @ b
```

!!! warning "Stacking order matters"
    `@expects` must be on **top** (outermost). This lets input bindings (like `n` and `k`) carry through to output validation via a shared `UnificationContext`.

### Tuple outputs

For functions that return multiple arrays:

```python
@ensures(result=((n, m), (n,)))
def decompose(x):
    return x, x.sum(axis=1)
```

### Dict outputs

For functions that return dictionaries:

```python
@ensures(result={"logits": (n, vocab), "hidden": (n, d)})
def model(x):
    ...
```

## Combined contracts with `@contract`

`@contract` validates both inputs and outputs in a single decorator:

```python
from shapeguard import contract

@contract(inputs={"a": (n, m), "b": (m, k)}, output=(n, k))
def matmul(a, b):
    return a @ b
```

This is equivalent to stacking `@expects` and `@ensures`, but more concise.

## PyTree specs

Shape specs can be nested dicts to match PyTree-structured inputs:

```python
@expects(
    params={"weights": (n, m), "bias": (m,)},
    x=(B, n)
)
def apply(params, x):
    return x @ params["weights"] + params["bias"]
```

ShapeGuard recursively validates each leaf array in the dict.

## JIT mode control

All decorators accept a `jit_mode` parameter to control behavior under JAX JIT tracing:

```python
@expects(x=(n, m), jit_mode="skip")
@jax.jit
def fast_layer(x):
    ...
```

See [JIT Modes](../concepts/jit-modes.md) for details.
