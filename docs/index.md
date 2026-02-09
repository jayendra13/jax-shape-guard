# ShapeGuard

**Runtime shape contracts and diagnostics for NumPy and JAX.**

ShapeGuard lets you declare the shapes your functions expect and ensures they hold at runtime. When shapes don't match, you get clear, actionable error messages that pinpoint exactly which dimension went wrong and where the conflict originated.

```python
from shapeguard import Dim, expects

n, m, k = Dim("n"), Dim("m"), Dim("k")

@expects(a=(n, m), b=(m, k))
def matmul(a, b):
    return a @ b
```

When shapes don't match:

```
ShapeGuardError:
  function: matmul
  argument: b
  expected: (m, k)
  actual:   (5, 7)
  reason:   dimension 'm' bound to 4 from a.shape[1], but got 5 from b.shape[0]
```

## Why ShapeGuard?

**Shape bugs are silent.** A matrix multiply with wrong dimensions doesn't crash immediately — it produces garbage results that propagate through your model. Traditional error messages like `shapes (3,4) and (5,3) not aligned` don't tell you *why* the shapes should match or *where* the mismatch originated.

ShapeGuard fixes this by:

- **Declaring intent** — shape specs document what your function expects
- **Catching errors early** — validation happens on function entry, before computation
- **Explaining failures** — error messages trace bindings back to their source
- **Working with JAX** — configurable behavior under JIT tracing

## Features

- **Symbolic dimensions** — `Dim("n")` unifies across arguments
- **Input contracts** — `@expects` validates arguments on entry
- **Output contracts** — `@ensures` validates return values
- **Combined contracts** — `@contract` checks both inputs and outputs
- **Batch dimensions** — `Batch()` for ML batch sizes
- **Ellipsis support** — `(..., n, m)` for variable leading dims
- **Grouped checks** — `ShapeContext` for imperative validation
- **Broadcasting** — `broadcast_shape` and `explain_broadcast`
- **ML helpers** — pre-defined dims and `attention_shapes`
- **JIT modes** — `"check"`, `"warn"`, or `"skip"` under JAX JIT

## Quick links

- [Getting Started](getting-started.md) — installation and first steps
- [Shape Contracts Guide](guide/shape-contracts.md) — core usage patterns
- [Unification](concepts/unification.md) — how dimension matching works under the hood
- [JIT Modes](concepts/jit-modes.md) — controlling validation under JAX JIT
- [API Reference](reference/core.md) — full API documentation
