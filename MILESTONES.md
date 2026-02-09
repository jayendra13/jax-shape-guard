# ShapeGuard Milestones

## Design Decisions

- **Dim identity**: Same object required (`n = Dim("n")` must be reused)
- **Performance**: Dev-only tool, prioritize error quality over speed
- **Backend priority**: NumPy-first, JAX support in v0.2

---

## Milestone 1: Core Foundation (v0.1-alpha)

### Goal
Minimal working library with symbolic dimensions, shape checking, and decorator.

### Files
```
shapeguard/
  __init__.py       # Public API exports
  core.py           # Dim class, UnificationContext
  spec.py           # Shape specification matching
  decorator.py      # @expects decorator
  errors.py         # ShapeGuardError
  _compat.py        # Backend detection
tests/
  test_core.py
  test_spec.py
  test_decorator.py
  test_errors.py
  conftest.py
pyproject.toml
```

### Deliverables
- [x] `Dim` class with identity-based unification
- [x] `UnificationContext` for tracking bindings across arguments
- [x] `ShapeSpec` matching: concrete `(3, 4)`, symbolic `(n, m)`, wildcard `(None, 4)`
- [x] `@expects` decorator for input validation
- [x] `ShapeGuardError` with function, argument, expected, actual, reason
- [x] `check_shape(x, spec, name)` standalone function
- [x] Backend-agnostic shape extraction (works with any `.shape` attribute)
- [x] Unit tests with 90%+ coverage (91% achieved)

### API Surface
```python
from shapeguard import Dim, expects, check_shape, ShapeGuardError

n, m = Dim("n"), Dim("m")

@expects(x=(n, m), y=(m,))
def forward(x, y):
    return x @ y

check_shape(arr, (n, 128), name="input")
```

---

## Milestone 2: ML-Practical Features (v0.1-beta)

### Goal
Ergonomic features for real ML workflows.

### Deliverables
- [x] `Batch` dimension (always first, flexible size per call)
- [x] Ellipsis support `(..., n, m)` for variable leading dims
- [x] `ShapeContext` manager for grouped checks with shared bindings
- [x] Improved error messages with binding trace (92% coverage)

### API Additions
```python
from shapeguard import Batch, ShapeContext

B = Batch()

@expects(x=(B, n, m))
def layer(x): ...

@expects(x=(..., n, m))
def normalize(x): ...

with ShapeContext() as ctx:
    ctx.check(x, (n, m), "x")
    ctx.check(y, (m, k), "y")
```

---

## Milestone 3: JAX Integration (v0.2)

### Goal
Seamless JAX compatibility including JIT behavior.

### Deliverables
- [x] JIT/tracing detection
- [x] Configurable JIT modes: `skip`, `warn`, `check`
- [x] PyTree shape specs for nested params
- [ ] Performance benchmarks (deferred)

### API Additions
```python
from shapeguard import expects, config

config.jit_mode = "skip"  # Global setting

@expects(x=(B, n, m), jit_mode="static")  # Per-function
@jax.jit
def forward(x): ...

@expects(
    params={"weights": (n, m), "bias": (m,)},
    x=(B, n)
)
def apply(params, x): ...
```

---

## Milestone 4: Broadcasting Support (v0.2)

### Goal
Explicit broadcasting inspection and validation.

### Deliverables
- [x] `broadcast_shape()` for concrete shapes
- [x] `explain_broadcast()` step-by-step explainer
- [ ] `_broadcast=True` option in `@expects` (deferred)

### API Additions
```python
from shapeguard import broadcast_shape, explain_broadcast

broadcast_shape((3, 1), (1, 4))  # → (3, 4)
broadcast_shape(a, b)            # From arrays

explain_broadcast((3, 1, 4), (5, 4))
# Broadcasting (3, 1, 4) with (5, 4):
#   Dim 0: 3 (from left only)
#   Dim 1: 1 → 5 (broadcast)
#   Dim 2: 4 = 4 (match)
#   Result: (3, 5, 4)
```

---

## Milestone 5: Output Contracts (v0.3)

### Goal
Validate function outputs, not just inputs.

### Deliverables
- [x] `@ensures` decorator for output validation
- [x] `@contract` combined decorator
- [x] Tuple/dict output support

### API Additions
```python
from shapeguard import expects, ensures, contract

@expects(a=(n, m), b=(m, k))
@ensures(result=(n, k))
def matmul(a, b):
    return a @ b

@contract(
    inputs={"a": (n, m), "b": (m, k)},
    output=(n, k)
)
def matmul(a, b):
    return a @ b
```

---

## Milestone 6: ML Helpers (v0.3)

### Goal
Domain-specific helpers for common ML patterns.

### Deliverables
- [x] Pre-defined dims: `B`, `T`, `C`, `H`, `W`, `D`
- [x] `attention_shapes()` helper
- [x] `conv_output_shape()` calculator

### API Additions
```python
from shapeguard.ml import B, T, C, H, W, D
from shapeguard.ml import attention_shapes, conv_output_shape

@expects(x=(B, T, D))
def transformer_layer(x): ...

@expects(**attention_shapes(B, heads, seq_q, seq_k, d_k))
def attention(q, k, v): ...

out_shape = conv_output_shape(
    input=(B, C, 224, 224),
    kernel=(3, 3),
    stride=2,
    padding=1
)
```

---

## Milestone 7: Testing Utilities (v0.4)

### Goal
Property-based testing support.

### Deliverables
- [ ] Hypothesis strategies for shaped arrays
- [ ] `verify_contract()` auto-test generator
- [ ] pytest plugin

### API Additions
```python
from shapeguard.testing import arrays, verify_contract
import hypothesis

@hypothesis.given(x=arrays(shape=(n, m), n=(1, 100), m=(1, 100)))
def test_normalize(x):
    result = normalize(x)
    assert result.shape == x.shape

verify_contract(matmul, samples=100)
```

---

## Summary Timeline

| Milestone | Version | Status |
|-----------|---------|--------|
| 1. Core Foundation | v0.1-alpha | ✅ Complete (91% coverage) |
| 2. ML Features | v0.1-beta | ✅ Complete (92% coverage) |
| 3. JAX Integration | v0.2 | ✅ Complete (92% coverage) |
| 4. Broadcasting | v0.2 | ✅ Complete |
| 5. Output Contracts | v0.3 | ✅ Complete (91% coverage) |
| 6. ML Helpers | v0.3 | ✅ Complete |
| 7. Testing Utils | v0.4 | 🔲 Not started |
