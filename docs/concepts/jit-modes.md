# JIT Modes in ShapeGuard

## What is JIT?

**JIT = Just-In-Time Compilation**

Normally, Python runs code line by line (slow). JIT compiles your function into optimized machine code *before* running it (fast).

```python
import jax
import jax.numpy as jnp

def slow_function(x):
    return x @ x.T + jnp.sin(x)

fast_function = jax.jit(slow_function)  # Compile it!
```

- **First call**: JAX **traces** the function to understand what operations happen, then compiles
- **Next calls**: Runs the compiled version (10-100x faster)

---

## The Problem: Tracing

When JAX traces, it doesn't use real values. It uses **tracers** — abstract placeholders.

```python
@jax.jit
def f(x):
    print(x)        # Prints: Traced<ShapedArray(float32[3,4])>
    print(x.shape)  # Prints: (3, 4) — shape IS available
    print(x[0, 0])  # Prints: Traced<ShapedArray(float32[])> — NOT a real number
    return x + 1
```

During tracing:
- `x.shape` → **works** (static info available)
- `x[0, 0]` → **tracer**, not a real value
- `if x[0, 0] > 0:` → **ERROR** (can't branch on tracer)

---

## How This Affects ShapeGuard

ShapeGuard reads `.shape` to validate — this **works** under JIT:

```python
@jax.jit
@expects(x=(n, m))  # ✓ Can read x.shape during tracing
def f(x):
    return x + 1
```

**But there's overhead**: ShapeGuard runs on *every* trace. If shapes are static, checking repeatedly is wasteful.

---

## JIT Modes

ShapeGuard provides three modes for controlling validation behavior under JIT:

| Mode | On Valid Shape | On Invalid Shape | Use Case |
|------|----------------|------------------|----------|
| `"check"` | ✓ Pass silently | ❌ Raise exception | Development, debugging |
| `"warn"` | ✓ Pass silently | ⚠️ Log warning, continue | Gradual adoption |
| `"skip"` | — No validation | — No validation | Production, max performance |

### Mode: `"check"` (Default)

Always validate. Raise `ShapeGuardError` on mismatch.

```python
@expects(x=(n, 128), jit_mode="check")
@jax.jit
def layer(x):
    return x @ weights

layer(wrong_shape)
# → ShapeGuardError: dim[1] expected 128, got 64
# → Program CRASHES 💥
```

**Use when**: Developing, testing, or when shape correctness is critical.

### Mode: `"warn"`

Validate, but only log warnings on mismatch. Program continues.

```python
@expects(x=(n, 128), jit_mode="warn")
@jax.jit
def layer(x):
    return x @ weights

layer(wrong_shape)
# → WARNING: ShapeGuard: dim[1] expected 128, got 64 in layer
# → Program CONTINUES (might fail later or produce garbage)
```

**Use when**:
- Gradual adoption — adding ShapeGuard to existing code without breaking production
- Collecting all shape issues in a run instead of crashing on the first one
- Non-critical paths where you want visibility but not enforcement

### Mode: `"skip"`

Skip all validation under JIT. Zero overhead.

```python
@expects(x=(n, 128), jit_mode="skip")
@jax.jit
def layer(x):
    return x @ weights

layer(anything)
# → No validation at all under JIT
```

**Use when**: Production deployment after thorough testing.

---

## Configuration

### Global Setting

```python
from shapeguard import config

config.jit_mode = "skip"  # Apply to all decorated functions
```

### Per-Function Setting

```python
@expects(x=(n, m), jit_mode="warn")  # Override global for this function
@jax.jit
def specific_layer(x):
    ...
```

Per-function settings override the global configuration.

---

## How JIT Detection Works

ShapeGuard detects JAX tracing by checking the trace level:

```python
def _is_tracing() -> bool:
    """Are we inside JAX's JIT tracer?"""
    try:
        from jax._src.core import cur_sublevel
        return cur_sublevel().level > 0
    except ImportError:
        return False  # JAX not installed
```

---

## Execution Flow

```
Normal Python (no JIT):
  f(x) → ShapeGuard checks → function runs → result
  f(x) → ShapeGuard checks → function runs → result
  (validation on every call)

With JIT, mode="check":
  f(x) → trace → ShapeGuard checks → compile → run → result
  f(x) → run cached (shape already validated during trace)

With JIT, mode="skip":
  f(x) → trace → (no check) → compile → run → result
  f(x) → run cached
  (zero validation overhead)

With JIT, mode="warn":
  f(x) → trace → ShapeGuard checks → log if bad → compile → run → result
  f(x) → run cached
  (warnings logged but never crashes)
```

---

## Summary

Think of the modes like teachers grading homework:

- **`"check"`** = Strict teacher — "Wrong answer? You fail!"
- **`"warn"`** = Friendly teacher — "That looks wrong... but okay, let's see what happens"
- **`"skip"`** = No teacher — "I trust you did your homework"

### Recommended Workflow

1. **Development**: `jit_mode="check"` — Catch all shape bugs early
2. **Testing/CI**: `jit_mode="check"` — Ensure correctness
3. **Staging**: `jit_mode="warn"` — Monitor for issues without blocking
4. **Production**: `jit_mode="skip"` — Maximum performance after validation

---

## See also

- [Shape Contracts](../guide/shape-contracts.md) — using `jit_mode` with `@expects`, `@ensures`, and `@contract`
- [API Reference: Configuration](../reference/config.md) — `Config` class and `JitMode` type
- [API Reference: Decorators](../reference/decorators.md) — decorator signatures with `jit_mode` parameter
