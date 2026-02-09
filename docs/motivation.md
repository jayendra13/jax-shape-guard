# Why ShapeGuard?

Shape bugs are among the most frustrating errors in numerical computing. They're silent, late, and cryptic. This page shows real JAX/NumPy pain points — drawn from [40+ JAX GitHub issues](https://github.com/google/jax/issues) — and how ShapeGuard addresses each category.

---

## 1. "Which argument is wrong?"

JAX errors often dump verbose internal state without identifying the problematic argument.

**Real issue**: [jax#2190](https://github.com/google/jax/issues/2190) — User implementing a GAN received a wall of text from `ConvGeneralDilated` without any indication of which argument was wrong. They had to manually discover that `padding=(2, 2)` should be `padding=((2, 2),)`.

**The JAX error**:
```
TypeError: ConvGeneralDilated() ... expected padding: Span[Tuple[int, int]]
```

**With ShapeGuard**:
```python
from shapeguard import Dim, expects

B, C_in, H, W = Dim("B"), Dim("C_in"), Dim("H"), Dim("W")
C_out, kH, kW = Dim("C_out"), Dim("kH"), Dim("kW")

@expects(lhs=(B, C_in, H, W), rhs=(C_out, C_in, kH, kW))
def conv2d(lhs, rhs, padding):
    ...
```

```
ShapeGuardError:
  function: conv2d
  argument: lhs
  expected: (B, C_in, H, W)
  actual:   (16, 3, 28)
  reason:   expected rank 4, got rank 3
```

The error immediately names the function, the argument, and why it failed.

---

## 2. Silent acceptance, cryptic failure later

JAX sometimes accepts logically wrong shapes and fails deep inside XLA with an internal error.

**Real issue**: [jax#4316](https://github.com/google/jax/issues/4316) — Convolution accepted mismatched tensor dimensions, then crashed in XLA:

```
RuntimeError: Convolution arguments must have same number of dimensions.
Got: f32[2,1,1,1,1,1] and f32[2,2,1,1]
This is a bug in JAX's shape-checking rules; please report it!
```

The user's code was wrong, but the error blamed JAX internals.

**With ShapeGuard**, validation happens at function entry — before any XLA compilation:

```python
@expects(lhs=(B, C_in, H, W), rhs=(C_out, C_in, kH, kW))
def conv(lhs, rhs):
    ...

conv(x_6d, kernel_4d)
```

```
ShapeGuardError:
  function: conv
  argument: lhs
  expected: (B, C_in, H, W)
  actual:   (2, 1, 1, 1, 1, 1)
  reason:   expected rank 4, got rank 6
```

Fail-fast, clear cause, no XLA involved.

---

## 3. Dimensions must agree across arguments

When multiple inputs must share dimensions, standard errors don't explain the relationship.

**Real issue**: [jax#1263](https://github.com/google/jax/issues/1263) — Using `lax.cond` with `vmap` produced:

```
TypeError: select pred must be scalar or have the same shape as on_true and
on_false, got pred shape (3,) for on_true and on_false of shape (3, 4).
```

The error doesn't explain *why* the shapes should match or *where* the mismatch originated.

**With ShapeGuard**, symbolic dimensions make the relationship explicit:

```python
B, n = Dim("B"), Dim("n")

@expects(pred=(B,), on_true=(B, n), on_false=(B, n))
def batched_cond(pred, on_true, on_false):
    ...
```

```
ShapeGuardError:
  function: batched_cond
  argument: on_false
  expected: (B, n)
  actual:   (4, 5)
  reason:   dimension 'B' bound to 3 from pred.shape[0], but got 4 from on_false.shape[0]
  bindings: {B=3 (from pred[0]), n=4 (from on_true[1])}
```

The error traces `B=3` back to `pred.shape[0]` and shows exactly where the conflict is.

---

## 4. Broadcasting surprises

Broadcasting is implicit and can produce unexpected results instead of errors.

**Real issue**: [jax#21331](https://github.com/google/jax/issues/21331) — After a JAX update, a callback function started failing:

```
TypeError: mul got incompatible shapes for broadcasting: (30,), (80,)
```

No context about *why* the shapes are incompatible or what broadcasting would look like.

**With ShapeGuard**, use `explain_broadcast` to debug:

```python
from shapeguard import explain_broadcast

print(explain_broadcast((30,), (80,)))
```

```
Broadcasting (30,) with (80,):
  Step 1: Align shapes from right
    (30)
    (80)
  Step 2: Compare dimensions
    dim -1: 30, 80 (INCOMPATIBLE)
  Error: Cannot broadcast - incompatible dimensions
```

Or validate upfront with `broadcast_shape`:

```python
from shapeguard import broadcast_shape

broadcast_shape(a, b)  # raises BroadcastError with clear message
```

---

## 5. Unexpected output shapes

Gradient and transform compositions can produce outputs with wrong shapes that propagate silently.

**Real issue**: [jax#5849](https://github.com/google/jax/issues/5849) — Combining `custom_jvp` with `vmap` produced shape `(250, 250, 2)` instead of a scalar, because the gradient computation ignored the vmap batching.

**With ShapeGuard**, output contracts catch this immediately:

```python
from shapeguard import expects, ensures

@expects(params=(...,))
@ensures(result=())  # scalar output expected
def compute_energy(params):
    ...
```

```
ShapeGuardError:
  function: compute_energy
  argument: result
  expected: ()
  actual:   (250, 250, 2)
  reason:   expected rank 0, got rank 3
```

The bug is caught at the function boundary instead of propagating through the rest of the computation.

---

## 6. Rank mismatches with unhelpful messages

When the number of dimensions is wrong, errors often don't state this directly.

**Real issue**: [jax#6605](https://github.com/google/jax/issues/6605) — `jax.numpy.pad()` with mismatched `pad_width`:

```
ValueError: pad_width given unexpected structure: [[0 0] [0 0] [0 0]].
See docstring for valid pad_width formats.
```

The user had a 2D array but 3 padding pairs. The error doesn't mention the rank mismatch.

**With ShapeGuard**, rank checking is explicit:

```python
n, m = Dim("n"), Dim("m")

@expects(x=(n, m))
def pad_2d(x, pad_width):
    ...

pad_2d(array_2d, pad_width_for_3d)
```

```
ShapeGuardError:
  function: pad_2d
  argument: x
  expected: (n, m)
  actual:   (28, 28, 3)
  reason:   expected rank 2, got rank 3
```

---

## The pattern

Across all these categories, the same problems repeat:

1. **Errors don't name the argument** — you have to guess which input is wrong
2. **Validation happens too late** — shapes pass through Python into XLA before failing
3. **Dimension relationships are implicit** — nothing explains *why* two shapes should match
4. **Error messages lack context** — no binding trace, no expected-vs-actual comparison

ShapeGuard addresses all four by validating at the function boundary with symbolic dimensions that track where each binding came from. See [Getting Started](getting-started.md) to try it, or [Unification](concepts/unification.md) to understand how dimension tracking works.
