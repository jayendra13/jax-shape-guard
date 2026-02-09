# JAX Shape Error Issues Analysis

Analysis of JAX GitHub issues related to dimension mismatches and uninformative error messages, categorized by how ShapeGuard can address them.

---

## Category 1: Unclear Which Argument Failed

Issues where the error dumps information but doesn't pinpoint the problematic argument.

### [Issue #2190: Confusing General Convolution type error](https://github.com/google/jax/issues/2190)

**Problem**: User implementing a GAN received a verbose, low-level error dumping entire function signatures without identifying which argument was wrong.

**Error received**:
```
TypeError: ConvGeneralDilated() ... expected padding: Span[Tuple[int, int]]
```
The user had to manually discover that `padding=(2, 2)` should be `padding=((2, 2),)`.

**User request**: "Implement a less verbose and more informative error that says what argument has received the wrong type."

**ShapeGuard solution**:
```python
@expects(
    lhs=(B, C_in, H, W),
    rhs=(C_out, C_in, kH, kW),
    # padding validated separately
)
def conv2d(lhs, rhs, padding):
    ...

# Error would clearly show:
# ShapeGuardError:
#   function: conv2d
#   argument: lhs
#   expected: (B, C_in, H, W)
#   actual:   (16, 3, 28)  ← missing dimension!
```

---

### [Issue #3613: Improve error message w/ incorrect out_axes argument to vmap](https://github.com/google/jax/issues/3613)

**Problem**: When passing wrong number of `out_axes` to `vmap`, error doesn't indicate whether `in_axes` or `out_axes` is wrong.

**Discussion**: Maintainer agreed to "pass in a string that gets included in the error message."

**ShapeGuard solution**: Contract-style validation makes it explicit which argument has the shape problem:
```python
@expects(inputs=(B, n), weights=(n, m))
@ensures(result=(B, m))
def layer(inputs, weights):
    return inputs @ weights

# Clear error: "argument: weights, expected: (n, m), actual: (10, 20, 30)"
```

---

## Category 2: Silent Shape Acceptance → Late/Cryptic Failure

Issues where JAX accepts logically wrong shapes and fails later in XLA or produces garbage.

### [Issue #4316: Bug in Jax shape checking rules](https://github.com/google/jax/issues/4316)

**Problem**: Convolution accepted mismatched tensor dimensions, then failed deep in XLA:
```
RuntimeError: Convolution arguments must have same number of dimensions.
Got: f32[2,1,1,1,1,1] and f32[2,2,1,1]
This is a bug in JAX's shape-checking rules; please report it!
```

**Root cause**: Shape-checking validation was missing at the JAX level.

**ShapeGuard solution**: Fail-fast at function entry:
```python
@expects(lhs=(B, C_in, *spatial), rhs=(C_out, C_in, *kernel))
def conv_general_dilated(lhs, rhs, ...):
    ...

# Would catch rank mismatch immediately with clear message
```

---

### [Issue #5087: Shape-checking bug](https://github.com/google/jax/issues/5087)

**Problem**: User passed `[1]` instead of `[1, 1]`, got cryptic "bug in JAX's shape-checking rules" message.

**Actual issue**: Simple user error, but error message blamed JAX internals.

**ShapeGuard solution**: Validate before calling internals:
```python
n = Dim("n")
check_shape(indices, (n,), "indices")
check_shape(updates, (n, features), "updates")
# Clear: "argument: indices, expected rank 1, got rank 2"
```

---

### [Issue #1522: Unexpected behaviour of np.broadcast_to](https://github.com/google/jax/issues/1522)

**Problem**: `np.broadcast_to(np.ones((2, 3)), (1, 3))` should raise an error but returned `(1, 2, 3)`.

**ShapeGuard solution**: Explicit shape contracts prevent silent broadcasting surprises:
```python
@expects(x=(n, m))
@ensures(result=(target_n, target_m))
def broadcast_to_2d(x, target_shape):
    ...
```

---

## Category 3: Dimension Unification Across Arguments

Issues where shapes must be consistent across multiple inputs but errors don't explain the relationship.

### [Issue #1263: cond batching rule shape mismatch](https://github.com/google/jax/issues/1263)

**Problem**: Error when using `lax.cond` with `vmap`:
```
TypeError: select pred must be scalar or have the same shape as on_true and
on_false, got pred shape (3,) for on_true and on_false of shape (3, 4).
```

**Root cause**: `Select` primitive has limited broadcasting; pred needed manual broadcast.

**ShapeGuard solution**: Make shape relationships explicit:
```python
B, n = Dim("B"), Dim("n")

@expects(pred=(B,), on_true=(B, n), on_false=(B, n))
def batched_cond(pred, on_true, on_false):
    # Contract shows B must unify across all three arguments
    ...

# Error: "dimension 'B' bound to 3 from pred.shape[0], but got 4 from on_true.shape[0]"
```

---

### [Issue #5849: Shape error combining custom_jvp with vmap](https://github.com/google/jax/issues/5849)

**Problem**: Gradient computation ignored vmap, producing shape `(250, 250, 2)` instead of scalar.

**Root cause**: Broadcasting behavior in autodiff transpose rules.

**ShapeGuard solution**: Output contracts catch unexpected shapes:
```python
@ensures(result=())  # Scalar output expected
def compute_energy_grad(params):
    ...

# Error: "output expected: (), actual: (250, 250, 2)"
```

---

### [Issue #5832: Error with custom_vjp + scan + vmap](https://github.com/google/jax/issues/5832)

**Problem**: Cotangent dimensions didn't match because "those sizes need to agree."

**ShapeGuard solution**: Unified dimension tracking:
```python
B, T, D = Dim("B"), Dim("T"), Dim("D")

@expects(x=(B, T, D))
@ensures(result=(B, T, D), cotangent=(B, T, D))
def scan_layer(x):
    ...
```

---

## Category 4: Broadcasting Opacity

Issues where broadcasting behavior is unclear or surprising.

### [Issue #21331: jax.pure_callback broadcasting error](https://github.com/google/jax/issues/21331)

**Problem**: After JAX update, callback produced:
```
TypeError: mul got incompatible shapes for broadcasting: (30,), (80,)
```

**Context**: Callback functions now receive JAX arrays vs NumPy arrays, changing broadcasting behavior.

**ShapeGuard solution**: `broadcast_shape()` for explicit reasoning:
```python
from shapeguard import broadcast_shape, explain_broadcast

# Before the multiply
explain_broadcast(a.shape, b.shape)
# Output:
# Broadcasting (30,) with (80,):
#   Error: dimension 0: 30 ≠ 80 (neither is 1)
```

---

### [Issue #5276: lax.broadcast_in_dim transposition rule](https://github.com/google/jax/issues/5276)

**Problem**: Transposition rule assumed input shape matched broadcast dimensions.

**ShapeGuard solution**: Explicit dimension contracts before broadcast:
```python
@expects(x=(n,), broadcast_dims=(0,))
def broadcast_in_dim(x, shape, broadcast_dims):
    check_shape(x, tuple(shape[i] for i in broadcast_dims), "x vs broadcast_dims")
```

---

## Category 5: Rank Mismatches

Issues where tensor rank (number of dimensions) is wrong.

### [Issue #6605: Better error for jax.numpy.pad() pad_widths mismatch](https://github.com/google/jax/issues/6605)

**Problem**: Error message was:
```
ValueError: pad_width given unexpected structure: [[0 0] [0 0] [0 0]].
See docstring for valid pad_width formats.
```
User had 2D array but 3 padding pairs.

**User needed**: "pad_widths length (3) must match array rank (2)"

**ShapeGuard solution**: Rank checking with clear messages:
```python
@expects(x=(...,))  # Any rank
def pad(x, pad_width, mode):
    if len(pad_width) != x.ndim:
        raise ShapeGuardError(
            f"pad_width has {len(pad_width)} pairs but array has rank {x.ndim}",
            argument="pad_width",
            expected=f"{x.ndim} pairs",
            actual=f"{len(pad_width)} pairs"
        )
```

---

### [Issue #3954: Wrong jnp.transpose behaviour for wrong axis length](https://github.com/google/jax/issues/3954)

**Problem**: Inconsistent error handling for transpose with wrong permutation length.

**ShapeGuard solution**:
```python
n_dims = Dim("n_dims")

@expects(x=(...,))  # Capture rank
def transpose(x, axes):
    check_shape(axes, (len(x.shape),), "axes length")
```

---

## Category 6: Tracer/JIT Shape Issues

Issues specific to JAX's tracing mechanism.

### [Issue #2797: grad of vmap of odeint with rng gives tracer error](https://github.com/google/jax/issues/2797)

**Problem**: `UnexpectedTracerError` - tracer escaped through global state.

**ShapeGuard relevance**: While not directly a shape issue, ShapeGuard's JIT-aware mode (v0.2) can:
- Skip validation under tracing to avoid interference
- Validate static shapes only
- Provide cleaner errors when shapes are traced

---

### [Issue #845: vmap fails due to internal mishandling of scalar](https://github.com/google/jax/issues/845)

**Problem**: Shape errors with scalar dimensions in vmap:
```
AxisError: transpose permutation [1, 0] must be a permutation of [0, 1]
```

**ShapeGuard solution**: Handle scalar edge cases explicitly:
```python
@expects(x=(...,), y=(...,))  # Accept any rank including 0
def vdot(x, y):
    if x.ndim == 0 and y.ndim == 0:
        return x * y  # Scalar path
    ...
```

---

## Summary: How ShapeGuard Addresses Each Category

| Category | Count | ShapeGuard Feature |
|----------|-------|-------------------|
| 1. Unclear which argument | 2 | `@expects` with named args + clear error format |
| 2. Silent acceptance | 4 | Fail-fast validation at function entry |
| 3. Dimension unification | 3 | Symbolic `Dim` with binding tracking |
| 4. Broadcasting opacity | 2 | `broadcast_shape()`, `explain_broadcast()` |
| 5. Rank mismatches | 2 | Rank checking with `RankMismatchError` |
| 6. Tracer/JIT issues | 2 | JIT-aware mode (skip/warn/static) |

---

## Key Patterns ShapeGuard Must Handle

Based on this analysis, ShapeGuard v0.1 should prioritize:

1. **Clear argument attribution**: Always show function name + argument name
2. **Expected vs actual**: Side-by-side comparison in errors
3. **Binding trace**: Show where each symbolic dimension got its value
4. **Rank-first checking**: Catch rank mismatches before dimension mismatches

For v0.2 (JAX integration):

1. **JIT detection**: Don't interfere with tracing
2. **Static shape validation**: Validate known shapes even under JIT
3. **PyTree support**: Handle nested param structures

---

## Example: Complete ShapeGuard Solution

Here's how a transformer attention function would look with ShapeGuard:

```python
from shapeguard import Dim, expects, ensures

B = Dim("batch")
H = Dim("heads")
T = Dim("seq_len")
K = Dim("key_seq")
D = Dim("head_dim")

@expects(
    query=(B, H, T, D),
    key=(B, H, K, D),
    value=(B, H, K, D),
    mask=(B, 1, T, K)  # or None
)
@ensures(result=(B, H, T, D))
def attention(query, key, value, mask=None):
    scores = query @ key.swapaxes(-2, -1) / D**0.5
    if mask is not None:
        scores = scores + mask
    weights = softmax(scores, axis=-1)
    return weights @ value
```

**If shapes mismatch**:
```
ShapeGuardError:
  function: attention
  argument: key
  expected: (batch, heads, key_seq, head_dim)
  actual:   (32, 8, 128, 32)
  reason:   dimension 'head_dim' bound to 64 from query.shape[3], but got 32 from key.shape[3]
  bindings: {batch=32 (from query[0]), heads=8 (from query[1]), head_dim=64 (from query[3])}
```

This error tells the user exactly:
- Which function failed
- Which argument was wrong
- What was expected vs actual
- Why it failed (head_dim conflict)
- Where each dimension binding came from

---

## Appendix: Complete Issue Index by Category

### Category 1: Unclear Which Argument Failed

| Issue | Title | State | Core Problem |
|-------|-------|-------|--------------|
| [#2190](https://github.com/google/jax/issues/2190) | Confusing General Convolution type error | Closed | Verbose error dumps all args, doesn't identify which one is wrong |
| [#3613](https://github.com/google/jax/issues/3613) | Improve error message w/ incorrect out_axes argument to vmap | Closed | Error doesn't distinguish in_axes vs out_axes problem |
| [#2495](https://github.com/google/jax/issues/2495) | Confusing error message when indexing arrays with floats | Closed | Error doesn't clearly explain float indexing issue |

### Category 2: Silent Shape Acceptance → Late/Cryptic Failure

| Issue | Title | State | Core Problem |
|-------|-------|-------|--------------|
| [#4316](https://github.com/google/jax/issues/4316) | Bug in Jax shape checking rules | Closed | conv_general_dilated accepts mismatched dims, fails in XLA |
| [#5087](https://github.com/google/jax/issues/5087) | Shape-checking bug | Closed | Wrong shape accepted, "bug in JAX's shape-checking rules" error |
| [#5536](https://github.com/google/jax/issues/5536) | Bug in shape checking rule when calling jax.lax.reduce_window | Closed | Missing shape validation for reduce_window |
| [#4734](https://github.com/google/jax/issues/4734) | SIGFPE + RuntimeError caused by missing shape rule for FFT | Closed | Missing shape rule causes SIGFPE crash |
| [#1522](https://github.com/google/jax/issues/1522) | Unexpected behaviour of np.broadcast_to | Closed | broadcast_to returns wrong shape instead of erroring |
| [#1672](https://github.com/google/jax/issues/1672) | jax.lax.pswapaxes calculates incorrect shapes with soft_pmap | Closed | Incorrect abstract_eval rule produces wrong shapes |
| [#459](https://github.com/google/jax/issues/459) | Generators should not be accepted as values for shape argument | Closed | Generator accepted for shape, causes cryptic error later |
| [#10729](https://github.com/google/jax/issues/10729) | Suspected bug in jax.lax.conv_general_dilated_patches | Closed | Missing input validation for padding argument |

### Category 3: Dimension Unification Across Arguments

| Issue | Title | State | Core Problem |
|-------|-------|-------|--------------|
| [#1263](https://github.com/google/jax/issues/1263) | cond batching rule shape mismatch | Closed | pred shape doesn't broadcast with on_true/on_false |
| [#5849](https://github.com/google/jax/issues/5849) | Shape error combining custom_jvp with vmap | Closed | Gradient shape ignores vmap, returns batched dims |
| [#5832](https://github.com/google/jax/issues/5832) | Error with custom_vjp + scan + vmap | Closed | Cotangent dimensions don't agree across transforms |
| [#598](https://github.com/google/jax/issues/598) | Abstract shape computed incorrectly for lax.scan | Closed | scan output missing loop dimension |
| [#4888](https://github.com/google/jax/issues/4888) | triangular_solve does not work well with batching | Closed | Batched dimensions not handled correctly |
| [#561](https://github.com/google/jax/issues/561) | MAML with Convnets Reshape Error | Closed | Reshape total size mismatch in meta-learning |

### Category 4: Broadcasting Opacity

| Issue | Title | State | Core Problem |
|-------|-------|-------|--------------|
| [#21331](https://github.com/google/jax/issues/21331) | jax.pure_callback produces "TypeError: mul got incompatible shapes for broadcasting" | Open | Broadcasting fails silently after API change |
| [#5276](https://github.com/google/jax/issues/5276) | lax.broadcast_in_dim transposition rule | Closed | Transpose rule assumes shape matches broadcast dims |
| [#14751](https://github.com/google/jax/issues/14751) | Remove unnecessary broadcast from jnp.vectorize | Closed | Unnecessary broadcasts introduced by vectorize |

### Category 5: Rank Mismatches

| Issue | Title | State | Core Problem |
|-------|-------|-------|--------------|
| [#6605](https://github.com/google/jax/issues/6605) | Issue better error message when jax.numpy.pad() pad_widths mismatches array rank | Closed | pad_widths length vs array rank mismatch unclear |
| [#3954](https://github.com/google/jax/issues/3954) | Wrong jnp.transpose behaviour for wrong axis length | Closed | Inconsistent error for wrong permutation length |
| [#2075](https://github.com/google/jax/issues/2075) | Initialisers fail for 1D shaped arrays | Closed | Rank-1 arrays not handled correctly |
| [#187](https://github.com/google/jax/issues/187) | Failure while indexing across both dimensions of a 2D array | Closed | Multi-dimensional indexing rank issues |

### Category 6: Tracer/JIT Shape Issues

| Issue | Title | State | Core Problem |
|-------|-------|-------|--------------|
| [#2797](https://github.com/google/jax/issues/2797) | grad of vmap of odeint with rng-dependent dynamics gives tracer error | Closed | Tracer escaped through global state |
| [#845](https://github.com/google/jax/issues/845) | vmap fails possibly due to internal mishandling of scalar | Closed | Scalar dimensions cause transpose errors |
| [#3822](https://github.com/google/jax/issues/3822) | Some tracer level is None when vmap'ing a custom_jvp | Closed | Tracer level issues with custom transforms |
| [#3883](https://github.com/google/jax/issues/3883) | vmap of dynamic_slice of a scalar fails | Closed | dynamic_slice with scalar in vmap fails |
| [#12596](https://github.com/google/jax/issues/12596) | Vmap of pure_callback with multiple arguments error | Closed | pure_callback shape handling under vmap |
| [#22489](https://github.com/google/jax/issues/22489) | jax.numpy.digitize doesn't work with shape polymorphism | Open | Shape polymorphism not supported |

### Category 7: Error Message Quality (General)

| Issue | Title | State | Core Problem |
|-------|-------|-------|--------------|
| [#8557](https://github.com/google/jax/issues/8557) | NotImplementedError: Differentiation rule for 'custom_lin' not implemented | Closed | Unhelpful error for missing differentiation rule |
| [#5380](https://github.com/google/jax/issues/5380) | Bug when computing Hessian Inverse | Closed | Shape error in Hessian computation |
| [#637](https://github.com/google/jax/issues/637) | Inconsistent output from documentation for hessian | Closed | Documentation shape examples inconsistent |
| [#4207](https://github.com/google/jax/issues/4207) | Array shape bug when using jax.lax.scan | Closed | scan produces unexpected output shapes |
| [#4651](https://github.com/google/jax/issues/4651) | Deepcopy of bfloat16 array messes up bfloat16 definition | Closed | Type/shape errors after deepcopy |
| [#1579](https://github.com/google/jax/issues/1579) | np.array issue | Closed | Array construction shape issues |
| [#2285](https://github.com/google/jax/issues/2285) | Vmap with non-array containers | Closed | Container handling shape issues |
| [#967](https://github.com/google/jax/issues/967) | Don't allow float indices and shapes | Closed | Float values in shape specifications |
| [#4638](https://github.com/google/jax/issues/4638) | Question regarding jax.lax.all_to_all behaviour | Closed | Unclear shape behavior documentation |
| [#15412](https://github.com/google/jax/issues/15412) | Recompile slow down due to different input shapes | Open | Shape variation triggers recompilation |

---

## Statistics

| Category | Issue Count |
|----------|-------------|
| 1. Unclear which argument | 3 |
| 2. Silent acceptance | 8 |
| 3. Dimension unification | 6 |
| 4. Broadcasting opacity | 3 |
| 5. Rank mismatches | 4 |
| 6. Tracer/JIT issues | 6 |
| 7. Error message quality | 10 |
| **Total** | **40** |

**Most common patterns:**
1. Error messages don't identify the problematic argument
2. Shape validation missing at JAX level, fails in XLA
3. Dimension relationships across arguments not explained
4. Broadcasting behavior implicit and surprising
