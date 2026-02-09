# Understanding Unification (The Simple Version)

> This document explains the core concept behind ShapeGuard's dimension checking.

## The Matching Game

Imagine you have a box of **colored stickers** with **letters** on them: `n`, `m`, `k`

When you put a letter sticker on something, you're giving it a **secret number**.

**The rule**: The same letter must ALWAYS mean the same number.

---

## Example: Two Boxes

```
Box A has shape: (n, m)  →  (3, 4)
Box B has shape: (m, k)  →  (?, ?)
```

You look at Box A:
- "n" gets the number **3** ✓
- "m" gets the number **4** ✓

Now you look at Box B:
- "m" must still be **4** (because we already decided!)
- "k" can be anything new

---

## The Notebook (UnificationContext)

`UnificationContext` is like a **notebook** that remembers what number each letter got:

```
📓 My Notebook:
   n = 3  (I learned this from Box A, first spot)
   m = 4  (I learned this from Box A, second spot)
```

When you see the same letter again, you check the notebook:
- Same number? ✅ Great!
- Different number? ❌ That's cheating! Error!

---

## Real Code Example

```python
from shapeguard import Dim, expects

n = Dim("n")  # A sticker named "n"
m = Dim("m")  # A sticker named "m"

@expects(a=(n, m), b=(m, n))
def swap_multiply(a, b):
    return a @ b
```

You call it with:
```python
a = array of shape (3, 4)
b = array of shape (4, 3)
```

The notebook fills in:
```
📓 Notebook:
   n = 3  (from a, spot 0)
   m = 4  (from a, spot 1)
```

Then checks `b`:
- `b[0]` should be `m` → Is it 4? YES ✅
- `b[1]` should be `n` → Is it 3? YES ✅

**It matches!** Function runs.

---

## When It Catches Errors

```python
a = array of shape (3, 4)
b = array of shape (5, 3)  # Wrong!
```

Notebook says `m = 4`, but `b[0]` is **5**.

```
❌ "Hey! You said 'm' was 4, but now you're saying it's 5!"
   "That's not allowed!"
```

---

## The Technical Version

For those who want the precise definition:

**Unification** is the process of finding a consistent assignment of concrete values to symbolic variables such that all constraints are satisfied.

In ShapeGuard:
- **Symbolic variables** = `Dim` objects (e.g., `Dim("n")`)
- **Concrete values** = integers from actual array shapes
- **Constraints** = shape specifications in `@expects`

The `UnificationContext` maintains a mapping from `Dim` objects to their bound integer values, raising `UnificationError` if the same `Dim` is constrained to different values.

```python
class UnificationContext:
    bindings: dict[Dim, Binding]  # The "notebook"

    def bind(self, dim, value, source):
        """Add a new entry or verify consistency."""
        if dim in self.bindings:
            if self.bindings[dim].value != value:
                raise UnificationError(...)  # Cheating detected!
        else:
            self.bindings[dim] = Binding(value, source)
```

---

## Why "Unification"?

The term comes from logic programming and type theory. When you have:

```
f(X, X)  matched against  f(3, 3)
```

You "unify" `X` with `3`. If you then try to match against `f(3, 4)`, unification fails because `X` can't be both `3` and `4`.

ShapeGuard applies this same concept to array dimensions, ensuring that when you declare two dimensions should be the same (by using the same `Dim` object), they actually are at runtime.

---

## Key Insight

The **notebook metaphor** captures why ShapeGuard's error messages are helpful:

Traditional error:
```
ValueError: shapes (3,4) and (5,3) not aligned
```

ShapeGuard error:
```
UnificationError: dimension 'm' bound to 4 from a.shape[1],
                  but got 5 from b.shape[0]
```

The notebook remembers **where** each binding came from, so when there's a conflict, it can tell you exactly what went wrong and why.

---

## See also

- [Shape Contracts](../guide/shape-contracts.md) — using `Dim` with `@expects`, `@ensures`, and `@contract`
- [Batch & Ellipsis](../guide/batch-and-ellipsis.md) — `Batch` dimensions and `ShapeContext`
- [API Reference: Core](../reference/core.md) — `Dim`, `UnificationContext`, `Binding`
- [API Reference: Errors](../reference/errors.md) — `UnificationError` and other exceptions
