# Broadcasting

ShapeGuard provides utilities for computing and explaining NumPy-style broadcasting.

## Computing broadcast shapes

`broadcast_shape` computes the result shape from broadcasting multiple shapes together, following NumPy rules:

```python
from shapeguard import broadcast_shape

broadcast_shape((3, 1), (1, 4))      # (3, 4)
broadcast_shape((2, 3, 4), (4,))     # (2, 3, 4)
broadcast_shape((5, 1, 3), (1, 4, 1))  # (5, 4, 3)
```

### Broadcasting rules

1. Align shapes from the right
2. Dimensions match if they are equal or one of them is 1
3. Missing dimensions on the left are treated as 1

### Working with arrays

`broadcast_shape` accepts arrays directly — it reads their `.shape`:

```python
import numpy as np

a = np.zeros((3, 1))
b = np.zeros((1, 4))
broadcast_shape(a, b)  # (3, 4)
```

### Error on incompatible shapes

When shapes can't be broadcast, you get a `BroadcastError`:

```python
broadcast_shape((3,), (4,))
# BroadcastError: Cannot broadcast shapes (3,) (4,)
#   reason: cannot broadcast dimension -1: sizes 3, 4 are incompatible
#           (must be equal or 1)
```

## Explaining broadcasts

`explain_broadcast` returns a human-readable, step-by-step explanation of the broadcast process. This is useful for debugging shape issues:

```python
from shapeguard import explain_broadcast

print(explain_broadcast((3, 1, 4), (5, 4)))
```

Output:

```
Broadcasting (3, 1, 4) with (5, 4):
  Step 1: Align shapes from right
    (3, 1, 4)
    (   5, 4)
  Step 2: Compare dimensions
    dim -3: 3 (only in first shape)
    dim -2: 1 → 5 (broadcast)
    dim -1: 4 = 4 (match)
  Result: (3, 5, 4)
```

### Diagnosing failures

`explain_broadcast` also works on incompatible shapes, showing where the conflict is:

```python
print(explain_broadcast((3, 4), (3, 5)))
```

```
Broadcasting (3, 4) with (3, 5):
  Step 1: Align shapes from right
    (3, 4)
    (3, 5)
  Step 2: Compare dimensions
    dim -2: 3 = 3 (match)
    dim -1: 4, 5 (INCOMPATIBLE)
  Error: Cannot broadcast - incompatible dimensions
```

## Multiple shapes

Both functions accept any number of shapes:

```python
broadcast_shape((3, 1, 1), (1, 4, 1), (1, 1, 5))  # (3, 4, 5)
explain_broadcast((3, 1, 1), (1, 4, 1), (1, 1, 5))
```
