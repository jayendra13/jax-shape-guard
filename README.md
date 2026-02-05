# ShapeGuard

Runtime shape contracts and diagnostics for NumPy and JAX.

## Installation

```bash
pip install shapeguard
```

## Quick Start

```python
from shapeguard import Dim, expects

n, m, k = Dim("n"), Dim("m"), Dim("k")

@expects(a=(n, m), b=(m, k))
def matmul(a, b):
    return a @ b
```

When shapes don't match, you get clear errors:

```
ShapeGuardError:
  function: matmul
  argument: b
  expected: (m, k)
  actual:   (5, 7)
  reason:   dimension 'm' bound to 4 from a.shape[1], but got 5 from b.shape[0]
```

## License

MIT
