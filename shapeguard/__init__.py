"""
ShapeGuard: Runtime shape contracts and diagnostics for NumPy and JAX.

Basic usage:
    from shapeguard import Dim, expects, check_shape

    n, m = Dim("n"), Dim("m")

    @expects(x=(n, m), y=(m,))
    def forward(x, y):
        return x @ y

ML workflows:
    from shapeguard import Batch, ShapeContext

    B = Batch()

    @expects(x=(B, n, m))
    def layer(x): ...

    # Ellipsis for variable leading dims
    @expects(x=(..., n, m))
    def normalize(x): ...

    # Grouped checks
    with ShapeContext() as ctx:
        ctx.check(x, (n, m), "x")
        ctx.check(y, (m, k), "y")
"""

from shapeguard.core import Dim, UnificationContext, Batch
from shapeguard.errors import ShapeGuardError
from shapeguard.decorator import expects
from shapeguard.spec import check_shape
from shapeguard.context import ShapeContext
from shapeguard.config import config

__version__ = "0.2.0a1"

__all__ = [
    # Core
    "Dim",
    "Batch",
    "UnificationContext",
    # Validation
    "expects",
    "check_shape",
    "ShapeContext",
    # Configuration
    "config",
    # Errors
    "ShapeGuardError",
]
