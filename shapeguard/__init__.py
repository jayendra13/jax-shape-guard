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

from shapeguard.broadcast import broadcast_shape, explain_broadcast
from shapeguard.config import config
from shapeguard.context import ShapeContext
from shapeguard.core import Batch, Dim, UnificationContext
from shapeguard.decorator import contract, ensures, expects
from shapeguard.errors import BroadcastError, OutputShapeError, ShapeGuardError
from shapeguard.spec import check_shape

__version__ = "0.3.0"

__all__ = [
    # Core
    "Dim",
    "Batch",
    "UnificationContext",
    # Validation
    "expects",
    "ensures",
    "contract",
    "check_shape",
    "ShapeContext",
    # Broadcasting
    "broadcast_shape",
    "explain_broadcast",
    # Configuration
    "config",
    # Errors
    "ShapeGuardError",
    "OutputShapeError",
    "BroadcastError",
]
