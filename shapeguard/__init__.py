"""
ShapeGuard: Runtime shape contracts and diagnostics for NumPy and JAX.

Basic usage:
    from shapeguard import Dim, expects, check_shape

    n, m = Dim("n"), Dim("m")

    @expects(x=(n, m), y=(m,))
    def forward(x, y):
        return x @ y
"""

from shapeguard.core import Dim, UnificationContext
from shapeguard.errors import ShapeGuardError
from shapeguard.decorator import expects
from shapeguard.spec import check_shape

__version__ = "0.1.0a1"

__all__ = [
    "Dim",
    "UnificationContext",
    "ShapeGuardError",
    "expects",
    "check_shape",
]
