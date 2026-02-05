"""
Context manager for grouped shape checks.
"""

from __future__ import annotations

from typing import Any

from shapeguard.core import UnificationContext
from shapeguard.spec import ShapeSpec, match_shape
from shapeguard.errors import ShapeGuardError
from shapeguard._compat import get_shape


class ShapeContext:
    """
    Context manager for grouped shape checks with shared bindings.

    Use this when you need to validate multiple arrays share dimensions
    but aren't in a single decorated function.

    Example:
        from shapeguard import ShapeContext, Dim

        n, m, k = Dim("n"), Dim("m"), Dim("k")

        with ShapeContext() as ctx:
            ctx.check(x, (n, m), "x")
            ctx.check(y, (m, k), "y")
            ctx.check(z, (n, k), "z")

        # All checks passed, dimensions unified:
        print(ctx.bindings)  # {n: 3, m: 4, k: 5}

    Can also be used without context manager:
        ctx = ShapeContext()
        ctx.check(x, (n, m), "x")
        ctx.check(y, (m, k), "y")
    """

    def __init__(self) -> None:
        self._ctx = UnificationContext()

    def __enter__(self) -> ShapeContext:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # No cleanup needed
        pass

    def check(
        self,
        x: Any,
        spec: ShapeSpec,
        name: str = "array",
    ) -> ShapeContext:
        """
        Check that an array's shape matches a specification.

        Args:
            x: Array-like object to check
            spec: Shape specification to match against
            name: Name to use in error messages

        Returns:
            self (for method chaining)

        Raises:
            ShapeGuardError: If shape doesn't match specification
        """
        actual = get_shape(x)

        try:
            match_shape(actual, spec, self._ctx, name)
        except ShapeGuardError as e:
            e.argument = name
            raise

        return self

    @property
    def bindings(self) -> dict[str, int]:
        """
        Get current dimension bindings as a name -> value dict.

        Useful for inspecting what values dimensions were bound to.
        """
        return {
            dim.name: binding.value
            for dim, binding in self._ctx.bindings.items()
        }

    def resolve(self, dim: Any) -> int | None:
        """
        Get the bound value for a dimension.

        Args:
            dim: A Dim object

        Returns:
            The bound integer value, or None if unbound
        """
        return self._ctx.resolve(dim)

    def format_bindings(self) -> str:
        """Format current bindings for display."""
        return self._ctx.format_bindings()
