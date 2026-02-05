"""
Shape specification parsing and matching.
"""

from __future__ import annotations

from typing import Any

from shapeguard.core import Dim, UnificationContext
from shapeguard.errors import (
    DimensionMismatchError,
    RankMismatchError,
    ShapeGuardError,
)
from shapeguard._compat import get_shape


# Type alias for shape specifications
# Each element can be: int (exact match), Dim (symbolic), or None (wildcard)
ShapeSpec = tuple[int | Dim | None, ...]


def match_shape(
    actual: tuple[int, ...],
    spec: ShapeSpec,
    ctx: UnificationContext,
    source: str,
) -> None:
    """
    Match an actual shape against a specification.

    Args:
        actual: The actual shape to check
        spec: The shape specification
        ctx: Unification context for tracking dimension bindings
        source: Description of where this shape came from (for error messages)

    Raises:
        RankMismatchError: If the number of dimensions doesn't match
        DimensionMismatchError: If a concrete dimension doesn't match
        UnificationError: If a symbolic dimension conflicts with prior binding
    """
    # Check rank
    if len(actual) != len(spec):
        raise RankMismatchError(
            expected_rank=len(spec),
            actual_rank=len(actual),
            expected_shape=spec,
            actual_shape=actual,
            bindings=ctx.format_bindings(),
        )

    # Check each dimension
    for i, (actual_dim, spec_dim) in enumerate(zip(actual, spec)):
        dim_source = f"{source}[{i}]"

        if spec_dim is None:
            # Wildcard: accept any value
            continue

        elif isinstance(spec_dim, Dim):
            # Symbolic dimension: unify with context
            ctx.bind(spec_dim, actual_dim, dim_source)

        elif isinstance(spec_dim, int):
            # Concrete dimension: must match exactly
            if actual_dim != spec_dim:
                raise DimensionMismatchError(
                    dim_index=i,
                    expected_value=spec_dim,
                    actual_value=actual_dim,
                    expected_shape=spec,
                    actual_shape=actual,
                    bindings=ctx.format_bindings(),
                )

        else:
            raise TypeError(
                f"Invalid spec element at position {i}: {spec_dim!r} "
                f"(expected int, Dim, or None)"
            )


def check_shape(
    x: Any,
    spec: ShapeSpec,
    name: str = "array",
    *,
    ctx: UnificationContext | None = None,
) -> UnificationContext:
    """
    Check that an array's shape matches a specification.

    This is the standalone shape checking function for use outside decorators.

    Args:
        x: Array-like object to check
        spec: Shape specification to match against
        name: Name to use in error messages
        ctx: Optional unification context (created if not provided)

    Returns:
        The unification context (useful for chaining checks)

    Raises:
        ShapeGuardError: If shape doesn't match specification

    Example:
        from shapeguard import check_shape, Dim

        n = Dim("n")
        check_shape(x, (n, 128), name="input")
        check_shape(y, (n, 64), name="output")  # n must match
    """
    if ctx is None:
        ctx = UnificationContext()

    actual = get_shape(x)

    try:
        match_shape(actual, spec, ctx, name)
    except ShapeGuardError as e:
        # Add name context to error
        e.argument = name
        raise

    return ctx


def format_spec(spec: ShapeSpec) -> str:
    """Format a shape spec for display in error messages."""

    def fmt_dim(d: int | Dim | None) -> str:
        if d is None:
            return "*"
        elif isinstance(d, Dim):
            return d.name
        else:
            return str(d)

    return "(" + ", ".join(fmt_dim(d) for d in spec) + ")"
