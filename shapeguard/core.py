"""
Core abstractions: Dim and UnificationContext.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class Dim:
    """
    Symbolic dimension that unifies at runtime.

    Two Dim objects unify if and only if they are the same object.
    This enforces explicit dimension sharing:

        n = Dim("n")
        m = Dim("m")

        @expects(x=(n, m), y=(m,))  # Same 'm' object = must match
        def f(x, y): ...

    The name is used for error messages only.
    """

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        # Identity-based equality
        return self is other

    def __hash__(self) -> int:
        return id(self)


class Batch(Dim):
    """
    Special dimension for batch sizes in ML workflows.

    Batch is a Dim that:
    - Has a default name "batch" (or custom)
    - Unifies across arguments within the same call
    - Each function call can have a different batch size

    Example:
        ```python
        B = Batch()

        @expects(x=(B, n, m), y=(B, m, k))
        def layer(x, y):
            # B unifies: x and y must have same batch size
            ...

        layer(x_32, y_32)  # B=32 for this call
        layer(x_64, y_64)  # B=64 for this call (different, OK)
        ```
    """

    __slots__ = ()

    def __init__(self, name: str = "batch") -> None:
        super().__init__(name)


# Sentinel for ellipsis in shape specs
class _EllipsisType:
    """Sentinel for variable-length leading dimensions in shape specs."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "..."

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _EllipsisType) or other is ...

    def __hash__(self) -> int:
        return hash(...)


# Singleton ellipsis instance for shape specs
# Use this OR Python's ... (Ellipsis) in specs
ELLIPSIS = _EllipsisType()


@dataclass
class Binding:
    """Record of a dimension binding with source info for error messages."""

    value: int
    source: str  # e.g., "x.shape[1]"


class UnificationContext:
    """
    Tracks dimension bindings during shape checking.

    Maintains a mapping from Dim objects to their bound integer values,
    along with source information for error messages.
    """

    bindings: dict[Dim, Binding] = field(default_factory=dict)

    def __init__(self) -> None:
        self.bindings: dict[Dim, Binding] = {}

    def bind(self, dim: Dim, value: int, source: str) -> None:
        """
        Bind a dimension to a concrete value.

        Args:
            dim: The symbolic dimension to bind
            value: The concrete integer value
            source: Description of where this binding came from (e.g., "x.shape[0]")

        Raises:
            UnificationError: If dim is already bound to a different value
        """
        from shapeguard.errors import UnificationError

        if dim in self.bindings:
            existing = self.bindings[dim]
            if existing.value != value:
                raise UnificationError(
                    dim=dim,
                    expected_value=existing.value,
                    expected_source=existing.source,
                    actual_value=value,
                    actual_source=source,
                )
        else:
            self.bindings[dim] = Binding(value=value, source=source)

    def resolve(self, dim: Dim) -> int | None:
        """
        Get the bound value for a dimension, or None if unbound.
        """
        binding = self.bindings.get(dim)
        return binding.value if binding else None

    def get_binding_source(self, dim: Dim) -> str | None:
        """Get the source description for a dimension's binding."""
        binding = self.bindings.get(dim)
        return binding.source if binding else None

    def format_bindings(self) -> str:
        """Format current bindings for error messages."""
        if not self.bindings:
            return "{}"
        parts = [f"{dim.name}={b.value} (from {b.source})" for dim, b in self.bindings.items()]
        return "{" + ", ".join(parts) + "}"
