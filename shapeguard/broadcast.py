"""Broadcasting utilities for shape validation."""

from __future__ import annotations

from typing import Any

from shapeguard._compat import get_shape, is_array
from shapeguard.errors import BroadcastError


def _normalize_shape(shape_or_array: tuple[int, ...] | Any) -> tuple[int, ...]:
    """Convert array-like or tuple to a shape tuple."""
    if isinstance(shape_or_array, tuple):
        return shape_or_array
    if is_array(shape_or_array):
        return get_shape(shape_or_array)
    # Try to convert to tuple (handles lists, etc.)
    try:
        return tuple(shape_or_array)
    except TypeError as err:
        raise TypeError(f"Cannot interpret {type(shape_or_array).__name__!r} as a shape") from err


def _broadcast_two_shapes(
    shape1: tuple[int, ...],
    shape2: tuple[int, ...],
) -> tuple[int, ...]:
    """
    Broadcast two shapes following NumPy rules.

    Raises BroadcastError on incompatibility.
    """
    # Align shapes from the right by padding the shorter one with 1s
    len1, len2 = len(shape1), len(shape2)
    max_len = max(len1, len2)

    # Pad shapes on the left with 1s
    padded1 = (1,) * (max_len - len1) + shape1
    padded2 = (1,) * (max_len - len2) + shape2

    result = []
    for i, (d1, d2) in enumerate(zip(padded1, padded2, strict=True)):
        if d1 == d2:
            result.append(d1)
        elif d1 == 1:
            result.append(d2)
        elif d2 == 1:
            result.append(d1)
        else:
            # Incompatible dimensions
            # Convert index to negative (from right) for clearer error
            dim_index = i - max_len
            raise BroadcastError(
                shapes=[shape1, shape2],
                dim_index=dim_index,
                dim_values=[d1, d2],
            )

    return tuple(result)


def broadcast_shape(*shapes: tuple[int, ...] | Any) -> tuple[int, ...]:
    """
    Compute the result shape from broadcasting multiple shapes.

    Follows NumPy broadcasting rules:
    - Align shapes from the right
    - Dimensions match if equal or one is 1
    - Missing dims on left treated as 1

    Args:
        *shapes: Shapes as tuples or array-like objects

    Returns:
        The broadcast result shape

    Raises:
        BroadcastError: If shapes are not broadcast-compatible
        ValueError: If no shapes are provided

    Example:
        ```python
        >>> broadcast_shape((3, 1), (1, 4))
        (3, 4)
        >>> broadcast_shape((2, 3, 4), (4,))
        (2, 3, 4)
        >>> broadcast_shape(arr1, arr2)  # works with arrays too
        (3, 4)
        ```
    """
    if not shapes:
        raise ValueError("broadcast_shape requires at least one shape")

    # Normalize all inputs to tuples
    normalized = [_normalize_shape(s) for s in shapes]

    # Start with first shape and broadcast pairwise
    result = normalized[0]
    for shape in normalized[1:]:
        try:
            result = _broadcast_two_shapes(result, shape)
        except BroadcastError as e:
            # Re-raise with all original shapes for better error message
            raise BroadcastError(
                shapes=[_normalize_shape(s) for s in shapes],
                dim_index=e.dim_index,
                dim_values=e.dim_values,
            ) from None

    return result


def explain_broadcast(*shapes: tuple[int, ...] | Any) -> str:
    """
    Return human-readable explanation of broadcast operation.

    Provides step-by-step breakdown of how broadcasting works,
    useful for debugging shape mismatches.

    Args:
        *shapes: Shapes as tuples or array-like objects

    Returns:
        Multi-line string explaining the broadcast process

    Example:
        ```python
        >>> print(explain_broadcast((3, 1, 4), (5, 4)))
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
    """
    if not shapes:
        return "No shapes provided"

    # Normalize all inputs
    normalized = [_normalize_shape(s) for s in shapes]

    if len(normalized) == 1:
        return f"Single shape {normalized[0]}, no broadcasting needed"

    lines = []

    # Header
    shape_strs = [str(s) for s in normalized]
    lines.append(f"Broadcasting {' with '.join(shape_strs)}:")

    # Find max length for alignment display
    max_len = max(len(s) for s in normalized)

    # Step 1: Show alignment
    lines.append("  Step 1: Align shapes from right")
    for shape in normalized:
        # Create padded representation
        padding = max_len - len(shape)
        if padding > 0:
            padded_str = "(" + "   " * padding + ", ".join(str(d) for d in shape) + ")"
        else:
            padded_str = "(" + ", ".join(str(d) for d in shape) + ")"
        lines.append(f"    {padded_str}")

    # Step 2: Compare dimensions
    lines.append("  Step 2: Compare dimensions")

    # Pad all shapes
    padded_shapes = []
    for shape in normalized:
        padded = (1,) * (max_len - len(shape)) + shape
        padded_shapes.append(padded)

    result_dims: list[int | None] = []
    error_at: int | None = None

    for i in range(max_len):
        dim_index = i - max_len  # Negative index from right
        dims = [s[i] for s in padded_shapes]
        unique_dims = set(dims)
        non_one_dims = [d for d in unique_dims if d != 1]

        if len(non_one_dims) == 0:
            # All 1s
            result_dims.append(1)
            lines.append(f"    dim {dim_index}: 1 = 1 (match)")
        elif len(non_one_dims) == 1:
            result_val = non_one_dims[0]
            result_dims.append(result_val)

            if 1 in unique_dims:
                # Broadcasting happened
                lines.append(f"    dim {dim_index}: 1 → {result_val} (broadcast)")
            else:
                # All same non-one value
                lines.append(f"    dim {dim_index}: {result_val} = {result_val} (match)")
        else:
            # Incompatible
            error_at = i
            result_dims.append(None)
            dim_str = ", ".join(str(d) for d in dims)
            lines.append(f"    dim {dim_index}: {dim_str} (INCOMPATIBLE)")

    # Result
    if error_at is not None:
        lines.append("  Error: Cannot broadcast - incompatible dimensions")
    else:
        result_str = "(" + ", ".join(str(d) for d in result_dims) + ")"
        lines.append(f"  Result: {result_str}")

    return "\n".join(lines)
