"""
Backend compatibility layer.

Provides unified interface for extracting shapes from NumPy, JAX,
and other array-like objects without importing them at module level.
"""

from __future__ import annotations

from typing import Any


def get_shape(x: Any) -> tuple[int, ...]:
    """
    Extract shape from any array-like object.

    Works with NumPy arrays, JAX arrays, PyTorch tensors, and any object
    with a `.shape` attribute that returns an iterable of integers.

    Args:
        x: Array-like object with a .shape attribute

    Returns:
        Shape as a tuple of integers

    Raises:
        TypeError: If x doesn't have a shape attribute
    """
    if not hasattr(x, "shape"):
        raise TypeError(
            f"Cannot get shape from {type(x).__name__!r}: "
            f"object has no 'shape' attribute"
        )

    # Convert to tuple of ints to handle JAX's traced shapes
    # and numpy's np.int64 dimension values
    return tuple(int(d) for d in x.shape)


def is_array(x: Any) -> bool:
    """
    Check if x is an array-like object.

    Returns True if x has both .shape and .dtype attributes,
    which is the common interface for NumPy, JAX, and PyTorch arrays.
    """
    return hasattr(x, "shape") and hasattr(x, "dtype")


def get_array_backend(x: Any) -> str:
    """
    Detect which array backend x belongs to.

    Returns:
        One of: "numpy", "jax", "torch", "unknown"
    """
    module = type(x).__module__

    if module.startswith("numpy"):
        return "numpy"
    elif module.startswith("jax"):
        return "jax"
    elif module.startswith("torch"):
        return "torch"
    else:
        return "unknown"
