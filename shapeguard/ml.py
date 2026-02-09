"""
ML-specific helpers for common shape patterns.

Pre-defined dimensions, attention shape specs, and convolution output calculators.

Usage:
    from shapeguard.ml import B, T, D, attention_shapes, conv_output_shape

    @expects(x=(B, T, D))
    def transformer_layer(x): ...
"""

from __future__ import annotations

import math

from shapeguard.core import Batch, Dim

# ---------------------------------------------------------------------------
# Pre-defined dimensions
# ---------------------------------------------------------------------------

B = Batch("B")  # batch
T = Dim("T")  # sequence / time
C = Dim("C")  # channels
H = Dim("H")  # height
W = Dim("W")  # width
D = Dim("D")  # feature / embedding


# ---------------------------------------------------------------------------
# Attention shapes
# ---------------------------------------------------------------------------


def attention_shapes(
    B: Dim | int,
    heads: Dim | int,
    seq_q: Dim | int,
    seq_k: Dim | int,
    d_k: Dim | int,
) -> dict[str, tuple[Dim | int, ...]]:
    """
    Return shape specs for multi-head attention Q, K, V tensors.

    Args:
        B: Batch dimension (Dim or int).
        heads: Number of attention heads.
        seq_q: Query sequence length.
        seq_k: Key/value sequence length.
        d_k: Head dimension.

    Returns:
        Dict with keys ``"q"``, ``"k"``, ``"v"`` mapping to shape tuples,
        suitable for unpacking into ``@expects(**attention_shapes(...))``.
    """
    return {
        "q": (B, heads, seq_q, d_k),
        "k": (B, heads, seq_k, d_k),
        "v": (B, heads, seq_k, d_k),
    }


# ---------------------------------------------------------------------------
# Convolution output shape
# ---------------------------------------------------------------------------


def _to_tuple(value: int | tuple[int, ...], n: int, name: str) -> tuple[int, ...]:
    """Broadcast a scalar int to an n-tuple, or validate tuple length."""
    if isinstance(value, int):
        return (value,) * n
    if len(value) != n:
        msg = f"{name} has length {len(value)}, expected {n} (number of spatial dims)"
        raise ValueError(msg)
    return value


def conv_output_shape(
    input: tuple[int, ...],
    kernel: int | tuple[int, ...],
    stride: int | tuple[int, ...] = 1,
    padding: int | tuple[int, ...] = 0,
) -> tuple[int, ...]:
    """
    Compute the output shape of a convolution.

    Applies ``floor((input_size + 2*padding - kernel_size) / stride) + 1``
    to each spatial dimension, preserving batch and channel dims.

    Args:
        input: Input shape as ``(batch, channels, *spatial)``.
        kernel: Kernel size — int (broadcast) or tuple per spatial dim.
        stride: Stride — int (broadcast) or tuple per spatial dim.
        padding: Padding — int (broadcast) or tuple per spatial dim.

    Returns:
        Output shape tuple ``(batch, channels, *output_spatial)``.

    Raises:
        ValueError: If input has fewer than 3 dims, tuple lengths don't match
            the number of spatial dims, or any output spatial dim is non-positive.
    """
    if len(input) < 3:
        msg = f"input must have at least 3 dims (batch, channels, *spatial), got {len(input)}"
        raise ValueError(msg)

    batch, channels = input[0], input[1]
    spatial = input[2:]
    n = len(spatial)

    kernel_t = _to_tuple(kernel, n, "kernel")
    stride_t = _to_tuple(stride, n, "stride")
    padding_t = _to_tuple(padding, n, "padding")

    out_spatial: list[int] = []
    for i in range(n):
        out = math.floor((spatial[i] + 2 * padding_t[i] - kernel_t[i]) / stride_t[i]) + 1
        if out <= 0:
            msg = (
                f"non-positive output size {out} at spatial dim {i}: "
                f"input={spatial[i]}, kernel={kernel_t[i]}, "
                f"stride={stride_t[i]}, padding={padding_t[i]}"
            )
            raise ValueError(msg)
        out_spatial.append(out)

    return (batch, channels, *out_spatial)


__all__ = [
    "B",
    "T",
    "C",
    "H",
    "W",
    "D",
    "attention_shapes",
    "conv_output_shape",
]
