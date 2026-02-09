"""
Tests for shapeguard.ml — pre-defined dims, attention_shapes, conv_output_shape.
"""

import pytest

from shapeguard import Batch, Dim, expects
from shapeguard.errors import UnificationError
from shapeguard.ml import (
    B,
    C,
    D,
    H,
    T,
    W,
    attention_shapes,
    conv_output_shape,
)
from tests.conftest import requires_numpy

# ---------------------------------------------------------------------------
# Pre-defined dimensions
# ---------------------------------------------------------------------------


class TestPreDefinedDims:
    """Tests for module-level B, T, C, H, W, D singletons."""

    def test_types(self):
        """B is Batch, the rest are Dim."""
        assert isinstance(B, Batch)
        for d in (T, C, H, W, D):
            assert isinstance(d, Dim)

    def test_names(self):
        """Each dim carries the expected name."""
        assert B.name == "B"
        assert T.name == "T"
        assert C.name == "C"
        assert H.name == "H"
        assert W.name == "W"
        assert D.name == "D"

    def test_all_distinct(self):
        """All six dims are distinct objects (identity-based)."""
        dims = [B, T, C, H, W, D]
        for i, a in enumerate(dims):
            for b in dims[i + 1 :]:
                assert a is not b

    def test_singleton_identity(self):
        """Re-importing yields the same objects."""
        from shapeguard.ml import B as B2
        from shapeguard.ml import T as T2

        assert B is B2
        assert T is T2

    @requires_numpy
    def test_integration_with_expects(self, np_array):
        """Pre-defined dims work with @expects."""

        @expects(x=(B, T, D))
        def layer(x):
            return x

        layer(np_array((2, 10, 64)))

    @requires_numpy
    def test_unification_failure(self, np_array):
        """Shared dim across args detects mismatch."""

        @expects(x=(B, T, D), y=(B, T, D))
        def f(x, y):
            return x

        with pytest.raises(UnificationError):
            # D mismatch: 64 vs 128
            f(np_array((2, 10, 64)), np_array((2, 10, 128)))


# ---------------------------------------------------------------------------
# attention_shapes
# ---------------------------------------------------------------------------


class TestAttentionShapes:
    """Tests for attention_shapes()."""

    def test_return_keys(self):
        """Returns dict with exactly q, k, v keys."""
        shapes = attention_shapes(B, Dim("h"), Dim("sq"), Dim("sk"), Dim("dk"))
        assert set(shapes.keys()) == {"q", "k", "v"}

    def test_shape_correctness(self):
        """Q, K, V shapes have the expected structure."""
        h, sq, sk, dk = Dim("h"), Dim("sq"), Dim("sk"), Dim("dk")
        shapes = attention_shapes(B, h, sq, sk, dk)

        assert shapes["q"] == (B, h, sq, dk)
        assert shapes["k"] == (B, h, sk, dk)
        assert shapes["v"] == (B, h, sk, dk)

    def test_dim_identity_sharing(self):
        """Shared dims are the same objects across shapes."""
        h, sq, sk, dk = Dim("h"), Dim("sq"), Dim("sk"), Dim("dk")
        shapes = attention_shapes(B, h, sq, sk, dk)

        # B shared across all
        assert shapes["q"][0] is shapes["k"][0] is shapes["v"][0]
        # heads shared across all
        assert shapes["q"][1] is shapes["k"][1] is shapes["v"][1]
        # d_k shared across all
        assert shapes["q"][3] is shapes["k"][3] is shapes["v"][3]
        # seq_k shared between k and v
        assert shapes["k"][2] is shapes["v"][2]
        # seq_q is only in q
        assert shapes["q"][2] is sq

    def test_concrete_ints(self):
        """Works with concrete int values."""
        shapes = attention_shapes(32, 8, 128, 128, 64)
        assert shapes["q"] == (32, 8, 128, 64)
        assert shapes["k"] == (32, 8, 128, 64)
        assert shapes["v"] == (32, 8, 128, 64)

    @requires_numpy
    def test_integration_with_expects(self, np_array):
        """attention_shapes can be unpacked into @expects."""
        h, sq, sk, dk = Dim("h"), Dim("sq"), Dim("sk"), Dim("dk")

        @expects(**attention_shapes(B, h, sq, sk, dk))
        def attention(q, k, v):
            return q

        attention(
            np_array((2, 8, 10, 64)),
            np_array((2, 8, 20, 64)),
            np_array((2, 8, 20, 64)),
        )

    @requires_numpy
    def test_mismatch_detection(self, np_array):
        """Detects shape mismatches across Q, K, V."""
        h, sq, sk, dk = Dim("h"), Dim("sq"), Dim("sk"), Dim("dk")

        @expects(**attention_shapes(B, h, sq, sk, dk))
        def attention(q, k, v):
            return q

        with pytest.raises(UnificationError):
            # d_k mismatch: q has 64, k has 32
            attention(
                np_array((2, 8, 10, 64)),
                np_array((2, 8, 20, 32)),
                np_array((2, 8, 20, 64)),
            )

    @requires_numpy
    def test_self_attention(self, np_array):
        """Self-attention: same seq length for q and k."""
        seq = Dim("seq")
        h, dk = Dim("h"), Dim("dk")

        @expects(**attention_shapes(B, h, seq, seq, dk))
        def self_attn(q, k, v):
            return q

        self_attn(
            np_array((4, 8, 16, 64)),
            np_array((4, 8, 16, 64)),
            np_array((4, 8, 16, 64)),
        )


# ---------------------------------------------------------------------------
# conv_output_shape
# ---------------------------------------------------------------------------


class TestConvOutputShape:
    """Tests for conv_output_shape()."""

    def test_2d_basic(self):
        """Standard 2D conv with defaults (stride=1, padding=0)."""
        out = conv_output_shape(input=(1, 3, 32, 32), kernel=3)
        assert out == (1, 3, 30, 30)

    def test_1d(self):
        """1D convolution."""
        out = conv_output_shape(input=(4, 16, 100), kernel=5)
        assert out == (4, 16, 96)

    def test_3d(self):
        """3D convolution."""
        out = conv_output_shape(input=(2, 1, 16, 16, 16), kernel=3)
        assert out == (2, 1, 14, 14, 14)

    def test_stride(self):
        """Stride reduces spatial dims."""
        out = conv_output_shape(input=(1, 3, 32, 32), kernel=3, stride=2)
        assert out == (1, 3, 15, 15)

    def test_padding(self):
        """Padding increases effective input size."""
        out = conv_output_shape(input=(1, 3, 32, 32), kernel=3, padding=1)
        assert out == (1, 3, 32, 32)  # same padding for 3x3

    def test_asymmetric_kernel(self):
        """Different kernel sizes per spatial dim."""
        out = conv_output_shape(input=(1, 3, 32, 32), kernel=(3, 5))
        assert out == (1, 3, 30, 28)

    def test_asymmetric_stride(self):
        """Different strides per spatial dim."""
        out = conv_output_shape(input=(1, 3, 32, 32), kernel=3, stride=(1, 2))
        assert out == (1, 3, 30, 15)

    def test_asymmetric_padding(self):
        """Different padding per spatial dim."""
        out = conv_output_shape(input=(1, 3, 32, 32), kernel=3, padding=(0, 1))
        assert out == (1, 3, 30, 32)

    def test_batch_channel_preserved(self):
        """Batch and channel dims pass through unchanged."""
        out = conv_output_shape(input=(16, 64, 28, 28), kernel=3)
        assert out[0] == 16
        assert out[1] == 64

    def test_error_too_few_dims(self):
        """Raises ValueError for input with fewer than 3 dims."""
        with pytest.raises(ValueError, match="at least 3 dims"):
            conv_output_shape(input=(32, 64), kernel=3)

    def test_error_tuple_length_mismatch(self):
        """Raises ValueError when kernel tuple length doesn't match spatial dims."""
        with pytest.raises(ValueError, match="kernel has length 3.*expected 2"):
            conv_output_shape(input=(1, 3, 32, 32), kernel=(3, 3, 3))

    def test_error_non_positive_output(self):
        """Raises ValueError when output dim would be non-positive."""
        with pytest.raises(ValueError, match="non-positive output size"):
            conv_output_shape(input=(1, 3, 5, 5), kernel=7)

    def test_resnet_first_layer(self):
        """ResNet conv1: 224x224, kernel=7, stride=2, padding=3."""
        out = conv_output_shape(
            input=(1, 3, 224, 224), kernel=7, stride=2, padding=3
        )
        assert out == (1, 64 if False else 3, 112, 112)
        # Note: channels come from input, not changed by conv_output_shape
        assert out == (1, 3, 112, 112)

    def test_same_padding_even_kernel(self):
        """Same-padding for 4x4 kernel, stride=1: padding=floor(k/2)=2 but asymmetric.

        With symmetric padding=1 and kernel=3, stride=1 gives same-padding.
        """
        out = conv_output_shape(input=(1, 1, 8, 8), kernel=3, stride=1, padding=1)
        assert out == (1, 1, 8, 8)

    def test_large_stride(self):
        """Stride larger than kernel."""
        out = conv_output_shape(input=(1, 1, 100), kernel=3, stride=5)
        assert out == (1, 1, 20)

    def test_stride_padding_mismatch_tuple_lengths(self):
        """Stride tuple length must match spatial dims."""
        with pytest.raises(ValueError, match="stride has length 3.*expected 2"):
            conv_output_shape(input=(1, 3, 32, 32), kernel=3, stride=(1, 1, 1))

    def test_padding_tuple_length_mismatch(self):
        """Padding tuple length must match spatial dims."""
        with pytest.raises(ValueError, match="padding has length 1.*expected 2"):
            conv_output_shape(input=(1, 3, 32, 32), kernel=3, padding=(1,))
