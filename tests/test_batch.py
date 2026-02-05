"""
Tests for Batch dimension.
"""

import pytest

from shapeguard import Batch, Dim, expects
from shapeguard.errors import UnificationError
from tests.conftest import requires_numpy


class TestBatch:
    """Tests for the Batch class."""

    def test_batch_default_name(self):
        B = Batch()
        assert B.name == "batch"

    def test_batch_custom_name(self):
        B = Batch("B")
        assert B.name == "B"

    def test_batch_is_dim(self):
        B = Batch()
        assert isinstance(B, Dim)

    def test_batch_repr(self):
        B = Batch()
        assert repr(B) == "batch"

    @requires_numpy
    def test_batch_unifies_within_call(self, np_array):
        """Batch dimension unifies across arguments in same call."""
        B = Batch()
        n, m = Dim("n"), Dim("m")

        @expects(x=(B, n), y=(B, m))
        def f(x, y):
            return x.shape[0], y.shape[0]

        # Same batch size: OK
        result = f(np_array((32, 10)), np_array((32, 20)))
        assert result == (32, 32)

    @requires_numpy
    def test_batch_conflict_raises(self, np_array):
        """Different batch sizes in same call raises UnificationError."""
        B = Batch()
        n, m = Dim("n"), Dim("m")

        @expects(x=(B, n), y=(B, m))
        def f(x, y):
            return x, y

        with pytest.raises(UnificationError) as exc_info:
            f(np_array((32, 10)), np_array((64, 20)))

        assert "batch" in str(exc_info.value)

    @requires_numpy
    def test_batch_different_calls_ok(self, np_array):
        """Different batch sizes across calls is OK."""
        B = Batch()
        n = Dim("n")

        @expects(x=(B, n))
        def f(x):
            return x.shape[0]

        # Each call can have different batch size
        assert f(np_array((32, 10))) == 32
        assert f(np_array((64, 10))) == 64
        assert f(np_array((1, 10))) == 1

    @requires_numpy
    def test_multiple_batch_dims(self, np_array):
        """Can use multiple Batch instances independently."""
        B1 = Batch("batch1")
        B2 = Batch("batch2")
        n = Dim("n")

        @expects(x=(B1, n), y=(B2, n))
        def f(x, y):
            return x.shape[0], y.shape[0]

        # Different Batch objects don't unify
        result = f(np_array((32, 10)), np_array((64, 10)))
        assert result == (32, 64)
