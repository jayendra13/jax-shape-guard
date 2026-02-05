"""
Tests for shapeguard.spec module.
"""

import pytest

from shapeguard.core import Dim, UnificationContext
from shapeguard.errors import (
    DimensionMismatchError,
    RankMismatchError,
    UnificationError,
)
from shapeguard.spec import check_shape, format_spec, match_shape
from tests.conftest import requires_numpy


class TestMatchShape:
    """Tests for the match_shape function."""

    def test_concrete_match(self):
        """Concrete dimensions should match exactly."""
        ctx = UnificationContext()
        match_shape((3, 4), (3, 4), ctx, "x")
        # No exception = success

    def test_concrete_mismatch(self):
        """Concrete dimension mismatch raises DimensionMismatchError."""
        ctx = UnificationContext()
        with pytest.raises(DimensionMismatchError) as exc_info:
            match_shape((3, 5), (3, 4), ctx, "x")

        err = exc_info.value
        assert err.actual == (3, 5)

    def test_rank_mismatch(self):
        """Different ranks raise RankMismatchError."""
        ctx = UnificationContext()
        with pytest.raises(RankMismatchError) as exc_info:
            match_shape((3, 4, 5), (3, 4), ctx, "x")

        err = exc_info.value
        assert "rank 3" in str(err)
        assert "rank 2" in str(err)

    def test_symbolic_binds(self):
        """Symbolic dimensions bind to actual values."""
        ctx = UnificationContext()
        n = Dim("n")
        match_shape((42,), (n,), ctx, "x")
        assert ctx.resolve(n) == 42

    def test_symbolic_unifies(self):
        """Same symbolic dim must match across dimensions."""
        ctx = UnificationContext()
        n = Dim("n")

        # First match binds n=3
        match_shape((3, 3), (n, n), ctx, "x")

        # n is now bound to 3
        assert ctx.resolve(n) == 3

    def test_symbolic_conflict(self):
        """Same symbolic dim with different values raises UnificationError."""
        ctx = UnificationContext()
        n = Dim("n")

        with pytest.raises(UnificationError):
            match_shape((3, 4), (n, n), ctx, "x")

    def test_wildcard_accepts_any(self):
        """None (wildcard) accepts any value."""
        ctx = UnificationContext()
        match_shape((3, 999, 4), (3, None, 4), ctx, "x")
        # No exception = success

    def test_mixed_spec(self):
        """Mix of concrete, symbolic, and wildcard."""
        ctx = UnificationContext()
        n = Dim("n")

        match_shape((5, 128, 999), (n, 128, None), ctx, "x")

        assert ctx.resolve(n) == 5


class TestCheckShape:
    """Tests for the check_shape standalone function."""

    @requires_numpy
    def test_check_shape_passes(self, np_array):
        """check_shape returns context on success."""
        n = Dim("n")
        x = np_array((10, 20))
        ctx = check_shape(x, (n, 20), name="input")
        assert ctx.resolve(n) == 10

    @requires_numpy
    def test_check_shape_fails(self, np_array):
        """check_shape raises on mismatch."""
        x = np_array((10, 20))
        with pytest.raises(DimensionMismatchError):
            check_shape(x, (10, 30), name="input")

    @requires_numpy
    def test_check_shape_shared_context(self, np_array):
        """check_shape can use shared context for multiple checks."""
        n = Dim("n")
        x = np_array((10, 20))
        y = np_array((10, 30))

        ctx = check_shape(x, (n, 20), name="x")
        check_shape(y, (n, 30), name="y", ctx=ctx)

        assert ctx.resolve(n) == 10

    @requires_numpy
    def test_check_shape_conflict_in_shared_context(self, np_array):
        """Shared context catches dimension conflicts across arrays."""
        n = Dim("n")
        x = np_array((10, 20))
        y = np_array((15, 30))  # Different first dim

        ctx = check_shape(x, (n, 20), name="x")

        with pytest.raises(UnificationError):
            check_shape(y, (n, 30), name="y", ctx=ctx)

    def test_check_shape_non_array_raises(self):
        """check_shape raises TypeError for non-array input."""
        with pytest.raises(TypeError):
            check_shape([1, 2, 3], (3,), name="list")


class TestFormatSpec:
    """Tests for the format_spec function."""

    def test_format_concrete(self):
        assert format_spec((3, 4)) == "(3, 4)"

    def test_format_symbolic(self):
        n = Dim("n")
        m = Dim("m")
        assert format_spec((n, m)) == "(n, m)"

    def test_format_wildcard(self):
        assert format_spec((3, None, 4)) == "(3, *, 4)"

    def test_format_mixed(self):
        n = Dim("batch")
        assert format_spec((n, 128, None)) == "(batch, 128, *)"

    def test_format_empty(self):
        assert format_spec(()) == "()"
