"""
Tests for ShapeContext manager.
"""

import pytest

from shapeguard import Dim, ShapeContext
from shapeguard.errors import RankMismatchError, UnificationError
from tests.conftest import requires_numpy


class TestShapeContext:
    """Tests for the ShapeContext class."""

    @requires_numpy
    def test_basic_usage(self, np_array):
        """Basic context manager usage."""
        n, m = Dim("n"), Dim("m")

        with ShapeContext() as ctx:
            ctx.check(np_array((3, 4)), (n, m), "x")

        assert ctx.bindings == {"n": 3, "m": 4}

    @requires_numpy
    def test_multiple_checks(self, np_array):
        """Multiple checks with shared context."""
        n, m, k = Dim("n"), Dim("m"), Dim("k")

        with ShapeContext() as ctx:
            ctx.check(np_array((3, 4)), (n, m), "x")
            ctx.check(np_array((4, 5)), (m, k), "y")
            ctx.check(np_array((3, 5)), (n, k), "z")

        assert ctx.bindings == {"n": 3, "m": 4, "k": 5}

    @requires_numpy
    def test_conflict_raises(self, np_array):
        """Conflicting dimensions raise UnificationError."""
        n, m = Dim("n"), Dim("m")

        with pytest.raises(UnificationError):
            with ShapeContext() as ctx:
                ctx.check(np_array((3, 4)), (n, m), "x")
                ctx.check(np_array((5, 4)), (n, m), "y")  # n=3 vs n=5

    @requires_numpy
    def test_without_context_manager(self, np_array):
        """Can use without context manager."""
        n, m = Dim("n"), Dim("m")

        ctx = ShapeContext()
        ctx.check(np_array((3, 4)), (n, m), "x")
        ctx.check(np_array((4, 5)), (m, Dim("k")), "y")

        assert ctx.bindings["n"] == 3
        assert ctx.bindings["m"] == 4

    @requires_numpy
    def test_method_chaining(self, np_array):
        """check() returns self for chaining."""
        n, m, k = Dim("n"), Dim("m"), Dim("k")

        ctx = (
            ShapeContext()
            .check(np_array((3, 4)), (n, m), "x")
            .check(np_array((4, 5)), (m, k), "y")
        )

        assert ctx.bindings == {"n": 3, "m": 4, "k": 5}

    @requires_numpy
    def test_resolve(self, np_array):
        """resolve() returns bound value for a Dim."""
        n = Dim("n")

        ctx = ShapeContext()
        ctx.check(np_array((42, 10)), (n, 10), "x")

        assert ctx.resolve(n) == 42

    @requires_numpy
    def test_resolve_unbound(self, np_array):
        """resolve() returns None for unbound Dim."""
        n = Dim("n")
        m = Dim("m")  # Never bound

        ctx = ShapeContext()
        ctx.check(np_array((42,)), (n,), "x")

        assert ctx.resolve(n) == 42
        assert ctx.resolve(m) is None

    @requires_numpy
    def test_format_bindings(self, np_array):
        """format_bindings() returns readable string."""
        n = Dim("n")

        ctx = ShapeContext()
        ctx.check(np_array((42,)), (n,), "x")

        formatted = ctx.format_bindings()
        assert "n=42" in formatted
        assert "x[0]" in formatted

    @requires_numpy
    def test_rank_mismatch(self, np_array):
        """Rank mismatch raises RankMismatchError."""
        n, m = Dim("n"), Dim("m")

        with pytest.raises(RankMismatchError):
            with ShapeContext() as ctx:
                ctx.check(np_array((3,)), (n, m), "x")  # 1D vs 2D spec

    @requires_numpy
    def test_with_ellipsis(self, np_array):
        """ShapeContext works with ellipsis specs."""
        n = Dim("n")

        with ShapeContext() as ctx:
            ctx.check(np_array((2, 3, 4)), (..., n), "x")
            ctx.check(np_array((5, 4)), (..., n), "y")

        assert ctx.bindings == {"n": 4}

    @requires_numpy
    def test_real_world_example(self, np_array):
        """Real-world ML validation example."""
        B, T, D, V = Dim("batch"), Dim("time"), Dim("dim"), Dim("vocab")

        with ShapeContext() as ctx:
            # Validate transformer inputs
            ctx.check(np_array((32, 100, 512)), (B, T, D), "embeddings")
            ctx.check(np_array((32, 100)), (B, T), "attention_mask")
            ctx.check(np_array((32, 100, 50000)), (B, T, V), "logits")

        assert ctx.bindings == {
            "batch": 32,
            "time": 100,
            "dim": 512,
            "vocab": 50000,
        }
