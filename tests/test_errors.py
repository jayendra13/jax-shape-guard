"""
Tests for shapeguard.errors module.
"""

import pytest

from shapeguard.core import Dim
from shapeguard.errors import (
    ShapeGuardError,
    UnificationError,
    RankMismatchError,
    DimensionMismatchError,
)


class TestShapeGuardError:
    """Tests for the base ShapeGuardError class."""

    def test_basic_error(self):
        err = ShapeGuardError("test message")
        assert "ShapeGuardError" in str(err)

    def test_error_with_all_fields(self):
        err = ShapeGuardError(
            "test",
            function="matmul",
            argument="x",
            expected=(3, 4),
            actual=(3, 5),
            reason="dim[1] mismatch",
            bindings="{n=3}",
        )

        msg = str(err)
        assert "matmul" in msg
        assert "x" in msg
        assert "(3, 4)" in msg
        assert "(3, 5)" in msg
        assert "dim[1] mismatch" in msg
        assert "{n=3}" in msg

    def test_error_attributes(self):
        err = ShapeGuardError(
            "test",
            function="f",
            argument="y",
            expected=(10,),
            actual=(20,),
            reason="mismatch",
        )

        assert err.function == "f"
        assert err.argument == "y"
        assert err.expected == (10,)
        assert err.actual == (20,)
        assert err.reason == "mismatch"


class TestUnificationError:
    """Tests for UnificationError."""

    def test_unification_error_message(self):
        n = Dim("batch_size")
        err = UnificationError(
            dim=n,
            expected_value=32,
            expected_source="x.shape[0]",
            actual_value=64,
            actual_source="y.shape[0]",
        )

        msg = str(err)
        assert "batch_size" in msg
        assert "32" in msg
        assert "64" in msg
        assert "x.shape[0]" in msg
        assert "y.shape[0]" in msg

    def test_unification_error_attributes(self):
        n = Dim("n")
        err = UnificationError(
            dim=n,
            expected_value=10,
            expected_source="a[0]",
            actual_value=20,
            actual_source="b[0]",
        )

        assert err.dim is n
        assert err.expected_value == 10
        assert err.actual_value == 20
        assert err.expected_source == "a[0]"
        assert err.actual_source == "b[0]"


class TestRankMismatchError:
    """Tests for RankMismatchError."""

    def test_rank_error_message(self):
        n = Dim("n")
        err = RankMismatchError(
            function="layer",
            argument="input",
            expected_rank=3,
            actual_rank=2,
            expected_shape=(n, 128, 64),
            actual_shape=(128, 64),
        )

        msg = str(err)
        assert "layer" in msg
        assert "input" in msg
        assert "rank 3" in msg
        assert "rank 2" in msg


class TestDimensionMismatchError:
    """Tests for DimensionMismatchError."""

    def test_dim_error_message(self):
        err = DimensionMismatchError(
            function="conv",
            argument="x",
            dim_index=2,
            expected_value=224,
            actual_value=256,
            expected_shape=(None, 3, 224, 224),
            actual_shape=(1, 3, 256, 256),
        )

        msg = str(err)
        assert "conv" in msg
        assert "x" in msg
        assert "dim[2]" in msg
        assert "224" in msg
        assert "256" in msg


class TestErrorInheritance:
    """Tests for error class hierarchy."""

    def test_all_errors_inherit_from_base(self):
        assert issubclass(UnificationError, ShapeGuardError)
        assert issubclass(RankMismatchError, ShapeGuardError)
        assert issubclass(DimensionMismatchError, ShapeGuardError)

    def test_can_catch_all_with_base(self):
        """All errors can be caught with ShapeGuardError."""
        n = Dim("n")

        errors = [
            ShapeGuardError("base"),
            UnificationError(n, 1, "a", 2, "b"),
            RankMismatchError(
                expected_rank=2,
                actual_rank=1,
                expected_shape=(1, 2),
                actual_shape=(1,),
            ),
            DimensionMismatchError(
                dim_index=0,
                expected_value=1,
                actual_value=2,
                expected_shape=(1,),
                actual_shape=(2,),
            ),
        ]

        for err in errors:
            with pytest.raises(ShapeGuardError):
                raise err
