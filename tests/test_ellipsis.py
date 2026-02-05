"""
Tests for ellipsis support in shape specs.
"""

import pytest

from shapeguard import Dim, expects, check_shape
from shapeguard.core import ELLIPSIS
from shapeguard.spec import format_spec
from shapeguard.errors import RankMismatchError, DimensionMismatchError, UnificationError
from tests.conftest import requires_numpy


class TestEllipsisSpec:
    """Tests for ellipsis in shape specifications."""

    @requires_numpy
    def test_ellipsis_trailing_dims(self, np_array):
        """Ellipsis matches variable leading dims."""
        n, m = Dim("n"), Dim("m")

        @expects(x=(..., n, m))
        def f(x):
            return x.shape

        # 2D: ellipsis matches nothing
        assert f(np_array((3, 4))) == (3, 4)

        # 3D: ellipsis matches (2,)
        assert f(np_array((2, 3, 4))) == (2, 3, 4)

        # 4D: ellipsis matches (1, 2)
        assert f(np_array((1, 2, 3, 4))) == (1, 2, 3, 4)

    @requires_numpy
    def test_ellipsis_with_python_ellipsis(self, np_array):
        """Can use Python's ... directly."""
        n = Dim("n")

        @expects(x=(..., n))
        def f(x):
            return x.shape[-1]

        assert f(np_array((5,))) == 5
        assert f(np_array((3, 5))) == 5
        assert f(np_array((2, 3, 5))) == 5

    @requires_numpy
    def test_ellipsis_with_sentinel(self, np_array):
        """Can use ELLIPSIS sentinel."""
        n = Dim("n")

        @expects(x=(ELLIPSIS, n))
        def f(x):
            return x.shape[-1]

        assert f(np_array((5,))) == 5
        assert f(np_array((3, 5))) == 5

    @requires_numpy
    def test_ellipsis_unifies_trailing(self, np_array):
        """Trailing dims after ellipsis still unify."""
        n, m = Dim("n"), Dim("m")

        @expects(x=(..., n, m), y=(..., m, n))
        def f(x, y):
            return x.shape[-2:]

        # n=3, m=4 should unify
        result = f(np_array((2, 3, 4)), np_array((5, 4, 3)))
        assert result == (3, 4)

    @requires_numpy
    def test_ellipsis_conflict_raises(self, np_array):
        """Conflicting dims after ellipsis raises error."""
        n = Dim("n")

        @expects(x=(..., n), y=(..., n))
        def f(x, y):
            return x, y

        with pytest.raises(UnificationError):
            f(np_array((2, 3)), np_array((2, 4)))  # n=3 vs n=4

    @requires_numpy
    def test_ellipsis_minimum_rank(self, np_array):
        """Ellipsis still requires minimum rank for fixed dims."""
        n, m = Dim("n"), Dim("m")

        @expects(x=(..., n, m))
        def f(x):
            return x

        # Need at least 2 dims
        with pytest.raises(RankMismatchError):
            f(np_array((5,)))  # Only 1 dim

    @requires_numpy
    def test_ellipsis_concrete_trailing(self, np_array):
        """Ellipsis with concrete trailing dims."""

        @expects(x=(..., 128))
        def f(x):
            return x.shape

        assert f(np_array((128,))) == (128,)
        assert f(np_array((10, 128))) == (10, 128)
        assert f(np_array((5, 10, 128))) == (5, 10, 128)

        with pytest.raises(DimensionMismatchError):
            f(np_array((10, 64)))  # Last dim not 128

    @requires_numpy
    def test_ellipsis_leading_and_trailing(self, np_array):
        """Ellipsis between leading and trailing fixed dims."""
        n = Dim("n")

        # (n, ..., 10) - n at start, 10 at end, anything in between
        @expects(x=(n, ..., 10))
        def f(x):
            return x.shape[0]

        assert f(np_array((3, 10))) == 3
        assert f(np_array((3, 5, 10))) == 3
        assert f(np_array((3, 4, 5, 10))) == 3

    @requires_numpy
    def test_multiple_ellipsis_error(self, np_array):
        """Multiple ellipsis in spec raises error."""
        with pytest.raises(ValueError, match="more than one ellipsis"):

            @expects(x=(..., ..., 10))
            def f(x):
                return x

            f(np_array((10,)))

    def test_format_spec_with_ellipsis(self):
        """format_spec handles ellipsis."""
        n = Dim("n")

        assert format_spec((..., n, 10)) == "(..., n, 10)"
        assert format_spec((ELLIPSIS, n)) == "(..., n)"


class TestCheckShapeEllipsis:
    """Tests for check_shape with ellipsis."""

    @requires_numpy
    def test_check_shape_ellipsis(self, np_array):
        """check_shape works with ellipsis."""
        n = Dim("n")

        ctx = check_shape(np_array((2, 3, 4)), (..., n), name="x")
        assert ctx.resolve(n) == 4

    @requires_numpy
    def test_check_shape_ellipsis_chain(self, np_array):
        """Chained check_shape with ellipsis."""
        n = Dim("n")

        ctx = check_shape(np_array((2, 3, 5)), (..., n), name="x")
        check_shape(np_array((4, 5)), (..., n), name="y", ctx=ctx)

        assert ctx.resolve(n) == 5
