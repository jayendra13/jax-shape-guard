"""
Tests for shapeguard.decorator module.
"""

import pytest

from shapeguard import Dim, expects, ShapeGuardError
from shapeguard.errors import (
    RankMismatchError,
    DimensionMismatchError,
    UnificationError,
)
from tests.conftest import requires_numpy, requires_jax


class TestExpectsDecorator:
    """Tests for the @expects decorator."""

    @requires_numpy
    def test_valid_shapes_pass(self, np_array):
        """Function executes normally when shapes match."""
        n, m = Dim("n"), Dim("m")

        @expects(x=(n, m))
        def f(x):
            return x.sum()

        result = f(np_array((3, 4)))
        assert result == 0.0

    @requires_numpy
    def test_rank_mismatch_raises(self, np_array):
        """Wrong rank raises RankMismatchError."""
        n, m = Dim("n"), Dim("m")

        @expects(x=(n, m))
        def f(x):
            return x

        with pytest.raises(RankMismatchError):
            f(np_array((3,)))

    @requires_numpy
    def test_concrete_dim_mismatch(self, np_array):
        """Concrete dimension mismatch raises DimensionMismatchError."""

        @expects(x=(10, 20))
        def f(x):
            return x

        with pytest.raises(DimensionMismatchError):
            f(np_array((10, 30)))

    @requires_numpy
    def test_symbolic_unification_across_args(self, np_array):
        """Same Dim across arguments must match."""
        n, m, k = Dim("n"), Dim("m"), Dim("k")

        @expects(a=(n, m), b=(m, k))
        def matmul(a, b):
            return a @ b

        # Valid: a is (3, 4), b is (4, 5) - m=4 unifies
        result = matmul(np_array((3, 4)), np_array((4, 5)))
        assert result.shape == (3, 5)

    @requires_numpy
    def test_symbolic_conflict_raises(self, np_array):
        """Conflicting symbolic dims raise UnificationError."""
        n, m, k = Dim("n"), Dim("m"), Dim("k")

        @expects(a=(n, m), b=(m, k))
        def matmul(a, b):
            return a @ b

        # Invalid: a is (3, 4), b is (5, 6) - m conflicts (4 vs 5)
        with pytest.raises(UnificationError) as exc_info:
            matmul(np_array((3, 4)), np_array((5, 6)))

        err = exc_info.value
        assert "m" in str(err)

    @requires_numpy
    def test_wildcard_accepts_any(self, np_array):
        """None dimensions accept any value."""

        @expects(x=(None, 128))
        def f(x):
            return x

        f(np_array((5, 128)))
        f(np_array((100, 128)))
        f(np_array((1, 128)))

    @requires_numpy
    def test_mixed_positional_and_keyword(self, np_array):
        """Works with both positional and keyword arguments."""
        n = Dim("n")

        @expects(x=(n, 10), y=(n, 20))
        def f(x, y):
            return x, y

        # Positional
        f(np_array((5, 10)), np_array((5, 20)))

        # Keyword
        f(x=np_array((5, 10)), y=np_array((5, 20)))

        # Mixed
        f(np_array((5, 10)), y=np_array((5, 20)))

    @requires_numpy
    def test_error_includes_function_name(self, np_array):
        """Error message includes function name."""

        @expects(x=(10,))
        def my_function(x):
            return x

        with pytest.raises(ShapeGuardError) as exc_info:
            my_function(np_array((20,)))

        assert "my_function" in str(exc_info.value)

    @requires_numpy
    def test_error_includes_argument_name(self, np_array):
        """Error message includes argument name."""

        @expects(my_input=(10,))
        def f(my_input):
            return my_input

        with pytest.raises(ShapeGuardError) as exc_info:
            f(np_array((20,)))

        assert "my_input" in str(exc_info.value)

    @requires_numpy
    def test_skips_non_array_arguments(self, np_array):
        """Non-array arguments are silently skipped."""
        n = Dim("n")

        @expects(x=(n, 10), scale=(1,))  # scale might be a scalar
        def f(x, scale):
            return x * scale

        # scale is a Python float, not an array - should be skipped
        f(np_array((5, 10)), 2.0)

    @requires_numpy
    def test_preserves_function_metadata(self, np_array):
        """Decorator preserves function name and docstring."""
        n = Dim("n")

        @expects(x=(n,))
        def documented_func(x):
            """This is the docstring."""
            return x

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "This is the docstring."

    def test_invalid_arg_name_raises(self):
        """Specifying non-existent parameter raises ValueError."""
        with pytest.raises(ValueError) as exc_info:

            @expects(nonexistent=(10,))
            def f(x):
                return x

        assert "nonexistent" in str(exc_info.value)

    @requires_numpy
    def test_partial_specification(self, np_array):
        """Can specify shapes for only some arguments."""
        n = Dim("n")

        @expects(x=(n, 10))  # Only x is checked
        def f(x, y, z):
            return x, y, z

        # Only x shape matters
        f(np_array((5, 10)), "anything", [1, 2, 3])


@requires_jax
class TestExpectsWithJAX:
    """Tests for @expects with JAX arrays."""

    def test_jax_array_works(self, jax_array):
        """Basic shape checking works with JAX arrays."""
        n, m = Dim("n"), Dim("m")

        @expects(x=(n, m))
        def f(x):
            return x.sum()

        result = f(jax_array((3, 4)))
        assert float(result) == 0.0

    def test_jax_unification(self, jax_array):
        """Symbolic unification works with JAX arrays."""
        n, m, k = Dim("n"), Dim("m"), Dim("k")

        @expects(a=(n, m), b=(m, k))
        def matmul(a, b):
            return a @ b

        result = matmul(jax_array((3, 4)), jax_array((4, 5)))
        assert result.shape == (3, 5)
