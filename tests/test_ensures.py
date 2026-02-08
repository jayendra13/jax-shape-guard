"""
Tests for @ensures decorator (output validation).
"""

import pytest

from shapeguard import Dim, ShapeGuardError, ensures, expects
from shapeguard.errors import OutputShapeError, RankMismatchError
from tests.conftest import requires_numpy


class TestEnsuresStandalone:
    """Tests for @ensures used without @expects."""

    @requires_numpy
    def test_single_array_pass(self, np_array):
        """Valid single-array output passes."""
        n, m = Dim("n"), Dim("m")

        @ensures(result=(n, m))
        def f(x):
            return x

        result = f(np_array((3, 4)))
        assert result.shape == (3, 4)

    @requires_numpy
    def test_single_array_fail_rank(self, np_array):
        """Wrong rank output raises."""
        n, m = Dim("n"), Dim("m")

        @ensures(result=(n, m))
        def f(x):
            return x

        with pytest.raises(ShapeGuardError):
            f(np_array((3,)))

    @requires_numpy
    def test_single_array_fail_dim(self, np_array):
        """Concrete dimension mismatch in output raises."""

        @ensures(result=(3, 4))
        def f(x):
            return x

        with pytest.raises(ShapeGuardError):
            f(np_array((3, 5)))

    @requires_numpy
    def test_scalar_output(self, np_array):
        """Scalar output with result=() passes for 0-d arrays."""

        @ensures(result=())
        def f(x):
            return x.sum()

        result = f(np_array((3, 4)))
        assert result.shape == ()

    @requires_numpy
    def test_scalar_output_fail(self, np_array):
        """Non-scalar output with result=() raises."""

        @ensures(result=())
        def f(x):
            return x

        with pytest.raises(ShapeGuardError):
            f(np_array((3, 4)))

    @requires_numpy
    def test_tuple_output_pass(self, np_array):
        """Tuple output validation passes."""
        n, m = Dim("n"), Dim("m")

        @ensures(result=((n, m), (n,)))
        def f(x):
            return x, x[:, 0]

        a, b = f(np_array((3, 4)))
        assert a.shape == (3, 4)
        assert b.shape == (3,)

    @requires_numpy
    def test_tuple_output_wrong_length(self, np_array):
        """Tuple output with wrong number of elements raises."""
        n, m = Dim("n"), Dim("m")

        @ensures(result=((n, m), (n,)))
        def f(x):
            return (x,)  # Only 1 element, expected 2

        with pytest.raises(OutputShapeError):
            f(np_array((3, 4)))

    @requires_numpy
    def test_tuple_output_not_tuple(self, np_array):
        """Non-tuple output when tuple expected raises."""
        n = Dim("n")

        @ensures(result=((n,), (n,)))
        def f(x):
            return x  # Single array, not tuple

        with pytest.raises(OutputShapeError):
            f(np_array((3,)))

    @requires_numpy
    def test_tuple_output_wrong_shape(self, np_array):
        """Tuple output with wrong element shape raises."""
        n = Dim("n")

        @ensures(result=((n,), (n,)))
        def f(x, y):
            return x, y

        # n binds to 3 from first element, but second is 5
        with pytest.raises(ShapeGuardError):
            f(np_array((3,)), np_array((5,)))

    @requires_numpy
    def test_dict_output_pass(self, np_array):
        """Dict output validation passes."""
        n, m = Dim("n"), Dim("m")

        @ensures(result={"logits": (n, m), "hidden": (n,)})
        def f(x):
            return {"logits": x, "hidden": x[:, 0]}

        out = f(np_array((3, 4)))
        assert out["logits"].shape == (3, 4)
        assert out["hidden"].shape == (3,)

    @requires_numpy
    def test_dict_output_missing_key(self, np_array):
        """Dict output missing expected key raises."""
        n = Dim("n")

        @ensures(result={"a": (n,), "b": (n,)})
        def f(x):
            return {"a": x}

        with pytest.raises(ShapeGuardError):
            f(np_array((3,)))

    @requires_numpy
    def test_symbolic_dims_bind_from_output(self, np_array):
        """Symbolic dims in output spec bind correctly."""
        n, m = Dim("n"), Dim("m")

        @ensures(result=((n, m), (m, n)))
        def f():
            import numpy as np

            return np.zeros((3, 4)), np.zeros((4, 3))

        a, b = f()
        assert a.shape == (3, 4)
        assert b.shape == (4, 3)

    @requires_numpy
    def test_symbolic_dims_conflict_in_output(self, np_array):
        """Conflicting symbolic dims in output raises."""
        n = Dim("n")

        @ensures(result=((n,), (n,)))
        def f():
            import numpy as np

            return np.zeros((3,)), np.zeros((5,))

        with pytest.raises(ShapeGuardError):
            f()

    @requires_numpy
    def test_error_includes_function_name(self, np_array):
        """Error message includes function name."""

        @ensures(result=(10,))
        def my_output_fn(x):
            return x

        with pytest.raises(ShapeGuardError) as exc_info:
            my_output_fn(np_array((20,)))

        assert "my_output_fn" in str(exc_info.value)

    @requires_numpy
    def test_error_includes_result(self, np_array):
        """Error message includes 'result' as argument."""

        @ensures(result=(10,))
        def f(x):
            return x

        with pytest.raises(ShapeGuardError) as exc_info:
            f(np_array((20,)))

        assert "result" in str(exc_info.value)

    @requires_numpy
    def test_preserves_function_metadata(self, np_array):
        """Decorator preserves function name and docstring."""
        n = Dim("n")

        @ensures(result=(n,))
        def documented_func(x):
            """My docstring."""
            return x

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "My docstring."


class TestEnsuresWithExpects:
    """Tests for @ensures stacked with @expects (shared context)."""

    @requires_numpy
    def test_shared_context_pass(self, np_array):
        """Dims from inputs carry to output validation."""
        n, m, k = Dim("n"), Dim("m"), Dim("k")

        @expects(a=(n, m), b=(m, k))
        @ensures(result=(n, k))
        def matmul(a, b):
            return a @ b

        result = matmul(np_array((3, 4)), np_array((4, 5)))
        assert result.shape == (3, 5)

    @requires_numpy
    def test_shared_context_output_mismatch(self, np_array):
        """Output violating input-bound dims raises."""
        n, m = Dim("n"), Dim("m")

        @expects(x=(n, m))
        @ensures(result=(n,))
        def f(x):
            import numpy as np

            return np.zeros((999,))  # n was bound to x.shape[0], not 999

        with pytest.raises(ShapeGuardError):
            f(np_array((3, 4)))

    @requires_numpy
    def test_shared_context_tuple_output(self, np_array):
        """Shared context works with tuple outputs."""
        n, m = Dim("n"), Dim("m")

        @expects(x=(n, m))
        @ensures(result=((n, m), (m,)))
        def f(x):
            return x, x[0]

        a, b = f(np_array((3, 4)))
        assert a.shape == (3, 4)
        assert b.shape == (4,)

    @requires_numpy
    def test_input_validation_still_works(self, np_array):
        """Input validation still runs when stacked with @ensures."""
        n, m = Dim("n"), Dim("m")

        @expects(x=(n, m))
        @ensures(result=(n, m))
        def f(x):
            return x

        with pytest.raises(RankMismatchError):
            f(np_array((3,)))
