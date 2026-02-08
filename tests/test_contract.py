"""
Tests for @contract decorator (combined input + output validation).
"""

import pytest

from shapeguard import Dim, ShapeGuardError, contract
from shapeguard.errors import OutputShapeError
from tests.conftest import requires_numpy


class TestContract:
    """Tests for the @contract decorator."""

    @requires_numpy
    def test_basic_pass(self, np_array):
        """Valid inputs and output pass."""
        n, m, k = Dim("n"), Dim("m"), Dim("k")

        @contract(inputs={"a": (n, m), "b": (m, k)}, output=(n, k))
        def matmul(a, b):
            return a @ b

        result = matmul(np_array((3, 4)), np_array((4, 5)))
        assert result.shape == (3, 5)

    @requires_numpy
    def test_input_validation_fails(self, np_array):
        """Input shape mismatch raises on input validation."""
        n, m, k = Dim("n"), Dim("m"), Dim("k")

        @contract(inputs={"a": (n, m), "b": (m, k)}, output=(n, k))
        def matmul(a, b):
            return a @ b

        with pytest.raises(ShapeGuardError):
            matmul(np_array((3, 4)), np_array((5, 6)))  # m: 4 vs 5

    @requires_numpy
    def test_output_validation_fails(self, np_array):
        """Output shape mismatch raises on output validation."""
        n, m = Dim("n"), Dim("m")

        @contract(inputs={"x": (n, m)}, output=(n,))
        def f(x):
            import numpy as np

            return np.zeros((999,))  # n was bound to x.shape[0]=3

        with pytest.raises(ShapeGuardError):
            f(np_array((3, 4)))

    @requires_numpy
    def test_shared_dims_across_input_output(self, np_array):
        """Symbolic dims unify across inputs and output."""
        n, m = Dim("n"), Dim("m")

        @contract(inputs={"x": (n, m)}, output=(m, n))
        def transpose(x):
            return x.T

        result = transpose(np_array((3, 4)))
        assert result.shape == (4, 3)

    @requires_numpy
    def test_tuple_output(self, np_array):
        """Tuple output validation works."""
        n, m = Dim("n"), Dim("m")

        @contract(inputs={"x": (n, m)}, output=((n, m), (m,)))
        def f(x):
            return x, x[0]

        a, b = f(np_array((3, 4)))
        assert a.shape == (3, 4)
        assert b.shape == (4,)

    @requires_numpy
    def test_dict_output(self, np_array):
        """Dict output validation works."""
        n, m = Dim("n"), Dim("m")

        @contract(
            inputs={"x": (n, m)},
            output={"full": (n, m), "col": (n,)},
        )
        def f(x):
            return {"full": x, "col": x[:, 0]}

        out = f(np_array((3, 4)))
        assert out["full"].shape == (3, 4)
        assert out["col"].shape == (3,)

    @requires_numpy
    def test_scalar_output(self, np_array):
        """Scalar output with output=() works."""

        @contract(inputs={"x": (3, 4)}, output=())
        def f(x):
            return x.sum()

        result = f(np_array((3, 4)))
        assert result.shape == ()

    def test_invalid_arg_name_raises(self):
        """Specifying non-existent parameter raises ValueError."""
        with pytest.raises(ValueError) as exc_info:

            @contract(inputs={"nonexistent": (10,)}, output=(10,))
            def f(x):
                return x

        assert "nonexistent" in str(exc_info.value)

    @requires_numpy
    def test_preserves_function_metadata(self, np_array):
        """Decorator preserves function name and docstring."""
        n = Dim("n")

        @contract(inputs={"x": (n,)}, output=(n,))
        def documented_func(x):
            """My docstring."""
            return x

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "My docstring."

    @requires_numpy
    def test_error_includes_function_name(self, np_array):
        """Error message includes function name."""

        @contract(inputs={"x": (10,)}, output=(10,))
        def my_contract_fn(x):
            return x

        with pytest.raises(ShapeGuardError) as exc_info:
            my_contract_fn(np_array((20,)))

        assert "my_contract_fn" in str(exc_info.value)

    @requires_numpy
    def test_partial_input_spec(self, np_array):
        """Can specify shapes for only some arguments."""
        n = Dim("n")

        @contract(inputs={"x": (n, 10)}, output=(n, 10))
        def f(x, y):
            return x

        result = f(np_array((5, 10)), "anything")
        assert result.shape == (5, 10)

    @requires_numpy
    def test_output_type_error_non_array(self, np_array):
        """Non-array output when array expected raises."""
        n = Dim("n")

        @contract(inputs={"x": (n,)}, output=(n,))
        def f(x):
            return "not an array"

        with pytest.raises(OutputShapeError):
            f(np_array((3,)))
