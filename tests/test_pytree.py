"""
Tests for PyTree shape specifications.
"""

import pytest

from shapeguard import Dim, expects, ShapeGuardError
from tests.conftest import requires_numpy


class TestPyTreeSpecs:
    """Tests for dict-based PyTree shape specifications."""

    @requires_numpy
    def test_simple_dict_spec(self, np_array):
        """Simple dict spec with array values."""
        n, m = Dim("n"), Dim("m")

        @expects(params={"w": (n, m), "b": (m,)})
        def f(params):
            return params["w"].shape

        result = f({"w": np_array((3, 4)), "b": np_array((4,))})
        assert result == (3, 4)

    @requires_numpy
    def test_nested_dict_spec(self, np_array):
        """Nested dict spec."""
        n, m = Dim("n"), Dim("m")

        @expects(
            params={
                "layer1": {"w": (n, m), "b": (m,)},
                "layer2": {"w": (m, 10), "b": (10,)},
            }
        )
        def f(params):
            return params["layer1"]["w"].shape

        result = f({
            "layer1": {"w": np_array((5, 8)), "b": np_array((8,))},
            "layer2": {"w": np_array((8, 10)), "b": np_array((10,))},
        })
        assert result == (5, 8)

    @requires_numpy
    def test_dict_unification(self, np_array):
        """Dims unify across dict entries."""
        n, m = Dim("n"), Dim("m")

        @expects(params={"w": (n, m), "b": (m,)}, x=(n,))
        def f(params, x):
            return x @ params["w"]

        # n=3, m=4 should unify
        result = f(
            {"w": np_array((3, 4)), "b": np_array((4,))},
            np_array((3,))
        )
        assert result.shape == (4,)

    @requires_numpy
    def test_dict_unification_conflict(self, np_array):
        """Conflicting dims in dict raise error."""
        m = Dim("m")

        @expects(params={"w": (5, m), "b": (m,)})
        def f(params):
            return params

        with pytest.raises(ShapeGuardError):
            # w has m=4, b has shape (5,) which conflicts
            f({"w": np_array((5, 4)), "b": np_array((5,))})

    @requires_numpy
    def test_missing_key_error(self, np_array):
        """Missing dict key raises clear error."""
        n, m = Dim("n"), Dim("m")

        @expects(params={"w": (n, m), "b": (m,)})
        def f(params):
            return params

        with pytest.raises(ShapeGuardError) as exc_info:
            f({"w": np_array((3, 4))})  # Missing "b"

        err_str = str(exc_info.value)
        assert "key 'b'" in err_str
        assert "keys: ['w']" in err_str

    @requires_numpy
    def test_non_dict_error(self, np_array):
        """Non-dict value for dict spec raises error."""
        n, m = Dim("n"), Dim("m")

        @expects(params={"w": (n, m)})
        def f(params):
            return params

        with pytest.raises(ShapeGuardError) as exc_info:
            f(np_array((3, 4)))  # Array instead of dict

        err_str = str(exc_info.value)
        assert "expected: dict" in err_str.lower()
        assert "ndarray" in err_str

    @requires_numpy
    def test_mixed_dict_and_array_args(self, np_array):
        """Can mix dict specs and array specs."""
        n, m = Dim("n"), Dim("m")

        @expects(
            params={"w": (n, m)},
            x=(n,),
            y=(m,)
        )
        def f(params, x, y):
            return x @ params["w"] + y

        result = f(
            {"w": np_array((3, 4))},
            np_array((3,)),
            np_array((4,))
        )
        assert result.shape == (4,)

    @requires_numpy
    def test_extra_keys_allowed(self, np_array):
        """Extra keys in value dict are allowed."""
        n, m = Dim("n"), Dim("m")

        @expects(params={"w": (n, m)})
        def f(params):
            return params["w"].shape

        # Has extra "b" key not in spec - should be OK
        result = f({"w": np_array((3, 4)), "b": np_array((4,)), "extra": 123})
        assert result == (3, 4)

    @requires_numpy
    def test_deeply_nested(self, np_array):
        """Deeply nested dict specs work."""
        n = Dim("n")

        @expects(
            model={
                "encoder": {
                    "layer0": {"w": (n, 64)},
                    "layer1": {"w": (64, 64)},
                },
                "decoder": {
                    "layer0": {"w": (64, n)},
                },
            }
        )
        def f(model):
            return model["encoder"]["layer0"]["w"].shape

        result = f({
            "encoder": {
                "layer0": {"w": np_array((10, 64))},
                "layer1": {"w": np_array((64, 64))},
            },
            "decoder": {
                "layer0": {"w": np_array((64, 10))},  # n=10 must match
            },
        })
        assert result == (10, 64)
