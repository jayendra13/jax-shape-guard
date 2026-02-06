"""
Tests for shapeguard.broadcast module.
"""

import pytest

from shapeguard.broadcast import broadcast_shape, explain_broadcast
from shapeguard.errors import BroadcastError


class TestBroadcastShape:
    """Tests for broadcast_shape function."""

    def test_same_shape(self):
        """Identical shapes broadcast to themselves."""
        assert broadcast_shape((3, 4), (3, 4)) == (3, 4)

    def test_scalar_broadcast(self):
        """Empty tuple (scalar) broadcasts with any shape."""
        assert broadcast_shape((3, 4), ()) == (3, 4)
        assert broadcast_shape((), (3, 4)) == (3, 4)

    def test_trailing_one(self):
        """Shapes with 1s broadcast correctly."""
        assert broadcast_shape((3, 1), (1, 4)) == (3, 4)

    def test_leading_dims(self):
        """Shorter shape gets leading dims from longer shape."""
        assert broadcast_shape((2, 3, 4), (4,)) == (2, 3, 4)
        assert broadcast_shape((4,), (2, 3, 4)) == (2, 3, 4)

    def test_multiple_shapes(self):
        """Broadcasting works with more than two shapes."""
        assert broadcast_shape((3, 1), (1, 4), (1,)) == (3, 4)
        assert broadcast_shape((2, 1, 1), (3, 1), (4,)) == (2, 3, 4)

    def test_single_shape(self):
        """Single shape returns unchanged."""
        assert broadcast_shape((3, 4, 5)) == (3, 4, 5)

    def test_all_ones(self):
        """All 1s broadcast to larger dimensions."""
        assert broadcast_shape((1, 1), (3, 4)) == (3, 4)
        assert broadcast_shape((1,), (1, 1), (3, 4)) == (3, 4)

    def test_complex_broadcast(self):
        """Complex multi-dimensional broadcasting."""
        # (256, 256, 3) with (3,) -> (256, 256, 3)
        assert broadcast_shape((256, 256, 3), (3,)) == (256, 256, 3)
        # (8, 1, 6, 1) with (7, 1, 5) -> (8, 7, 6, 5)
        assert broadcast_shape((8, 1, 6, 1), (7, 1, 5)) == (8, 7, 6, 5)

    def test_incompatible_raises(self):
        """Incompatible shapes raise BroadcastError."""
        with pytest.raises(BroadcastError) as exc:
            broadcast_shape((3, 4), (5, 4))

        err = exc.value
        assert err.dim_index == -2  # Second-to-last dimension
        assert 3 in err.dim_values
        assert 5 in err.dim_values
        assert "dimension" in str(err)

    def test_incompatible_trailing(self):
        """Incompatible trailing dimensions raise error."""
        with pytest.raises(BroadcastError) as exc:
            broadcast_shape((3, 4), (3, 5))

        assert exc.value.dim_index == -1

    def test_incompatible_multiple_shapes(self):
        """Error raised when any pair is incompatible."""
        with pytest.raises(BroadcastError):
            broadcast_shape((2, 3), (2, 4), (2, 3))

    def test_no_shapes_raises(self):
        """No shapes raises ValueError."""
        with pytest.raises(ValueError) as exc:
            broadcast_shape()
        assert "at least one shape" in str(exc.value)

    def test_from_arrays(self, np_array):
        """Broadcasting works with array objects."""
        result = broadcast_shape(np_array((3, 1)), np_array((1, 4)))
        assert result == (3, 4)

    def test_mixed_arrays_and_tuples(self, np_array):
        """Can mix arrays and tuples."""
        result = broadcast_shape(np_array((3, 1)), (1, 4))
        assert result == (3, 4)

    def test_from_list(self):
        """Lists are interpreted as shapes."""
        assert broadcast_shape([3, 1], [1, 4]) == (3, 4)

    def test_invalid_type_raises(self):
        """Non-shape-like objects raise TypeError."""
        with pytest.raises(TypeError):
            broadcast_shape(42)

    def test_empty_shapes(self):
        """Two empty shapes broadcast to empty."""
        assert broadcast_shape((), ()) == ()


class TestExplainBroadcast:
    """Tests for explain_broadcast function."""

    def test_explanation_format(self):
        """Explanation contains expected components."""
        result = explain_broadcast((3, 1), (1, 4))
        assert "3" in result
        assert "4" in result
        assert "broadcast" in result.lower()
        assert "(3, 4)" in result

    def test_step_by_step(self):
        """Explanation shows alignment and comparison steps."""
        result = explain_broadcast((3, 1, 4), (5, 4))
        assert "Step 1" in result
        assert "Step 2" in result
        assert "Align" in result
        assert "Result" in result

    def test_incompatible_explanation(self):
        """Incompatible shapes show error in explanation."""
        result = explain_broadcast((3, 4), (5, 4))
        assert "incompatible" in result.lower() or "error" in result.lower()

    def test_match_annotation(self):
        """Matching dimensions are annotated."""
        result = explain_broadcast((4,), (4,))
        assert "match" in result.lower()

    def test_broadcast_annotation(self):
        """Broadcasting dimensions show the transformation."""
        result = explain_broadcast((1,), (5,))
        assert "broadcast" in result.lower()
        assert "1" in result
        assert "5" in result

    def test_single_shape(self):
        """Single shape explains no broadcasting needed."""
        result = explain_broadcast((3, 4))
        assert "no broadcasting" in result.lower() or "single" in result.lower()

    def test_no_shapes(self):
        """No shapes returns appropriate message."""
        result = explain_broadcast()
        assert "no shapes" in result.lower()

    def test_multiple_shapes(self):
        """Explanation works with multiple shapes."""
        result = explain_broadcast((2, 1), (1, 3), (1,))
        assert "(2, 1)" in result
        assert "(1, 3)" in result
        assert "(1,)" in result

    def test_with_arrays(self, np_array):
        """Works with array inputs."""
        result = explain_broadcast(np_array((3, 1)), np_array((1, 4)))
        assert "(3, 4)" in result


class TestBroadcastErrorMessage:
    """Tests for BroadcastError message formatting."""

    def test_error_message_contains_shapes(self):
        """Error message includes the problematic shapes."""
        try:
            broadcast_shape((3, 4), (5, 6))
        except BroadcastError as e:
            msg = str(e)
            assert "(3, 4)" in msg
            assert "(5, 6)" in msg

    def test_error_message_contains_dimension(self):
        """Error message identifies the problematic dimension."""
        try:
            broadcast_shape((3, 4), (5, 4))
        except BroadcastError as e:
            msg = str(e)
            assert "dimension" in msg

    def test_error_attributes(self):
        """BroadcastError has correct attributes."""
        try:
            broadcast_shape((3, 4), (5, 4))
        except BroadcastError as e:
            assert e.shapes == [(3, 4), (5, 4)]
            assert e.dim_index == -2
            assert set(e.dim_values) == {3, 5}


class TestEdgeCases:
    """Edge case tests for broadcasting."""

    def test_high_dimensional(self):
        """Broadcasting works with many dimensions."""
        shape1 = (2, 3, 4, 5, 6)
        shape2 = (1, 1, 1, 5, 6)
        assert broadcast_shape(shape1, shape2) == (2, 3, 4, 5, 6)

    def test_very_different_ranks(self):
        """Broadcasting shapes with very different ranks."""
        assert broadcast_shape((2, 3, 4, 5), (5,)) == (2, 3, 4, 5)
        assert broadcast_shape((5,), (2, 3, 4, 5)) == (2, 3, 4, 5)

    def test_all_broadcast_from_ones(self):
        """Every dimension broadcasts from 1."""
        assert broadcast_shape((1, 1, 1), (2, 3, 4)) == (2, 3, 4)
        assert broadcast_shape((2, 1, 1), (1, 3, 1), (1, 1, 4)) == (2, 3, 4)

    def test_numpy_style_examples(self):
        """Examples from NumPy broadcasting documentation."""
        # From https://numpy.org/doc/stable/user/basics.broadcasting.html
        assert broadcast_shape((5, 4), (1,)) == (5, 4)
        assert broadcast_shape((5, 4), (4,)) == (5, 4)
        assert broadcast_shape((15, 3, 5), (15, 1, 5)) == (15, 3, 5)
        assert broadcast_shape((15, 3, 5), (3, 5)) == (15, 3, 5)
        assert broadcast_shape((15, 3, 5), (3, 1)) == (15, 3, 5)
