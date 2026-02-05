"""
Tests for shapeguard.core module.
"""

import pytest

from shapeguard.core import Dim, UnificationContext
from shapeguard.errors import UnificationError


class TestDim:
    """Tests for the Dim class."""

    def test_dim_has_name(self):
        n = Dim("n")
        assert n.name == "n"

    def test_dim_repr(self):
        n = Dim("batch_size")
        assert repr(n) == "batch_size"

    def test_same_object_equals_itself(self):
        n = Dim("n")
        assert n == n

    def test_different_objects_not_equal(self):
        """Two Dim objects with same name are NOT equal (identity-based)."""
        n1 = Dim("n")
        n2 = Dim("n")
        assert n1 != n2

    def test_dim_is_hashable(self):
        n = Dim("n")
        d = {n: 42}
        assert d[n] == 42

    def test_different_dims_have_different_hashes(self):
        n1 = Dim("n")
        n2 = Dim("n")
        # Different objects should have different hashes (usually)
        assert hash(n1) != hash(n2)


class TestUnificationContext:
    """Tests for the UnificationContext class."""

    def test_empty_context(self):
        ctx = UnificationContext()
        n = Dim("n")
        assert ctx.resolve(n) is None

    def test_bind_and_resolve(self):
        ctx = UnificationContext()
        n = Dim("n")
        ctx.bind(n, 42, "x.shape[0]")
        assert ctx.resolve(n) == 42

    def test_bind_same_value_succeeds(self):
        """Binding to same value multiple times is OK."""
        ctx = UnificationContext()
        n = Dim("n")
        ctx.bind(n, 42, "x.shape[0]")
        ctx.bind(n, 42, "y.shape[0]")  # Should not raise
        assert ctx.resolve(n) == 42

    def test_bind_different_value_raises(self):
        """Binding to different value raises UnificationError."""
        ctx = UnificationContext()
        n = Dim("n")
        ctx.bind(n, 42, "x.shape[0]")

        with pytest.raises(UnificationError) as exc_info:
            ctx.bind(n, 100, "y.shape[0]")

        err = exc_info.value
        assert err.dim is n
        assert err.expected_value == 42
        assert err.actual_value == 100

    def test_multiple_dims(self):
        """Can track multiple dimensions independently."""
        ctx = UnificationContext()
        n = Dim("n")
        m = Dim("m")

        ctx.bind(n, 10, "x.shape[0]")
        ctx.bind(m, 20, "x.shape[1]")

        assert ctx.resolve(n) == 10
        assert ctx.resolve(m) == 20

    def test_format_bindings_empty(self):
        ctx = UnificationContext()
        assert ctx.format_bindings() == "{}"

    def test_format_bindings_with_values(self):
        ctx = UnificationContext()
        n = Dim("n")
        ctx.bind(n, 42, "x.shape[0]")
        formatted = ctx.format_bindings()
        assert "n=42" in formatted
        assert "x.shape[0]" in formatted

    def test_get_binding_source(self):
        ctx = UnificationContext()
        n = Dim("n")
        ctx.bind(n, 42, "x.shape[0]")
        assert ctx.get_binding_source(n) == "x.shape[0]"

    def test_get_binding_source_unbound(self):
        ctx = UnificationContext()
        n = Dim("n")
        assert ctx.get_binding_source(n) is None
