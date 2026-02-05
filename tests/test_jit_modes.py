"""
Tests for JIT mode behavior.
"""

from unittest.mock import patch

import pytest

from shapeguard import Dim, config, expects
from shapeguard._compat import is_jax_installed, is_jax_tracing
from shapeguard.errors import DimensionMismatchError
from tests.conftest import requires_jax, requires_numpy


class TestJitDetection:
    """Tests for JAX tracing detection."""

    def test_not_tracing_by_default(self):
        """Outside JIT, is_jax_tracing returns False."""
        # Even with JAX installed, we're not inside a traced context
        assert is_jax_tracing() is False

    def test_is_jax_installed(self):
        """is_jax_installed returns correct value."""
        result = is_jax_installed()
        assert isinstance(result, bool)

    @requires_jax
    def test_tracing_inside_jit(self, jax_array):
        """Inside JIT, is_jax_tracing returns True."""
        import jax

        results = []

        @jax.jit
        def f(x):
            results.append(is_jax_tracing())
            return x + 1

        f(jax_array((3, 4)))

        # During tracing, should have been True
        assert True in results


class TestJitModeCheck:
    """Tests for jit_mode='check' (default)."""

    @requires_numpy
    def test_check_mode_validates(self, np_array):
        """Check mode validates shapes."""
        n = Dim("n")

        @expects(x=(n, 128), jit_mode="check")
        def f(x):
            return x

        # Valid shape
        f(np_array((10, 128)))

        # Invalid shape - should raise
        with pytest.raises(DimensionMismatchError):
            f(np_array((10, 64)))

    @requires_numpy
    def test_check_mode_is_default(self, np_array):
        """Check mode is the default."""
        n = Dim("n")

        @expects(x=(n, 128))  # No jit_mode specified
        def f(x):
            return x

        with pytest.raises(DimensionMismatchError):
            f(np_array((10, 64)))


class TestJitModeSkip:
    """Tests for jit_mode='skip'."""

    @requires_numpy
    def test_skip_mode_still_validates_outside_jit(self, np_array):
        """Skip mode still validates when not tracing."""
        n = Dim("n")

        @expects(x=(n, 128), jit_mode="skip")
        def f(x):
            return x

        # Outside JIT, should still validate
        with pytest.raises(DimensionMismatchError):
            f(np_array((10, 64)))

    @requires_numpy
    def test_skip_mode_skips_when_tracing(self, np_array):
        """Skip mode skips validation when tracing."""
        n = Dim("n")

        @expects(x=(n, 128), jit_mode="skip")
        def f(x):
            return x.shape

        # Mock is_jax_tracing to return True
        with patch("shapeguard.decorator.is_jax_tracing", return_value=True):
            # Should NOT raise even with wrong shape
            result = f(np_array((10, 64)))
            assert result == (10, 64)

    @requires_jax
    def test_skip_mode_with_real_jit(self, jax_array):
        """Skip mode works with real JAX JIT."""
        import jax

        n = Dim("n")

        @expects(x=(n, 128), jit_mode="skip")
        @jax.jit
        def f(x):
            return x + 1

        # Should not raise during JIT compilation even with "wrong" shape
        # (in reality, shape is traced and check is skipped)
        result = f(jax_array((10, 128)))
        assert result.shape == (10, 128)


class TestJitModeWarn:
    """Tests for jit_mode='warn'."""

    @requires_numpy
    def test_warn_mode_raises_outside_jit(self, np_array):
        """Warn mode raises when not tracing (same as check)."""
        n = Dim("n")

        @expects(x=(n, 128), jit_mode="warn")
        def f(x):
            return x

        # Outside JIT, should still raise
        with pytest.raises(DimensionMismatchError):
            f(np_array((10, 64)))

    @requires_numpy
    def test_warn_mode_logs_when_tracing(self, np_array, caplog):
        """Warn mode logs warning when tracing."""
        import logging

        n = Dim("n")

        @expects(x=(n, 128), jit_mode="warn")
        def f(x):
            return x.shape

        # Mock is_jax_tracing to return True
        with patch("shapeguard.decorator.is_jax_tracing", return_value=True):
            with caplog.at_level(logging.WARNING, logger="shapeguard"):
                # Should NOT raise, but should log
                result = f(np_array((10, 64)))
                assert result == (10, 64)

            # Check warning was logged
            assert "validation failed" in caplog.text.lower() or len(caplog.records) > 0


class TestGlobalConfig:
    """Tests for global config affecting jit_mode."""

    @requires_numpy
    def test_global_config_affects_decorator(self, np_array):
        """Global config.jit_mode is used when not overridden."""
        original = config.jit_mode

        try:
            config.jit_mode = "skip"

            n = Dim("n")

            @expects(x=(n, 128))  # No jit_mode - uses global
            def f(x):
                return x.shape

            # Mock tracing
            with patch("shapeguard.decorator.is_jax_tracing", return_value=True):
                # Should skip (from global config)
                result = f(np_array((10, 64)))
                assert result == (10, 64)

        finally:
            config.jit_mode = original

    @requires_numpy
    def test_per_function_overrides_global(self, np_array):
        """Per-function jit_mode overrides global config."""
        original = config.jit_mode

        try:
            config.jit_mode = "skip"  # Global says skip

            n = Dim("n")

            @expects(x=(n, 128), jit_mode="check")  # But this says check
            def f(x):
                return x

            # Even with global skip, this function uses check
            with pytest.raises(DimensionMismatchError):
                f(np_array((10, 64)))

        finally:
            config.jit_mode = original


class TestJitModeMetadata:
    """Tests for jit_mode metadata on decorated functions."""

    @requires_numpy
    def test_jit_mode_stored_on_wrapper(self, np_array):
        """jit_mode is stored as metadata on wrapper."""
        n = Dim("n")

        @expects(x=(n,), jit_mode="skip")
        def f(x):
            return x

        assert f.__shapeguard_jit_mode__ == "skip"

    @requires_numpy
    def test_jit_mode_none_when_not_specified(self, np_array):
        """jit_mode metadata is None when not specified."""
        n = Dim("n")

        @expects(x=(n,))
        def f(x):
            return x

        assert f.__shapeguard_jit_mode__ is None
