"""
Tests for ShapeGuard configuration.
"""

import pytest

from shapeguard import config
from shapeguard.config import Config


class TestConfig:
    """Tests for the Config class."""

    def test_default_jit_mode(self):
        """Default JIT mode is 'check'."""
        c = Config()
        assert c.jit_mode == "check"

    def test_set_jit_mode_valid(self):
        """Can set valid JIT modes."""
        c = Config()

        c.jit_mode = "skip"
        assert c.jit_mode == "skip"

        c.jit_mode = "warn"
        assert c.jit_mode == "warn"

        c.jit_mode = "check"
        assert c.jit_mode == "check"

    def test_set_jit_mode_invalid(self):
        """Invalid JIT mode raises ValueError."""
        c = Config()

        with pytest.raises(ValueError, match="Invalid jit_mode"):
            c.jit_mode = "invalid"

        with pytest.raises(ValueError):
            c.jit_mode = "SKIP"  # Case sensitive

    def test_config_repr(self):
        """Config has readable repr."""
        c = Config()
        assert "jit_mode" in repr(c)
        assert "check" in repr(c)

    def test_global_config_exists(self):
        """Global config singleton exists."""
        assert config is not None
        assert isinstance(config, Config)

    def test_global_config_modifiable(self):
        """Global config can be modified."""
        original = config.jit_mode

        try:
            config.jit_mode = "skip"
            assert config.jit_mode == "skip"
        finally:
            config.jit_mode = original
