"""
Global configuration for ShapeGuard.
"""

from __future__ import annotations

from typing import Literal

JitMode = Literal["check", "warn", "skip"]


class Config:
    """
    Global ShapeGuard configuration.

    Attributes:
        jit_mode: How to handle validation under JAX JIT tracing.
            - "check": Always validate, raise on mismatch (default)
            - "warn": Validate, log warning on mismatch, continue
            - "skip": Skip validation entirely under JIT

    Example:
        ```python
        from shapeguard import config

        config.jit_mode = "skip"  # Disable checks under JIT globally
        ```
    """

    __slots__ = ("_jit_mode",)

    def __init__(self) -> None:
        self._jit_mode: JitMode = "check"

    @property
    def jit_mode(self) -> JitMode:
        """Get current JIT mode."""
        return self._jit_mode

    @jit_mode.setter
    def jit_mode(self, value: JitMode) -> None:
        """Set JIT mode with validation."""
        valid_modes = ("check", "warn", "skip")
        if value not in valid_modes:
            raise ValueError(f"Invalid jit_mode: {value!r}. Must be one of: {valid_modes}")
        self._jit_mode = value

    def __repr__(self) -> str:
        return f"Config(jit_mode={self._jit_mode!r})"


# Global singleton instance
config = Config()
