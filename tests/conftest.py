"""
Shared pytest fixtures for ShapeGuard tests.
"""

import pytest

# Try to import numpy, skip tests if not available
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Try to import jax, skip tests if not available
try:
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


requires_numpy = pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not installed")
requires_jax = pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")


@pytest.fixture
def np_array():
    """Factory for creating NumPy arrays with specified shape."""
    if not HAS_NUMPY:
        pytest.skip("NumPy not installed")

    def _make(shape):
        return np.zeros(shape)

    return _make


@pytest.fixture
def jax_array():
    """Factory for creating JAX arrays with specified shape."""
    if not HAS_JAX:
        pytest.skip("JAX not installed")

    def _make(shape):
        return jnp.zeros(shape)

    return _make
