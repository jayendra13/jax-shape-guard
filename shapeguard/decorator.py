"""
Contract decorators for shape validation.
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Callable, TypeVar

from shapeguard.core import UnificationContext
from shapeguard.spec import ShapeSpec, match_shape
from shapeguard.errors import ShapeGuardError
from shapeguard._compat import get_shape, is_array, is_jax_tracing
from shapeguard.config import JitMode, config


F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger("shapeguard")


# Type for PyTree shape specs (nested dicts)
PyTreeSpec = dict[str, Any] | ShapeSpec


def _check_pytree(
    value: Any,
    spec: PyTreeSpec,
    ctx: UnificationContext,
    source: str,
    fn_name: str,
) -> None:
    """
    Recursively check a value against a PyTree spec.

    Handles nested dicts and arrays.
    """
    if isinstance(spec, dict):
        # Spec is a dict - value should be a dict-like with matching keys
        if not isinstance(value, dict):
            raise ShapeGuardError(
                f"Expected dict for {source}, got {type(value).__name__}",
                function=fn_name,
                argument=source,
                expected="dict",
                actual=type(value).__name__,
            )

        for key, sub_spec in spec.items():
            if key not in value:
                raise ShapeGuardError(
                    f"Missing key '{key}' in {source}",
                    function=fn_name,
                    argument=source,
                    expected=f"key '{key}'",
                    actual=f"keys: {list(value.keys())}",
                )
            _check_pytree(value[key], sub_spec, ctx, f"{source}[{key!r}]", fn_name)

    elif isinstance(spec, tuple):
        # Spec is a shape tuple - value should be an array
        if not is_array(value):
            raise ShapeGuardError(
                f"Expected array for {source}, got {type(value).__name__}",
                function=fn_name,
                argument=source,
                expected="array",
                actual=type(value).__name__,
            )

        actual = get_shape(value)
        try:
            match_shape(actual, spec, ctx, source)
        except ShapeGuardError as e:
            e.function = fn_name
            raise

    else:
        raise TypeError(
            f"Invalid spec for {source}: {spec!r}. "
            f"Expected tuple (shape) or dict (pytree)."
        )


def expects(
    *,
    jit_mode: JitMode | None = None,
    **shape_specs: PyTreeSpec,
) -> Callable[[F], F]:
    """
    Decorator to validate input shapes on function entry.

    Args:
        jit_mode: Override global config.jit_mode for this function.
            - "check": Always validate, raise on mismatch (default)
            - "warn": Validate, log warning on mismatch, continue
            - "skip": Skip validation under JIT
        **shape_specs: Mapping from argument names to shape specifications

    Returns:
        Decorator function

    Example:
        from shapeguard import Dim, expects

        n, m, k = Dim("n"), Dim("m"), Dim("k")

        @expects(a=(n, m), b=(m, k))
        def matmul(a, b):
            return a @ b

    Shape specifications can contain:
        - int: Exact dimension match (e.g., 128)
        - Dim: Symbolic dimension that unifies across arguments
        - None: Wildcard that accepts any value
        - ...: Ellipsis for variable leading dimensions

    PyTree specs (nested dicts):
        @expects(
            params={"weights": (n, m), "bias": (m,)},
            x=(B, n)
        )
        def apply(params, x): ...

    JIT mode control:
        @expects(x=(n, m), jit_mode="skip")
        @jax.jit
        def fast_layer(x): ...
    """

    def decorator(fn: F) -> F:
        # Get function signature for argument binding
        sig = inspect.signature(fn)
        fn_name = fn.__qualname__

        # Validate that all spec keys are valid argument names
        param_names = set(sig.parameters.keys())
        for arg_name in shape_specs:
            if arg_name not in param_names:
                raise ValueError(
                    f"@expects: '{arg_name}' is not a parameter of {fn_name}. "
                    f"Valid parameters: {sorted(param_names)}"
                )

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Determine effective JIT mode
            effective_mode = jit_mode if jit_mode is not None else config.jit_mode

            # Check if we should skip validation
            if effective_mode == "skip" and is_jax_tracing():
                return fn(*args, **kwargs)

            # Bind arguments to parameter names
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
            except TypeError:
                # Let the original function raise its own error
                return fn(*args, **kwargs)

            # Create unification context for this call
            ctx = UnificationContext()

            # Check each specified argument
            for arg_name, spec in shape_specs.items():
                if arg_name not in bound.arguments:
                    continue

                value = bound.arguments[arg_name]

                try:
                    if isinstance(spec, dict):
                        # PyTree spec
                        _check_pytree(value, spec, ctx, arg_name, fn_name)
                    else:
                        # Regular shape spec
                        if not is_array(value):
                            continue

                        actual = get_shape(value)
                        match_shape(actual, spec, ctx, arg_name)

                except ShapeGuardError as e:
                    # Enrich error with function context
                    e.function = fn_name
                    if e.argument is None:
                        e.argument = arg_name
                    if e.bindings is None:
                        e.bindings = ctx.format_bindings()

                    # Handle based on JIT mode
                    if effective_mode == "warn" and is_jax_tracing():
                        logger.warning(
                            "ShapeGuard validation failed in %s: %s",
                            fn_name,
                            e.reason or str(e),
                        )
                        continue
                    else:
                        raise

            return fn(*args, **kwargs)

        # Attach metadata for introspection
        wrapper.__shapeguard_specs__ = shape_specs  # type: ignore
        wrapper.__shapeguard_jit_mode__ = jit_mode  # type: ignore

        return wrapper  # type: ignore

    return decorator
