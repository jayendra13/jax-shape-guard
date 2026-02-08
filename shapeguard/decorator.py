"""
Contract decorators for shape validation.
"""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from shapeguard._compat import get_shape, is_array, is_jax_tracing
from shapeguard.config import JitMode, config
from shapeguard.core import UnificationContext
from shapeguard.errors import OutputShapeError, ShapeGuardError
from shapeguard.spec import ShapeSpec, match_shape

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
            f"Invalid spec for {source}: {spec!r}. Expected tuple (shape) or dict (pytree)."
        )


def _is_output_tuple_spec(spec: Any) -> bool:
    """Check if spec is a tuple-of-specs (tuple output) vs a flat shape spec."""
    if not isinstance(spec, tuple) or not spec:
        return False
    return isinstance(spec[0], (tuple, dict, list))


def _check_output(
    output: Any,
    spec: Any,
    ctx: UnificationContext,
    fn_name: str,
) -> None:
    """
    Validate a function's return value against an output spec.

    Supports:
    - Single array: spec is a flat shape tuple e.g. (n, k)
    - Tuple of arrays: spec is a tuple of shape tuples e.g. ((n, m), (n,))
    - Dict of arrays: spec is a dict e.g. {"logits": (B, vocab), "h": (B, D)}
    - Scalar: spec is () (empty tuple)
    """
    try:
        if isinstance(spec, dict):
            _check_pytree(output, spec, ctx, "result", fn_name)
        elif _is_output_tuple_spec(spec):
            if not isinstance(output, (tuple, list)):
                raise OutputShapeError(
                    f"Expected tuple output from {fn_name}, got {type(output).__name__}",
                    function=fn_name,
                    expected=f"tuple of {len(spec)} arrays",
                    actual=type(output).__name__,
                )
            if len(output) != len(spec):
                raise OutputShapeError(
                    f"Expected {len(spec)} outputs from {fn_name}, got {len(output)}",
                    function=fn_name,
                    expected=f"tuple of {len(spec)}",
                    actual=f"tuple of {len(output)}",
                )
            for i, (elem, elem_spec) in enumerate(zip(output, spec, strict=True)):
                source = f"result[{i}]"
                if isinstance(elem_spec, dict):
                    _check_pytree(elem, elem_spec, ctx, source, fn_name)
                else:
                    if not is_array(elem):
                        raise OutputShapeError(
                            f"Expected array for {source}, got {type(elem).__name__}",
                            function=fn_name,
                            expected=elem_spec,
                            actual=type(elem).__name__,
                        )
                    actual = get_shape(elem)
                    match_shape(actual, elem_spec, ctx, source)
        else:
            # Single array spec
            if not is_array(output):
                raise OutputShapeError(
                    f"Expected array output from {fn_name}, got {type(output).__name__}",
                    function=fn_name,
                    expected=spec,
                    actual=type(output).__name__,
                )
            actual = get_shape(output)
            match_shape(actual, spec, ctx, "result")
    except ShapeGuardError as e:
        e.function = fn_name
        if e.argument is None:
            e.argument = "result"
        if e.bindings is None:
            e.bindings = ctx.format_bindings()
        raise


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

            # Check if @ensures is stacked — use shared context for output
            ensures_spec = getattr(fn, "__shapeguard_ensures__", None)
            if ensures_spec is not None:
                # Call the original unwrapped function, bypassing ensures wrapper
                original_fn = getattr(fn, "__wrapped__", fn)
                result = original_fn(*args, **kwargs)
                _check_output(result, ensures_spec, ctx, fn_name)
                return result

            return fn(*args, **kwargs)

        # Attach metadata for introspection
        wrapper.__shapeguard_specs__ = shape_specs  # type: ignore
        wrapper.__shapeguard_jit_mode__ = jit_mode  # type: ignore

        return wrapper  # type: ignore

    return decorator


def ensures(
    *,
    result: Any,
    jit_mode: JitMode | None = None,
) -> Callable[[F], F]:
    """
    Decorator to validate output shapes on function return.

    When stacked with @expects (expects on top), bindings from input validation
    carry over to output validation via a shared UnificationContext.

    Args:
        result: Shape spec for the return value. Can be:
            - A shape tuple for a single array: (n, k)
            - A tuple of shape tuples for tuple output: ((n, m), (n,))
            - A dict of shape specs for dict output: {"logits": (B, V), "h": (B, D)}
            - An empty tuple () for scalar output
        jit_mode: Override global config.jit_mode for this function.

    Example:
        @expects(a=(n, m), b=(m, k))
        @ensures(result=(n, k))
        def matmul(a, b):
            return a @ b
    """

    def decorator(fn: F) -> F:
        fn_name = getattr(fn, "__qualname__", str(fn))

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Determine effective JIT mode
            effective_mode = jit_mode if jit_mode is not None else config.jit_mode

            # Check if we should skip validation
            if effective_mode == "skip" and is_jax_tracing():
                return fn(*args, **kwargs)

            output = fn(*args, **kwargs)

            ctx = UnificationContext()

            try:
                _check_output(output, result, ctx, fn_name)
            except ShapeGuardError as e:
                if effective_mode == "warn" and is_jax_tracing():
                    logger.warning(
                        "ShapeGuard output validation failed in %s: %s",
                        fn_name,
                        e.reason or str(e),
                    )
                    return output
                raise

            return output

        # Mark this wrapper so @expects can detect it
        wrapper.__shapeguard_ensures__ = result  # type: ignore

        return wrapper  # type: ignore

    return decorator


def contract(
    *,
    inputs: dict[str, PyTreeSpec],
    output: Any,
    jit_mode: JitMode | None = None,
) -> Callable[[F], F]:
    """
    Combined input + output validation in a single decorator.

    Syntactic sugar for @expects + @ensures with a shared UnificationContext.

    Args:
        inputs: Mapping from argument names to shape specifications.
        output: Shape spec for the return value (same format as @ensures).
        jit_mode: Override global config.jit_mode for this function.

    Example:
        @contract(inputs={"a": (n, m), "b": (m, k)}, output=(n, k))
        def matmul(a, b):
            return a @ b
    """

    def decorator(fn: F) -> F:
        sig = inspect.signature(fn)
        fn_name = fn.__qualname__

        # Validate that all spec keys are valid argument names
        param_names = set(sig.parameters.keys())
        for arg_name in inputs:
            if arg_name not in param_names:
                raise ValueError(
                    f"@contract: '{arg_name}' is not a parameter of {fn_name}. "
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
                return fn(*args, **kwargs)

            # Single shared context for inputs and output
            ctx = UnificationContext()

            # Validate inputs
            for arg_name, spec in inputs.items():
                if arg_name not in bound.arguments:
                    continue

                value = bound.arguments[arg_name]

                try:
                    if isinstance(spec, dict):
                        _check_pytree(value, spec, ctx, arg_name, fn_name)
                    else:
                        if not is_array(value):
                            continue
                        actual = get_shape(value)
                        match_shape(actual, spec, ctx, arg_name)

                except ShapeGuardError as e:
                    e.function = fn_name
                    if e.argument is None:
                        e.argument = arg_name
                    if e.bindings is None:
                        e.bindings = ctx.format_bindings()

                    if effective_mode == "warn" and is_jax_tracing():
                        logger.warning(
                            "ShapeGuard validation failed in %s: %s",
                            fn_name,
                            e.reason or str(e),
                        )
                        continue
                    else:
                        raise

            # Call function and validate output
            result = fn(*args, **kwargs)

            try:
                _check_output(result, output, ctx, fn_name)
            except ShapeGuardError as e:
                if effective_mode == "warn" and is_jax_tracing():
                    logger.warning(
                        "ShapeGuard output validation failed in %s: %s",
                        fn_name,
                        e.reason or str(e),
                    )
                    return result
                raise

            return result

        wrapper.__shapeguard_specs__ = inputs  # type: ignore
        wrapper.__shapeguard_output_spec__ = output  # type: ignore
        wrapper.__shapeguard_jit_mode__ = jit_mode  # type: ignore

        return wrapper  # type: ignore

    return decorator
