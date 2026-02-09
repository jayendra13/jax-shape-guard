# Configuration

Global configuration and JIT mode settings.

## config

Global `Config` singleton instance. Import and modify directly:

```python
from shapeguard import config

config.jit_mode = "skip"  # Disable checks under JIT globally
```

## Config

::: shapeguard.config.Config

## JitMode

```python
JitMode = Literal["check", "warn", "skip"]
```

Type alias for the three JIT validation modes:

| Mode | Behavior |
|------|----------|
| `"check"` | Always validate, raise on mismatch (default) |
| `"warn"` | Validate, log warning on mismatch, continue |
| `"skip"` | Skip validation entirely under JIT |

See [JIT Modes](../concepts/jit-modes.md) for detailed explanation.
