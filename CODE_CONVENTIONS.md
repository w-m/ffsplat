# Python Code Conventions

This document outlines our code conventions and best practices, with practical examples from our codebase.

## Type Annotations

### Modern Python Type Hints (Python 3.9+)

Use built-in collection types for annotations instead of importing from `typing`:

```python
# ❌ Don't do this
from typing import Dict, List, Union, Optional
def process_data(items: List[str], config: Dict[str, Optional[int]]) -> Union[str, None]:
    pass

# ✅ Do this instead
def process_data(items: list[str], config: dict[str, int | None]) -> str | None:
    pass
```

### Union Types

Use the `|` operator instead of `Union` for combining types:

```python
# ❌ Don't do this
from typing import Union
def transform(data: Union[str, int]) -> Union[float, None]:
    pass

# ✅ Do this instead
def transform(data: str | int) -> float | None:
    pass
```

### Optional Types

Use `type | None` instead of `Optional[type]`:

```python
# ❌ Don't do this
from typing import Optional
def configure(setting: Optional[str] = None):
    pass

# ✅ Do this instead
def configure(setting: str | None = None):
    pass
```

### Array Dimensions with Static Type Checkers

When using jaxtyping with mypy, include a space before single dimensions to prevent conflicts:

```python
# ❌ Don't do this
opacities: Float[NDArray, "N"]  # May cause issues with static type checkers

# ✅ Do this instead
opacities: Float[NDArray, " N"]  # Space before N prevents conflicts
```

## PyTorch Patterns

### Device Management

When managing device placement in PyTorch:

```python
# ❌ Don't do this
class MyModule:
    def to(self, device: str) -> None:
        self.device = device
        self.tensor = self.tensor.to(device)

# ✅ Do this instead
class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("tensor", torch.zeros(10))  # Moves with module

    @property
    def device(self) -> str:
        """Get the current device as a string."""
        return str(self.tensor.device)
```

### Transform Implementations

When implementing transforms:

```python
# ❌ Don't do this
class Transform:
    def forward(self, x):
        return x + 1
    def backward(self, x):
        return x - 1

# ✅ Do this instead
class Transform(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x + 1
    def inverse(self, x: Tensor) -> Tensor:
        """Explicit inverse transform."""
        return x - 1
```

## Type Safety

### Type Variables and Bounds

Use TypeVar with bounds to enforce type constraints:

```python
# ❌ Don't do this
T = TypeVar('T')  # Too permissive

# ✅ Do this instead
T = TypeVar('T', bound=NamedAttribute)  # Enforces type constraint
```

### Collection Type Hints

Use abstract base classes from collections.abc for better type safety:

```python
# ❌ Don't do this
def process(context: dict[str, Tensor]) -> None:
    pass

# ✅ Do this instead
from collections.abc import Mapping
def process(context: Mapping[str, Tensor]) -> None:
    pass
```
