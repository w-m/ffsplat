"""Exception classes for attribute transforms."""


class CombineError(Exception):
    """Errors related to the CombineTransform operations."""

    pass


class CodebookConfigError(CombineError):
    """Error when codebook attributes are not properly configured."""

    def __init__(self) -> None:
        super().__init__("Must set codebook_attr and index_attr for lookup")


class UnknownCombineMethodError(CombineError):
    """Error when an invalid combine method is specified."""

    def __init__(self, method: str) -> None:
        super().__init__(f"Unknown combine method: {method}")


class RescalingError(Exception):
    """Errors related to the RescalingTransform operations."""

    pass


class ZeroRangeError(RescalingError):
    """Error when the rescaling range is zero."""

    def __init__(self) -> None:
        super().__init__("Zero range in rescaling transform")


class UnknownRemappingMethodError(Exception):
    """Error when an invalid remapping method is specified."""

    def __init__(self, method: str) -> None:
        super().__init__(f"Unknown remapping method: {method}")


class ShapeMismatchError(RescalingError):
    """Error when tensor shapes don't match as expected."""

    def __init__(self, min_shape: tuple, max_shape: tuple) -> None:
        super().__init__(f"min_val shape {min_shape} must match max_val shape {max_shape}")


class InsufficientDimensionsError(RescalingError):
    """Error when data tensor has fewer dimensions than required."""

    def __init__(self, data_dims: int, required_dims: int, shape: tuple) -> None:
        super().__init__(
            f"Data has {data_dims} dimensions but needs at least {required_dims} "
            f"to match min/max values of shape {shape}"
        )


class TrailingDimensionMismatchError(RescalingError):
    """Error when trailing dimensions of data don't match min/max dimensions."""

    def __init__(self, data_dims: tuple, required_dims: tuple) -> None:
        super().__init__(
            f"Data's last {len(required_dims)} dimensions {data_dims} must match min/max value shape {required_dims}"
        )


class BackwardNotImplementedError(Exception):
    """Error when a transform's backward operation is not implemented."""

    def __init__(self, transform_name: str) -> None:
        super().__init__(f"{transform_name} backward not implemented")


class MissingContextError(Exception):
    """Error when context is required but not provided."""

    def __init__(self) -> None:
        super().__init__("Context required when using combine transform")


class DirectAccessError(Exception):
    """Error when trying to directly access transformed attributes."""

    def __init__(self, attr_name: str) -> None:
        super().__init__(
            f"Direct access to transformed {attr_name} is not supported. "
            "Use to_torch() to get all transformed tensors instead."
        )
