class XTError(Exception):
    """Base error for xTile Python DSL conversion."""


class XTConversionError(XTError):
    """Raised when a Python kernel cannot be converted to MLIR."""
