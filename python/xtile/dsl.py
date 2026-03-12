from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from .errors import XTError
from .types import memref


F = TypeVar("F", bound=Callable[..., object])


def kernel(fn: F) -> F:
    setattr(fn, "__xt_kernel__", True)
    return fn


def bid(dim: int) -> int:
    raise XTError(f"xt.bid({dim}) is only valid during xt.convert(...)")


def load(*args: object, **kwargs: object) -> object:
    raise XTError("xt.load(...) is only valid during xt.convert(...)")


def store(*args: object, **kwargs: object) -> None:
    raise XTError("xt.store(...) is only valid during xt.convert(...)")


def exp(*args: object, **kwargs: object) -> object:
    raise XTError("xt.exp(...) is only valid during xt.convert(...)")


def reshape(*args: object, **kwargs: object) -> object:
    raise XTError("xt.reshape(...) is only valid during xt.convert(...)")


def transpose(*args: object, **kwargs: object) -> object:
    raise XTError("xt.transpose(...) is only valid during xt.convert(...)")


__all__ = [
    "bid",
    "exp",
    "kernel",
    "load",
    "memref",
    "reshape",
    "store",
    "transpose",
]
