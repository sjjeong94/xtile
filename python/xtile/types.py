from __future__ import annotations

from dataclasses import dataclass
import re

from mlir import ir

from .errors import XTConversionError


_MEMREF_RE = re.compile(r"^(?P<body>(?:[0-9?]+x)*)(?P<elem>[a-z][a-z0-9]+)$")


def _build_element_type(type_name: str) -> ir.Type:
    if type_name == "f32":
        return ir.F32Type.get()
    if type_name == "f64":
        return ir.F64Type.get()
    if type_name == "i8":
        return ir.IntegerType.get_signless(8)
    if type_name == "i16":
        return ir.IntegerType.get_signless(16)
    if type_name == "i32":
        return ir.IntegerType.get_signless(32)
    if type_name == "i64":
        return ir.IntegerType.get_signless(64)
    raise XTConversionError(f"unsupported element type: {type_name}")


@dataclass(frozen=True)
class MemRefAnnotation:
    spec: str


def memref(spec: str) -> MemRefAnnotation:
    return MemRefAnnotation(spec)


@dataclass(frozen=True)
class MemRefSpec:
    shape: tuple[int | None, ...]
    element_type: str

    @classmethod
    def parse(cls, annotation: object) -> "MemRefSpec":
        if isinstance(annotation, MemRefAnnotation):
            spec = annotation.spec
        elif isinstance(annotation, str):
            spec = annotation
        else:
            raise XTConversionError(
                "kernel arguments must use xt.memref(...) annotations"
            )

        match = _MEMREF_RE.fullmatch(spec)
        if match is None:
            raise XTConversionError(f"invalid memref annotation: {spec}")

        body = match.group("body")
        dims = []
        if body:
            for token in body[:-1].split("x"):
                dims.append(None if token == "?" else int(token))
        return cls(shape=tuple(dims), element_type=match.group("elem"))

    def to_mlir_type(self) -> ir.MemRefType:
        dyn = ir.ShapedType.get_dynamic_size()
        shape = [dyn if dim is None else dim for dim in self.shape]
        return ir.MemRefType.get(shape, _build_element_type(self.element_type))


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    element_type: str

    def to_mlir_type(self) -> ir.RankedTensorType:
        return ir.RankedTensorType.get(self.shape, _build_element_type(self.element_type))

    def reshape(self, shape: tuple[int, ...]) -> "TensorSpec":
        src_elems = _element_count(self.shape)
        dst_elems = _element_count(shape)
        if src_elems != dst_elems:
            raise XTConversionError(
                f"reshape changes element count: {self.shape} -> {shape}"
            )
        return TensorSpec(shape=shape, element_type=self.element_type)

    def transpose(self, order: tuple[int, ...]) -> "TensorSpec":
        if len(order) != len(self.shape):
            raise XTConversionError(
                f"transpose order rank mismatch: rank {len(self.shape)} vs {order}"
            )
        if sorted(order) != list(range(len(self.shape))):
            raise XTConversionError(f"invalid transpose order: {order}")
        return TensorSpec(
            shape=tuple(self.shape[index] for index in order),
            element_type=self.element_type,
        )

    def reduce_last_dim(self) -> "TensorSpec":
        if not self.shape:
            raise XTConversionError("reduce requires rank-1 or higher tensors")
        return TensorSpec(
            shape=(*self.shape[:-1], 1),
            element_type=self.element_type,
        )


def _element_count(shape: tuple[int, ...]) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return count
