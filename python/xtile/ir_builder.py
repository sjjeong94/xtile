from __future__ import annotations

from mlir import ir

from .ast_parser import (
    BinaryOp,
    ConstOp,
    IntExpr,
    KernelGraph,
    LoadOp,
    ReshapeOp,
    StoreOp,
    TransposeOp,
    UnaryOp,
)


def build_module(graph: KernelGraph) -> ir.Module:
    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with ir.Location.unknown(ctx):
            module = ir.Module.create()
            func_type = ir.FunctionType.get(
                [arg.memref.to_mlir_type() for arg in graph.args],
                [],
            )
            with ir.InsertionPoint(module.body):
                func_op = ir.Operation.create(
                    "func.func",
                    attributes={
                        "sym_name": ir.StringAttr.get(graph.name),
                        "function_type": ir.TypeAttr.get(func_type),
                    },
                    regions=1,
                )

            entry = func_op.regions[0].blocks.append(*func_type.inputs)
            memrefs = {
                arg.name: entry.arguments[index] for index, arg in enumerate(graph.args)
            }
            tiles: dict[str, ir.Value] = {}
            ints: dict[str, ir.Value] = {}
            bid_results: tuple[ir.Value, ...] = ()

            with ir.InsertionPoint(entry):
                if graph.uses_block_ids:
                    bid_op = ir.Operation.create(
                        "xt.get_tile_block_id",
                        results=[ir.IntegerType.get_signless(32)] * 3,
                    )
                    bid_results = tuple(bid_op.results)

                for op in graph.ops:
                    if isinstance(op, ConstOp):
                        ints[op.name] = ir.Operation.create(
                            "arith.constant",
                            results=[ir.IntegerType.get_signless(32)],
                            attributes={
                                "value": ir.IntegerAttr.get(
                                    ir.IntegerType.get_signless(32), op.value
                                )
                            },
                        ).result
                        continue

                    if isinstance(op, LoadOp):
                        operands = [memrefs[op.source]]
                        operands.extend(
                            _resolve_int(expr, ints, bid_results) for expr in op.indices
                        )
                        attributes = {}
                        if op.shared is not None:
                            attributes["shared"] = ir.IntegerAttr.get(
                                ir.IntegerType.get_signless(64), op.shared
                            )
                        tiles[op.name] = ir.Operation.create(
                            "xt.load",
                            operands=operands,
                            results=[op.result.to_mlir_type()],
                            attributes=attributes,
                        ).result
                        continue

                    if isinstance(op, BinaryOp):
                        tiles[op.name] = ir.Operation.create(
                            f"xt.{op.op_name}",
                            operands=[tiles[op.lhs], tiles[op.rhs]],
                            results=[op.result.to_mlir_type()],
                        ).result
                        continue

                    if isinstance(op, UnaryOp):
                        tiles[op.name] = ir.Operation.create(
                            f"xt.{op.op_name}",
                            operands=[tiles[op.operand]],
                            results=[op.result.to_mlir_type()],
                        ).result
                        continue

                    if isinstance(op, ReshapeOp):
                        tiles[op.name] = ir.Operation.create(
                            "xt.reshape",
                            operands=[tiles[op.operand]],
                            results=[op.result.to_mlir_type()],
                        ).result
                        continue

                    if isinstance(op, TransposeOp):
                        tiles[op.name] = ir.Operation.create(
                            "xt.transpose",
                            operands=[tiles[op.operand]],
                            results=[op.result.to_mlir_type()],
                        ).result
                        continue

                    if isinstance(op, StoreOp):
                        operands = [tiles[op.tile], memrefs[op.dest]]
                        operands.extend(
                            _resolve_int(expr, ints, bid_results) for expr in op.indices
                        )
                        ir.Operation.create("xt.store", operands=operands)
                        continue

                ir.Operation.create("func.return")

            return module


def _resolve_int(
    expr: IntExpr, ints: dict[str, ir.Value], bid_results: tuple[ir.Value, ...]
) -> ir.Value:
    if expr.kind == "bid":
        return bid_results[expr.value]
    if expr.name is not None and expr.name in ints:
        return ints[expr.name]
    return ir.Operation.create(
        "arith.constant",
        results=[ir.IntegerType.get_signless(32)],
        attributes={
            "value": ir.IntegerAttr.get(ir.IntegerType.get_signless(32), expr.value)
        },
    ).result
