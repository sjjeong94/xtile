# Nova

- NPU hardware specific mlir
- xTile에서 lowering 되어서 생성
- xTile 보다 hardware 정보들이 더 추가됨

## Conversion examples

before conversion
```mlir
module {
  func.func @softmax_kernel(%arg0: memref<?xf32>, %arg1: memref<?xf32>) {
    %0:3 = "xt.get_tile_block_id"() : () -> (i32, i32, i32)
    %1 = "xt.load"(%arg0, %0#0) : (memref<?xf32>, i32) -> tensor<16x16xf32>
    %2 = "xt.reduce_max"(%1) : (tensor<16x16xf32>) -> tensor<16x1xf32>
    %3 = "xt.sub"(%1, %2) : (tensor<16x16xf32>, tensor<16x1xf32>) -> tensor<16x16xf32>
    %4 = "xt.exp"(%3) : (tensor<16x16xf32>) -> tensor<16x16xf32>
    %5 = "xt.reduce_sum"(%4) : (tensor<16x16xf32>) -> tensor<16x1xf32>
    %6 = "xt.reciprocal"(%5) : (tensor<16x1xf32>) -> tensor<16x1xf32>
    %7 = "xt.mul"(%4, %6) : (tensor<16x16xf32>, tensor<16x1xf32>) -> tensor<16x16xf32>
    "xt.store"(%7, %arg1, %0#0) : (tensor<16x16xf32>, memref<?xf32>, i32) -> ()
    return
  }
}
```

after conversion
```mlir
module {
  func.func @softmax_kernel(%arg0: memref<?xf32>, %arg1: memref<?xf32>) {
    %0:3 = "xt.get_tile_block_id"() : () -> (i32, i32, i32)
    %1 = "xt.load"(%arg0, %0#0) : (memref<?xf32>, i32) -> tensor<16x16xf32>
    %2 = "nova.reduce"(%1) {mode = 1 : i64}: (tensor<16x16xf32>) -> tensor<16x1xf32>
    %3 = "nova.broadcast"(%1, %2) {mode = 3: i64, lhs_s = 1.0 : f32, lhs_b = 0.0: f32, rhs_s = 1.0 : f32, rhs_b = 0.0 : f32} : (tensor<16x16xf32>, tensor<16x1xf32>) -> tensor<16x16xf32>
    %4 = "xt.exp"(%3) : (tensor<16x16xf32>) -> tensor<16x16xf32>
    %5 = "nova.reduce"(%4) {mode = 0 : i64}: (tensor<16x16xf32>) -> tensor<16x1xf32>
    %6 = "xt.reciprocal"(%5) : (tensor<16x1xf32>) -> tensor<16x1xf32>
    %7 = "nova.broadcast"(%4, %6) {mode = 2: i64, lhs_s = 1.0 : f32, lhs_b = 0.0: f32, rhs_s = 1.0 : f32, rhs_b = 0.0 : f32} : (tensor<16x16xf32>, tensor<16x1xf32>) -> tensor<16x16xf32>
    "xt.store"(%7, %arg1, %0#0) : (tensor<16x16xf32>, memref<?xf32>, i32) -> ()
    return
  }
}
```
