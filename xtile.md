# xTile

- NPU에서 사용하기 위한 mlir 기반의 Tile level IR
- DRAM (Global Memory) 데이터는 memref를 SRAM (Local Memory) 데이터는 tensor 사용
- `/home/sjjeong94/projects/llvm-project/build/` 에 빌드되어 있는 mlir 사용
- IR의 기본적인 형태는 https://docs.nvidia.com/cuda/tile-ir/13.2/index.html 문서 참고

## IR Examples

```mlir
func.func @exp(%arg0: memref<2048x16xf32>, %arg1: memref<2048x16xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %zero = arith.constant 0 : i32

  %0 = xt.load(%arg0, %bid_x, %zero) {tile=[16, 16]} : memref<2048x16xf32> -> tensor<16x16xf32>
  %1 = xt.exp(%0) : tensor<16x16xf32>
  xt.store(%1, %arg1, %bid_x, %zero) {tile=[16, 16]} : tensor<16x16xf32> -> memref<2048x16xf32>
  func.return
}

func.func @add(%arg0: memref<2048x16xf32>, %arg1: memref<2048x16xf32>, %arg2: memref<2048x16xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %zero = arith.constant 0 : i32

  %0 = xt.load(%arg0, %bid_x, %zero) {tile=[16, 16]} : memref<2048x16xf32> -> tensor<16x16xf32>
  %1 = xt.load(%arg1, %bid_x, %zero) {tile=[16, 16]} : memref<2048x16xf32> -> tensor<16x16xf32>
  %2 = xt.add(%0, %1) : tensor<16x16xf32>
  xt.store(%2, %arg2, %bid_x, %zero) {tile=[16, 16]} : tensor<16x16xf32> -> memref<2048x16xf32>
  func.return
}
```

### 1D examples
```mlir
func.func @exp(%arg0: memref<2048xf32>, %arg1: memref<2048xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32

  %0 = xt.load(%arg0, %bid_x) {tile=[16]} : memref<2048xf32> -> tensor<16xf32>
  %1 = xt.exp(%0) : tensor<16xf32>
  xt.store(%1, %arg1, %bid_x) {tile=[16]} : tensor<16xf32> -> memref<2048xf32>
  func.return
}

func.func @add(%arg0: memref<2048xf32>, %arg1: memref<2048xf32>, %arg2: memref<2048xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32

  %0 = xt.load(%arg0, %bid_x) {tile=[16]} : memref<2048xf32> -> tensor<16xf32>
  %1 = xt.load(%arg1, %bid_x) {tile=[16]} : memref<2048xf32> -> tensor<16xf32>
  %2 = xt.add(%0, %1) : tensor<16xf32>
  xt.store(%2, %arg2, %bid_x) {tile=[16]} : tensor<16xf32> -> memref<2048xf32>
  func.return
}
```

### 3D examples
```mlir
func.func @exp(%arg0: memref<2048x32x32f32>, %arg1: memref<2048x32x32f32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32

  %0 = xt.load(%arg0, %bid_x, %bid_y, %bid_z) {tile=[16, 16, 16]} : memref<2048x32x32xf32> -> tensor<16x16x16xf32>
  %1 = xt.exp(%0) : tensor<16x16x16xf32>
  xt.store(%1, %arg1, %bid_x, %bid_y, %bid_z) {tile=[16, 16, 16]} : tensor<16x16x16xf32> -> memref<2048x32x32xf32>
  func.return
}
```
