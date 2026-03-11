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

  %0 = xt.load(%arg0, %bid_x, %zero) : memref<2048x16xf32> -> tensor<16x16xf32>
  %1 = xt.exp(%0) : tensor<16x16xf32>
  xt.store(%1, %arg1, %bid_x, %zero) : tensor<16x16xf32> -> memref<2048x16xf32>
  func.return
}

func.func @add(%arg0: memref<2048x16xf32>, %arg1: memref<2048x16xf32>, %arg2: memref<2048x16xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %zero = arith.constant 0 : i32

  %0 = xt.load(%arg0, %bid_x, %zero) : memref<2048x16xf32> -> tensor<16x16xf32>
  %1 = xt.load(%arg1, %bid_x, %zero) : memref<2048x16xf32> -> tensor<16x16xf32>
  %2 = xt.add(%0, %1) : tensor<16x16xf32>
  xt.store(%2, %arg2, %bid_x, %zero) : tensor<16x16xf32> -> memref<2048x16xf32>
  func.return
}
```

### 1D examples
```mlir
func.func @exp(%arg0: memref<2048xf32>, %arg1: memref<2048xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32

  %0 = xt.load(%arg0, %bid_x) : memref<2048xf32> -> tensor<16xf32>
  %1 = xt.exp(%0) : tensor<16xf32>
  xt.store(%1, %arg1, %bid_x) : tensor<16xf32> -> memref<2048xf32>
  func.return
}

func.func @add(%arg0: memref<2048xf32>, %arg1: memref<2048xf32>, %arg2: memref<2048xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32

  %0 = xt.load(%arg0, %bid_x) : memref<2048xf32> -> tensor<16xf32>
  %1 = xt.load(%arg1, %bid_x) : memref<2048xf32> -> tensor<16xf32>
  %2 = xt.add(%0, %1) : tensor<16xf32>
  xt.store(%2, %arg2, %bid_x) : tensor<16xf32> -> memref<2048xf32>
  func.return
}
```

### 3D examples
```mlir
func.func @exp(%arg0: memref<2048x32x32f32>, %arg1: memref<2048x32x32f32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32

  %0 = xt.load(%arg0, %bid_x, %bid_y, %bid_z) : memref<2048x32x32xf32> -> tensor<16x16x16xf32>
  %1 = xt.exp(%0) : tensor<16x16x16xf32>
  xt.store(%1, %arg1, %bid_x, %bid_y, %bid_z) : tensor<16x16x16xf32> -> memref<2048x32x32xf32>
  func.return
}
```

### 4D examples
```mlir
func.func @exp(%arg0: memref<64x32x32x32xf32>, %arg1: memref<64x32x32x32xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %zero = arith.constant 0 : i32

  %0 = xt.load(%arg0, %bid_x, %bid_y, %bid_z, %zero) : memref<64x32x32x32xf32> -> tensor<16x16x16x16xf32>
  %1 = xt.exp(%0) : tensor<16x16x16x16xf32>
  xt.store(%1, %arg1, %bid_x, %bid_y, %bid_z, %zero) : tensor<16x16x16x16xf32> -> memref<64x32x32x32xf32>
  func.return
}
```

### Shared examples
```mlir
func.func @shared(%arg0: memref<128x256xi8>, %arg1: memref<256x512xi8>, %arg2: memref<128x512xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %zero = arith.constant 0 : i32

  %0 = xt.load(%arg0, %bid_x, %zero) : memref<128x256xi8> -> tensor<64x256xi8>
  %1 = xt.load(%arg1, %zero, %bid_y) {shared=1} : memref<256x512xi8> -> tensor<256x64xi8>
  %2 = xt.matmul(%0, %1) : (tensor<64x256xi8>, tensor<256x64xi8>) -> tensor<64x64xf32>
  xt.store(%2, %arg2, %bid_x, %bid_y) : tensor<64x64xf32> -> memref<128x512xf32>
  func.return
}
```

### Dynamic shape examples
```mlir
func.func @shared(%arg0: memref<?x256xi8>, %arg1: memref<256x?xi8>, %arg2: memref<128x512xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %zero = arith.constant 0 : i32

  %0 = xt.load(%arg0, %bid_x, %zero) : memref<?x256xi8> -> tensor<64x256xi8>
  %1 = xt.load(%arg1, %zero, %bid_y) {shared=1} : memref<256x?xi8> -> tensor<256x64xi8>
  %2 = xt.matmul(%0, %1) : (tensor<64x256xi8>, tensor<256x64xi8>) -> tensor<64x64xf32>
  xt.store(%2, %arg2, %bid_x, %bid_y) : tensor<64x64xf32> -> memref<128x512xf32>
  func.return
}
```

### Broadcast examples
```mlir
func.func @broadcast(%arg0: memref<2048x16xf32>, %arg1: memref<1x16xf32>, %arg2: memref<2048x16xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %zero = arith.constant 0 : i32

  %0 = xt.load(%arg0, %bid_x, %zero) : memref<2048x16xf32> -> tensor<16x16xf32>
  %1 = xt.load(%arg1, %zero, %zero) {shared=1} : memref<1x16xf32> -> tensor<1x16xf32>
  %2 = xt.add(%0, %1) : tensor<16x16xf32>
  xt.store(%2, %arg2, %bid_x, %zero) : tensor<16x16xf32> -> memref<2048x16xf32>
  func.return
}
```

### conv2d examples
```mlir
func.func @broadcast(%arg0: memref<8x32x64x128xi8>, %arg1: memref<3x3x128x256xi8>, %arg2: memref<8x32x64x256xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %zero = arith.constant 0 : i32

  %0 = xt.load(%arg0, %bid_x, %zero, %zero, %zero) : memref<8x32x64x128xi8> -> tensor<1x32x64x128xi8>
  %1 = xt.load(%arg1, %zero, %zero, %zero, %bid_y) {shared=1} : memref<3x3x128x256xi8> -> tensor<3x3x128x64xi8>
  %2 = xt.conv2d(%0, %1) {pad=[1, 1, 1, 1], stride=[1, 1], dilation=[1, 1]} : (tensor<1x32x64x128xi8>, tensor<3x3x128x64xi8>) -> tensor<1x32x64x64xf32>
  xt.store(%2, %arg2, %bid_x, %zero, %zero, %bid_y) : tensor<1x32x64x64xf32> -> memref<8x32x64x256xf32>
  func.return
}
```
