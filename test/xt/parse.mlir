// RUN: xt-opt %s | FileCheck %s

func.func @generic_tile_ops(%arg0: memref<2048x16xf32>, %arg1: memref<2048x16xf32>) {
  %bid:3 = xt.get_tile_block_id : i32, i32, i32
  %zero = arith.constant 0 : i32
  %0 = xt.load(%arg0, %bid#0, %zero) : (memref<2048x16xf32>, i32, i32) -> tensor<16x16xf32>
  %1 = xt.exp(%0) : tensor<16x16xf32> -> tensor<16x16xf32>
  %2 = xt.add(%1, %1) : tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf32>
  xt.store(%2, %arg1, %bid#0, %zero) : (tensor<16x16xf32>, memref<2048x16xf32>, i32, i32) -> ()
  func.return
}

func.func @generic_shared_and_contract(%arg0: memref<128x256xi8>, %arg1: memref<256x512xi8>, %arg2: memref<128x512xf32>) {
  %bid:3 = xt.get_tile_block_id : i32, i32, i32
  %zero = arith.constant 0 : i32
  %0 = xt.load(%arg0, %bid#0, %zero) : (memref<128x256xi8>, i32, i32) -> tensor<64x256xi8>
  %1 = xt.load(%arg1, %zero, %bid#1) {shared = 1 : i64} : (memref<256x512xi8>, i32, i32) -> tensor<256x64xi8>
  %2 = xt.matmul(%0, %1) : tensor<64x256xi8>, tensor<256x64xi8> -> tensor<64x64xf32>
  xt.store(%2, %arg2, %bid#0, %bid#1) : (tensor<64x64xf32>, memref<128x512xf32>, i32, i32) -> ()
  func.return
}

func.func @generic_mma(%arg0: tensor<16x32xi8>, %arg1: tensor<32x8xi8>, %arg2: tensor<16x8xf32>) -> tensor<16x8xf32> {
  %0 = xt.mma(%arg0, %arg1, %arg2) : tensor<16x32xi8>, tensor<32x8xi8>, tensor<16x8xf32> -> tensor<16x8xf32>
  func.return %0 : tensor<16x8xf32>
}

func.func @generic_reduce(%arg0: tensor<16x16xf32>) -> tensor<16x1xf32> {
  %0 = xt.reduce_sum(%arg0) {axis = 1 : i64} : tensor<16x16xf32> -> tensor<16x1xf32>
  func.return %0 : tensor<16x1xf32>
}

func.func @generic_load_conv2d(%arg0: memref<1x34x66x128xi8>, %arg1: tensor<3x3x128x64xi8>) -> tensor<1x32x64x32xf32> {
  %c0 = arith.constant 0 : i32
  %2 = xt.load_conv2d(%arg0, %arg1, %c0, %c0, %c0, %c0) {dilation = array<i64: 1, 1>, group = 1 : i64, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (memref<1x34x66x128xi8>, tensor<3x3x128x64xi8>, i32, i32, i32, i32) -> tensor<1x32x64x32xf32>
  func.return %2 : tensor<1x32x64x32xf32>
}

func.func @generic_reshape_transpose(%arg0: tensor<64x16xf32>) -> tensor<64x16xf32> {
  %0 = xt.reshape(%arg0) : tensor<64x16xf32> -> tensor<2x32x16xf32>
  %1 = xt.transpose(%0) : tensor<2x32x16xf32> -> tensor<2x16x32xf32>
  %2 = xt.reshape(%1) : tensor<2x16x32xf32> -> tensor<64x16xf32>
  func.return %2 : tensor<64x16xf32>
}

func.func @generic_permute(%arg0: tensor<2x3x5xf32>) -> tensor<5x2x3xf32> {
  %0 = xt.permute(%arg0) {permutation = [2, 0, 1]} : tensor<2x3x5xf32> -> tensor<5x2x3xf32>
  func.return %0 : tensor<5x2x3xf32>
}

func.func @kernel_with_grid() attributes {xt.grid = array<i32: 32, 8, 1>} {
  func.return
}

func.func @cast_ops(%arg0: tensor<5x16xi8>, %arg1: tensor<5x16xf32>) {
  %0 = xt.itof(%arg0) : tensor<5x16xi8> -> tensor<5x16xf32>
  %1 = xt.ftoi(%arg1) : tensor<5x16xf32> -> tensor<5x16xi8>
  func.return
}

// CHECK-LABEL: func.func @generic_tile_ops
// CHECK: %[[X:.*]], %[[Y:.*]], %[[Z:.*]] = xt.get_tile_block_id : i32, i32, i32
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK: %[[LOAD:.*]] = xt.load(%arg0, %[[X]], %[[ZERO]]) : (memref<2048x16xf32>, i32, i32) -> tensor<16x16xf32>
// CHECK: %[[EXP:.*]] = xt.exp(%[[LOAD]]) : tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK: %[[ADD:.*]] = xt.add(%[[EXP]], %[[EXP]]) : tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK: xt.store(%[[ADD]], %arg1, %[[X]], %[[ZERO]]) : (tensor<16x16xf32>, memref<2048x16xf32>, i32, i32) -> ()
// CHECK-LABEL: func.func @generic_shared_and_contract
// CHECK: %[[SHARED_X:.*]], %[[SHARED_Y:.*]], %[[SHARED_Z:.*]] = xt.get_tile_block_id : i32, i32, i32
// CHECK: %[[SHARED_ZERO:.*]] = arith.constant 0 : i32
// CHECK: %[[LHS:.*]] = xt.load(%arg0, %[[SHARED_X]], %[[SHARED_ZERO]]) : (memref<128x256xi8>, i32, i32) -> tensor<64x256xi8>
// CHECK: %[[RHS:.*]] = xt.load(%arg1, %[[SHARED_ZERO]], %[[SHARED_Y]]) {shared = 1 : i64} : (memref<256x512xi8>, i32, i32) -> tensor<256x64xi8>
// CHECK: %[[MM:.*]] = xt.matmul(%[[LHS]], %[[RHS]]) : tensor<64x256xi8>, tensor<256x64xi8> -> tensor<64x64xf32>
// CHECK: xt.store(%[[MM]], %arg2, %[[SHARED_X]], %[[SHARED_Y]]) : (tensor<64x64xf32>, memref<128x512xf32>, i32, i32) -> ()
// CHECK-LABEL: func.func @generic_mma
// CHECK: %[[MMA:.*]] = xt.mma(%arg0, %arg1, %arg2) : tensor<16x32xi8>, tensor<32x8xi8>, tensor<16x8xf32> -> tensor<16x8xf32>
// CHECK-LABEL: func.func @generic_reduce
// CHECK: %[[SUM:.*]] = xt.reduce_sum(%arg0) {axis = 1 : i64} : tensor<16x16xf32> -> tensor<16x1xf32>
// CHECK-LABEL: func.func @generic_load_conv2d
// CHECK: %[[C0:.*]] = arith.constant 0 : i32
// CHECK: %[[LCONV:.*]] = xt.load_conv2d(%arg0, %arg1, %[[C0]], %[[C0]], %[[C0]], %[[C0]]) {dilation = array<i64: 1, 1>, group = 1 : i64, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (memref<1x34x66x128xi8>, tensor<3x3x128x64xi8>, i32, i32, i32, i32) -> tensor<1x32x64x32xf32>
// CHECK-LABEL: func.func @generic_reshape_transpose
// CHECK: %[[RESHAPE0:.*]] = xt.reshape(%arg0) : tensor<64x16xf32> -> tensor<2x32x16xf32>
// CHECK: %[[TRANSPOSE:.*]] = xt.transpose(%[[RESHAPE0]]) : tensor<2x32x16xf32> -> tensor<2x16x32xf32>
// CHECK: %[[RESHAPE1:.*]] = xt.reshape(%[[TRANSPOSE]]) : tensor<2x16x32xf32> -> tensor<64x16xf32>
// CHECK-LABEL: func.func @generic_permute
// CHECK: %[[PERMUTE:.*]] = xt.permute(%arg0) {permutation = [2, 0, 1]} : tensor<2x3x5xf32> -> tensor<5x2x3xf32>
// CHECK-LABEL: func.func @kernel_with_grid() attributes {xt.grid = array<i32: 32, 8, 1>}
// CHECK-LABEL: func.func @cast_ops
// CHECK: %[[ITOF:.*]] = xt.itof(%arg0) : tensor<5x16xi8> -> tensor<5x16xf32>
// CHECK: %[[FTOI:.*]] = xt.ftoi(%arg1) : tensor<5x16xf32> -> tensor<5x16xi8>
