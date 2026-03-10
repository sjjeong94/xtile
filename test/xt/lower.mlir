// RUN: xt-opt --xt-lower-to-loops %s | FileCheck %s

func.func @lower_1d(%arg0: memref<2048xf32>, %arg1: memref<2048xf32>, %arg2: memref<2048xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %0 = xt.load(%arg0, %bid_x) {tile = [16]} : memref<2048xf32> -> tensor<16xf32>
  %1 = xt.load(%arg1, %bid_x) {tile = [16]} : memref<2048xf32> -> tensor<16xf32>
  %2 = xt.add(%0, %1) : tensor<16xf32>
  xt.store(%2, %arg2, %bid_x) {tile = [16]} : tensor<16xf32> -> memref<2048xf32>
  func.return
}

func.func @lower_all(%arg0: memref<2048x16xf32>, %arg1: memref<2048x16xf32>, %arg2: memref<2048x16xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %zero = arith.constant 0 : i32
  %0 = xt.load(%arg0, %bid_x, %zero) {tile = [16, 16]} : memref<2048x16xf32> -> tensor<16x16xf32>
  %1 = xt.load(%arg1, %bid_x, %zero) {tile = [16, 16]} : memref<2048x16xf32> -> tensor<16x16xf32>
  %2 = xt.add(%0, %1) : tensor<16x16xf32>
  %3 = xt.exp(%2) : tensor<16x16xf32>
  xt.store(%3, %arg2, %bid_x, %zero) {tile = [16, 16]} : tensor<16x16xf32> -> memref<2048x16xf32>
  func.return
}

func.func @lower_3d(%arg0: memref<2048x32x32xf32>, %arg1: memref<2048x32x32xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %0 = xt.load(%arg0, %bid_x, %bid_y, %bid_z) {tile = [16, 16, 16]} : memref<2048x32x32xf32> -> tensor<16x16x16xf32>
  %1 = xt.exp(%0) : tensor<16x16x16xf32>
  xt.store(%1, %arg1, %bid_x, %bid_y, %bid_z) {tile = [16, 16, 16]} : tensor<16x16x16xf32> -> memref<2048x32x32xf32>
  func.return
}

// CHECK-LABEL: func.func @lower_1d
// CHECK-NOT: xt.
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.store
// CHECK-LABEL: func.func @lower_all
// CHECK-NOT: xt.
// CHECK: arith.constant 0 : i32
// CHECK: scf.for
// CHECK: memref.load
// CHECK: math.exp
// CHECK: memref.store
// CHECK-LABEL: func.func @lower_3d
// CHECK-NOT: xt.
// CHECK: scf.for
// CHECK: math.exp
// CHECK: memref.store
