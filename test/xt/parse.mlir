// RUN: xt-opt %s | FileCheck %s

func.func @exp_1d(%arg0: memref<2048xf32>, %arg1: memref<2048xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %0 = xt.load(%arg0, %bid_x) {tile = [16]} : memref<2048xf32> -> tensor<16xf32>
  %1 = xt.exp(%0) : tensor<16xf32>
  xt.store(%1, %arg1, %bid_x) {tile = [16]} : tensor<16xf32> -> memref<2048xf32>
  func.return
}

func.func @exp(%arg0: memref<2048x16xf32>, %arg1: memref<2048x16xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %zero = arith.constant 0 : i32

  %0 = xt.load(%arg0, %bid_x, %zero) {tile = [16, 16]} : memref<2048x16xf32> -> tensor<16x16xf32>
  %1 = xt.exp(%0) : tensor<16x16xf32>
  xt.store(%1, %arg1, %bid_x, %zero) {tile = [16, 16]} : tensor<16x16xf32> -> memref<2048x16xf32>
  func.return
}

func.func @add(%arg0: memref<2048x16xf32>, %arg1: memref<2048x16xf32>, %arg2: memref<2048x16xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %zero = arith.constant 0 : i32

  %0 = xt.load(%arg0, %bid_x, %zero) {tile = [16, 16]} : memref<2048x16xf32> -> tensor<16x16xf32>
  %1 = xt.load(%arg1, %bid_x, %zero) {tile = [16, 16]} : memref<2048x16xf32> -> tensor<16x16xf32>
  %2 = xt.add(%0, %1) : tensor<16x16xf32>
  xt.store(%2, %arg2, %bid_x, %zero) {tile = [16, 16]} : tensor<16x16xf32> -> memref<2048x16xf32>
  func.return
}

func.func @exp_3d(%arg0: memref<2048x32x32xf32>, %arg1: memref<2048x32x32xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %0 = xt.load(%arg0, %bid_x, %bid_y, %bid_z) {tile = [16, 16, 16]} : memref<2048x32x32xf32> -> tensor<16x16x16xf32>
  %1 = xt.exp(%0) : tensor<16x16x16xf32>
  xt.store(%1, %arg1, %bid_x, %bid_y, %bid_z) {tile = [16, 16, 16]} : tensor<16x16x16xf32> -> memref<2048x32x32xf32>
  func.return
}

// CHECK-LABEL: func.func @exp_1d
// CHECK: %[[BIDX1:.*]], %[[BIDY1:.*]], %[[BIDZ1:.*]] = xt.get_tile_block_id() : i32
// CHECK: %[[LOAD1:.*]] = xt.load(%arg0, %[[BIDX1]]) {tile = [16]} : memref<2048xf32> -> tensor<16xf32>
// CHECK: %[[EXP1:.*]] = xt.exp(%[[LOAD1]]) : tensor<16xf32>
// CHECK: xt.store(%[[EXP1]], %arg1, %[[BIDX1]]) {tile = [16]} : tensor<16xf32> -> memref<2048xf32>
// CHECK: %[[BIDX:.*]], %[[BIDY:.*]], %[[BIDZ:.*]] = xt.get_tile_block_id() : i32
// CHECK: %[[LOAD:.*]] = xt.load(%arg0, %[[BIDX]], %[[ZERO:.*]]) {tile = [16, 16]} : memref<2048x16xf32> -> tensor<16x16xf32>
// CHECK: %[[EXP:.*]] = xt.exp(%[[LOAD]]) : tensor<16x16xf32>
// CHECK: xt.store(%[[EXP]], %arg1, %[[BIDX]], %[[ZERO]]) {tile = [16, 16]} : tensor<16x16xf32> -> memref<2048x16xf32>
// CHECK-LABEL: func.func @exp_3d
// CHECK: %[[BIDX3:.*]], %[[BIDY3:.*]], %[[BIDZ3:.*]] = xt.get_tile_block_id() : i32
// CHECK: %[[LOAD3:.*]] = xt.load(%arg0, %[[BIDX3]], %[[BIDY3]], %[[BIDZ3]]) {tile = [16, 16, 16]} : memref<2048x32x32xf32> -> tensor<16x16x16xf32>
// CHECK: %[[EXP3:.*]] = xt.exp(%[[LOAD3]]) : tensor<16x16x16xf32>
// CHECK: xt.store(%[[EXP3]], %arg1, %[[BIDX3]], %[[BIDY3]], %[[BIDZ3]]) {tile = [16, 16, 16]} : tensor<16x16x16xf32> -> memref<2048x32x32xf32>
