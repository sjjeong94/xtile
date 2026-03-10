// RUN: xt-opt %s | FileCheck %s

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

// CHECK: %[[BIDX:.*]], %[[BIDY:.*]], %[[BIDZ:.*]] = xt.get_tile_block_id() : i32
// CHECK: %[[LOAD:.*]] = xt.load(%arg0, %[[BIDX]], %[[ZERO:.*]]) {tile = [16, 16]} : memref<2048x16xf32> -> tensor<16x16xf32>
// CHECK: %[[EXP:.*]] = xt.exp(%[[LOAD]]) : tensor<16x16xf32>
// CHECK: xt.store(%[[EXP]], %arg1, %[[BIDX]], %[[ZERO]]) {tile = [16, 16]} : tensor<16x16xf32> -> memref<2048x16xf32>
