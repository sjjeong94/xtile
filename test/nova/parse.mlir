// RUN: xt-opt %s | FileCheck %s

func.func @parse_square(%arg0: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %0 = nova.square(%arg0) : tensor<16x16xf32> -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}

func.func @parse_free(%arg0: tensor<16x16xf32>) {
  nova.free(%arg0) : tensor<16x16xf32>
  func.return
}

func.func @parse_load_store(%arr: memref<128x16xf32>, %tile: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %0 = nova.load(%arr) {index = array<i64: 0, 2>, shared = 1 : i64} : memref<128x16xf32> -> tensor<16x16xf32>
  nova.store(%tile, %arr) {index = array<i64: 1, 2>, shared = 1 : i64} : (tensor<16x16xf32>, memref<128x16xf32>) -> ()
  func.return %0 : tensor<16x16xf32>
}

// CHECK-LABEL: func.func @parse_square
// CHECK: %[[RES:.*]] = nova.square(%arg0) : tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK: return %[[RES]] : tensor<16x16xf32>
// CHECK-LABEL: func.func @parse_free
// CHECK: nova.free(%arg0) : tensor<16x16xf32>
// CHECK: return
// CHECK-LABEL: func.func @parse_load_store
// CHECK: %[[LOAD:.*]] = nova.load(%arg0) {index = array<i64: 0, 2>, shared = 1 : i64} : memref<128x16xf32> -> tensor<16x16xf32>
// CHECK: nova.store(%arg1, %arg0) {index = array<i64: 1, 2>, shared = 1 : i64} : (tensor<16x16xf32>, memref<128x16xf32>) -> ()
// CHECK: return %[[LOAD]] : tensor<16x16xf32>
