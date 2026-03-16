// RUN: xt-opt --nova-allocate %s | FileCheck %s

func.func @allocate_basic(%src: memref<64x16xf32>, %dst: memref<64x16xf32>) {
  %0 = "nova.load"(%src) <{index = array<i64: 0, 0>}> : (memref<64x16xf32>) -> tensor<16x16xf32>
  %1 = "nova.square"(%0) : (tensor<16x16xf32>) -> tensor<16x16xf32>
  "nova.store"(%1, %dst) <{index = array<i64: 0, 0>}> : (tensor<16x16xf32>, memref<64x16xf32>) -> ()
  %2 = "nova.load"(%src) <{index = array<i64: 1, 0>}> : (memref<64x16xf32>) -> tensor<16x16xf32>
  %3 = "nova.square"(%2) : (tensor<16x16xf32>) -> tensor<16x16xf32>
  %4 = arith.constant dense<1.0> : tensor<1x1xf32>
  "nova.store"(%3, %dst) <{index = array<i64: 1, 0>}> : (tensor<16x16xf32>, memref<64x16xf32>) -> ()
  "nova.free"(%3) : (tensor<16x16xf32>) -> ()
  func.return
}

func.func @allocate_multi_bank(%a: tensor<70000xf32>, %b: tensor<16x16xf32>, %dst: memref<16x16xf32>) {
  %0 = nova.square(%a) : tensor<70000xf32> -> tensor<70000xf32>
  %1 = nova.square(%b) : tensor<16x16xf32> -> tensor<16x16xf32>
  nova.store(%1, %dst) {index = array<i64: 0, 0>} : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
  nova.free(%0) : tensor<70000xf32>
  func.return
}

// CHECK-LABEL: func.func @allocate_basic
// CHECK: %[[LOAD0:.*]] = nova.load(%arg0) {index = array<i64: 0, 0>} : memref<64x16xf32> -> tensor<16x16xf32, {bank = 0 : i64}>
// CHECK: %[[SQUARE0:.*]] = nova.square(%[[LOAD0]]) : tensor<16x16xf32, {bank = 0 : i64}> -> tensor<16x16xf32, {bank = 2 : i64}>
// CHECK: nova.store(%[[SQUARE0]], %arg1) {index = array<i64: 0, 0>} : (tensor<16x16xf32, {bank = 2 : i64}>, memref<64x16xf32>) -> ()
// CHECK: %[[LOAD1:.*]] = nova.load(%arg0) {index = array<i64: 1, 0>} : memref<64x16xf32> -> tensor<16x16xf32, {bank = 0 : i64}>
// CHECK: %[[SQUARE1:.*]] = nova.square(%[[LOAD1]]) : tensor<16x16xf32, {bank = 0 : i64}> -> tensor<16x16xf32, {bank = 2 : i64}>
// CHECK: %[[SCALAR:.*]] = arith.constant dense<1.000000e+00> : tensor<1x1xf32>
// CHECK: nova.store(%[[SQUARE1]], %arg1) {index = array<i64: 1, 0>} : (tensor<16x16xf32, {bank = 2 : i64}>, memref<64x16xf32>) -> ()
// CHECK: nova.free(%[[SQUARE1]]) : tensor<16x16xf32, {bank = 2 : i64}>
// CHECK-LABEL: func.func @allocate_multi_bank
// CHECK: %[[LARGE:.*]] = nova.square(%arg0) : tensor<70000xf32> -> tensor<70000xf32, {bank = 0 : i64}>
// CHECK: %[[SMALL:.*]] = nova.square(%arg1) : tensor<16x16xf32> -> tensor<16x16xf32, {bank = 4 : i64}>
// CHECK: nova.store(%[[SMALL]], %arg2) {index = array<i64: 0, 0>} : (tensor<16x16xf32, {bank = 4 : i64}>, memref<16x16xf32>) -> ()
// CHECK: nova.free(%[[LARGE]]) : tensor<70000xf32, {bank = 0 : i64}>
