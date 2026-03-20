// RUN: xt-opt --nova-allocate %s | FileCheck %s

func.func @allocate_basic(%src: memref<64x16xf32>, %dst: memref<64x16xf32>) {
  %0 = nova.load(%src) {start = array<i64: 0, 0>} : memref<64x16xf32> -> tensor<16x16xf32>
  %1 = nova.square(%0) : tensor<16x16xf32> -> tensor<16x16xf32>
  nova.store(%1, %dst) {start = array<i64: 0, 0>} : (tensor<16x16xf32>, memref<64x16xf32>) -> ()
  %2 = nova.load(%src) {start = array<i64: 1, 0>} : memref<64x16xf32> -> tensor<16x16xf32>
  %3 = nova.square(%2) : tensor<16x16xf32> -> tensor<16x16xf32>
  %4 = arith.constant dense<1.0> : tensor<1x1xf32>
  nova.store(%3, %dst) {start = array<i64: 1, 0>} : (tensor<16x16xf32>, memref<64x16xf32>) -> ()
  nova.free(%3) : tensor<16x16xf32>
  func.return
}

func.func @allocate_multi_bank(%a: tensor<70000xf32>, %b: tensor<16x16xf32>, %dst: memref<16x16xf32>) {
  %0 = nova.square(%a) : tensor<70000xf32> -> tensor<70000xf32>
  %1 = nova.square(%b) : tensor<16x16xf32> -> tensor<16x16xf32>
  nova.store(%1, %dst) {start = array<i64: 0, 0>} : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
  nova.free(%0) : tensor<70000xf32>
  func.return
}

func.func @allocate_space_assignment(%lhs_src: memref<16x32xf32>, %rhs_src: memref<32x8xf32>, %scale_src: memref<16x8xf32>, %bias_src: memref<16x8xf32>) {
  %lhs = nova.load(%lhs_src) {start = array<i64: 0, 0>} : memref<16x32xf32> -> tensor<16x32xf32>
  %rhs = nova.load(%rhs_src) {start = array<i64: 0, 0>} : memref<32x8xf32> -> tensor<32x8xf32>
  %scale = nova.load(%scale_src) {start = array<i64: 0, 0>} : memref<16x8xf32> -> tensor<16x8xf32>
  %bias = nova.load(%bias_src) {start = array<i64: 0, 0>} : memref<16x8xf32> -> tensor<16x8xf32>
  %result = nova.matmul(%lhs, %rhs, %scale, %bias) : tensor<16x32xf32>, tensor<32x8xf32>, tensor<16x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
  nova.free(%result) : tensor<16x8xf32>
  func.return
}

// CHECK-LABEL: func.func @allocate_basic
// CHECK: %[[LOAD0:.*]] = nova.load(%arg0) {start = array<i64: 0, 0>} : memref<64x16xf32> -> tensor<16x16xf32, {bank = 0 : i64, space = 3 : i64}>
// CHECK: %[[SQUARE0:.*]] = nova.square(%[[LOAD0]]) : tensor<16x16xf32, {bank = 0 : i64, space = 3 : i64}> -> tensor<16x16xf32, {bank = 2 : i64, space = 3 : i64}>
// CHECK: nova.store(%[[SQUARE0]], %arg1) {start = array<i64: 0, 0>} : (tensor<16x16xf32, {bank = 2 : i64, space = 3 : i64}>, memref<64x16xf32>) -> ()
// CHECK: %[[LOAD1:.*]] = nova.load(%arg0) {start = array<i64: 1, 0>} : memref<64x16xf32> -> tensor<16x16xf32, {bank = 0 : i64, space = 3 : i64}>
// CHECK: %[[SQUARE1:.*]] = nova.square(%[[LOAD1]]) : tensor<16x16xf32, {bank = 0 : i64, space = 3 : i64}> -> tensor<16x16xf32, {bank = 2 : i64, space = 3 : i64}>
// CHECK: %[[SCALAR:.*]] = arith.constant dense<1.000000e+00> : tensor<1x1xf32>
// CHECK: nova.store(%[[SQUARE1]], %arg1) {start = array<i64: 1, 0>} : (tensor<16x16xf32, {bank = 2 : i64, space = 3 : i64}>, memref<64x16xf32>) -> ()
// CHECK-NOT: nova.free(
// CHECK-LABEL: func.func @allocate_multi_bank
// CHECK: %[[LARGE:.*]] = nova.square(%arg0) : tensor<70000xf32> -> tensor<70000xf32, {bank = 0 : i64, space = 3 : i64}>
// CHECK: %[[SMALL:.*]] = nova.square(%arg1) : tensor<16x16xf32> -> tensor<16x16xf32, {bank = 4 : i64, space = 3 : i64}>
// CHECK: nova.store(%[[SMALL]], %arg2) {start = array<i64: 0, 0>} : (tensor<16x16xf32, {bank = 4 : i64, space = 3 : i64}>, memref<16x16xf32>) -> ()
// CHECK-NOT: nova.free(
// CHECK-LABEL: func.func @allocate_space_assignment
// CHECK: %[[LHS:.*]] = nova.load(%arg0) {start = array<i64: 0, 0>} : memref<16x32xf32> -> tensor<16x32xf32, {bank = 0 : i64, space = 3 : i64}>
// CHECK: %[[RHS:.*]] = nova.load(%arg1) {start = array<i64: 0, 0>} : memref<32x8xf32> -> tensor<32x8xf32, {bank = 2 : i64, space = 3 : i64}>
// CHECK: %[[SCALE:.*]] = nova.load(%arg2) {start = array<i64: 0, 0>} : memref<16x8xf32> -> tensor<16x8xf32, {space = 4 : i64}>
// CHECK: %[[BIAS:.*]] = nova.load(%arg3) {start = array<i64: 0, 0>} : memref<16x8xf32> -> tensor<16x8xf32, {space = 5 : i64}>
// CHECK: %[[RESULT:.*]] = nova.matmul(%[[LHS]], %[[RHS]], %[[SCALE]], %[[BIAS]]) : tensor<16x32xf32, {bank = 0 : i64, space = 3 : i64}>, tensor<32x8xf32, {bank = 2 : i64, space = 3 : i64}>, tensor<16x8xf32, {space = 4 : i64}>, tensor<16x8xf32, {space = 5 : i64}> -> tensor<16x8xf32, {bank = 4 : i64, space = 3 : i64}>
// CHECK-NOT: nova.free(
