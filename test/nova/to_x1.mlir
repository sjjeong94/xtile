// RUN: xt-opt --nova-to-x1 %s | FileCheck %s

func.func @barrier_only() {
  nova.barrier() {mode = 1 : i32}
  func.return
}

// CHECK-LABEL: func.func @barrier_only
// CHECK: x1.barrier() {mode = 1 : i32}
// CHECK-NEXT: return

func.func @rowwise_softmax(%arg0: memref<128x64xf32>,
                           %arg1: memref<128x64xf32>) attributes {xt.double_buffering = 1 : i32, xt.grid = array<i32: 2, 1, 1>} {
  %0 = nova.load(%arg0) {start = array<i64: 0, 0>} : memref<128x64xf32> -> tensor<64x64xf32, {bank = 0 : i64, space = 3 : i64, threading = 32 : i64}>
  %1 = nova.reduce(%0) {mode = 1 : i32} : tensor<64x64xf32, {bank = 0 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x1xf32, {bank = 2 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier() {mode = 0 : i32}
  %2 = nova.broadcast(%0, %1) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 3 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<64x64xf32, {bank = 0 : i64, space = 3 : i64, threading = 32 : i64}>, tensor<64x1xf32, {bank = 2 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x64xf32, {bank = 4 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier() {mode = 0 : i32}
  %3 = nova.exp(%2) : tensor<64x64xf32, {bank = 4 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x64xf32, {bank = 2 : i64, space = 3 : i64}>
  nova.barrier() {mode = 0 : i32}
  %4 = nova.reduce(%3) {mode = 0 : i32} : tensor<64x64xf32, {bank = 2 : i64, space = 3 : i64}> -> tensor<64x1xf32, {bank = 4 : i64, space = 3 : i64}>
  nova.barrier() {mode = 0 : i32}
  %5 = nova.reciprocal(%4) : tensor<64x1xf32, {bank = 4 : i64, space = 3 : i64}> -> tensor<64x1xf32, {bank = 6 : i64, space = 3 : i64}>
  nova.barrier() {mode = 0 : i32}
  %6 = nova.broadcast(%3, %5) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 2 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<64x64xf32, {bank = 2 : i64, space = 3 : i64}>, tensor<64x1xf32, {bank = 6 : i64, space = 3 : i64}> -> tensor<64x64xf32, {bank = 4 : i64, space = 3 : i64}>
  nova.barrier() {mode = 0 : i32}
  nova.store(%6, %arg1) {start = array<i64: 0, 0>} : (tensor<64x64xf32, {bank = 4 : i64, space = 3 : i64}>, memref<128x64xf32>) -> ()
  %7 = nova.load(%arg0) {start = array<i64: 64, 0>} : memref<128x64xf32> -> tensor<64x64xf32, {bank = 2 : i64, space = 3 : i64, threading = 32 : i64}>
  %8 = nova.reduce(%7) {mode = 1 : i32} : tensor<64x64xf32, {bank = 2 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x1xf32, {bank = 6 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier() {mode = 0 : i32}
  %9 = nova.broadcast(%7, %8) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 3 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<64x64xf32, {bank = 2 : i64, space = 3 : i64, threading = 32 : i64}>, tensor<64x1xf32, {bank = 6 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x64xf32, {bank = 8 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier() {mode = 0 : i32}
  %10 = nova.exp(%9) : tensor<64x64xf32, {bank = 8 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x64xf32, {bank = 2 : i64, space = 3 : i64}>
  nova.barrier() {mode = 0 : i32}
  %11 = nova.reduce(%10) {mode = 0 : i32} : tensor<64x64xf32, {bank = 2 : i64, space = 3 : i64}> -> tensor<64x1xf32, {bank = 6 : i64, space = 3 : i64}>
  nova.barrier() {mode = 0 : i32}
  %12 = nova.reciprocal(%11) : tensor<64x1xf32, {bank = 6 : i64, space = 3 : i64}> -> tensor<64x1xf32, {bank = 8 : i64, space = 3 : i64}>
  nova.barrier() {mode = 0 : i32}
  %13 = nova.broadcast(%10, %12) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 2 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<64x64xf32, {bank = 2 : i64, space = 3 : i64}>, tensor<64x1xf32, {bank = 8 : i64, space = 3 : i64}> -> tensor<64x64xf32, {bank = 6 : i64, space = 3 : i64}>
  nova.barrier() {mode = 0 : i32}
  nova.store(%13, %arg1) {start = array<i64: 64, 0>} : (tensor<64x64xf32, {bank = 6 : i64, space = 3 : i64}>, memref<128x64xf32>) -> ()
  nova.barrier() {mode = 1 : i32}
  return
}

// CHECK-LABEL: func.func @rowwise_softmax
// CHECK-NEXT: x1.load(%arg0) {bank = 0 : i64, shape = [32, 64], space = 3 : i64, start = [0, 0], thread = 0 : i64} : memref<128x64xf32>
// CHECK-NEXT: x1.load(%arg0) {bank = 1 : i64, shape = [32, 64], space = 3 : i64, start = [32, 0], thread = 1 : i64} : memref<128x64xf32>
// CHECK-NEXT: x1.reduce() {inp0 = 0 : i64, inp1 = 1 : i64, m = 32 : i64, mode = 1 : i32, n = 64 : i64, res0 = 2 : i64, res1 = 3 : i64}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.broadcast() {lhs0 = 0 : i64, lhs1 = 1 : i64, lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 3 : i32, res0 = 4 : i64, res1 = 5 : i64, rhs0 = 2 : i64, rhs1 = 3 : i64, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.exp() {inp0 = 4 : i64, inp1 = 5 : i64, m = 32 : i64, n = 64 : i64, res0 = 2 : i64, res1 = 3 : i64}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.reduce() {inp0 = 2 : i64, inp1 = 3 : i64, m = 32 : i64, mode = 0 : i32, n = 64 : i64, res0 = 4 : i64, res1 = 5 : i64}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.reciprocal() {inp0 = 4 : i64, inp1 = 5 : i64, m = 64 : i64, n = 1 : i64, res0 = 6 : i64, res1 = 7 : i64}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.broadcast() {lhs0 = 2 : i64, lhs1 = 3 : i64, lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 2 : i32, res0 = 4 : i64, res1 = 5 : i64, rhs0 = 6 : i64, rhs1 = 7 : i64, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.store(%arg1) {bank = 4 : i64, shape = [32, 64], space = 3 : i64, start = [0, 0], thread = 0 : i64} : memref<128x64xf32>
// CHECK-NEXT: x1.store(%arg1) {bank = 5 : i64, shape = [32, 64], space = 3 : i64, start = [32, 0], thread = 1 : i64} : memref<128x64xf32>
// CHECK-NEXT: x1.load(%arg0) {bank = 2 : i64, shape = [32, 64], space = 3 : i64, start = [64, 0], thread = 0 : i64} : memref<128x64xf32>
// CHECK-NEXT: x1.load(%arg0) {bank = 3 : i64, shape = [32, 64], space = 3 : i64, start = [96, 0], thread = 1 : i64} : memref<128x64xf32>
// CHECK-NEXT: x1.reduce() {inp0 = 2 : i64, inp1 = 3 : i64, m = 32 : i64, mode = 1 : i32, n = 64 : i64, res0 = 6 : i64, res1 = 7 : i64}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.broadcast() {lhs0 = 2 : i64, lhs1 = 3 : i64, lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 3 : i32, res0 = 8 : i64, res1 = 9 : i64, rhs0 = 6 : i64, rhs1 = 7 : i64, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.exp() {inp0 = 8 : i64, inp1 = 9 : i64, m = 32 : i64, n = 64 : i64, res0 = 2 : i64, res1 = 3 : i64}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.reduce() {inp0 = 2 : i64, inp1 = 3 : i64, m = 32 : i64, mode = 0 : i32, n = 64 : i64, res0 = 6 : i64, res1 = 7 : i64}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.reciprocal() {inp0 = 6 : i64, inp1 = 7 : i64, m = 64 : i64, n = 1 : i64, res0 = 8 : i64, res1 = 9 : i64}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.broadcast() {lhs0 = 2 : i64, lhs1 = 3 : i64, lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 2 : i32, res0 = 6 : i64, res1 = 7 : i64, rhs0 = 8 : i64, rhs1 = 9 : i64, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.store(%arg1) {bank = 6 : i64, shape = [32, 64], space = 3 : i64, start = [64, 0], thread = 0 : i64} : memref<128x64xf32>
// CHECK-NEXT: x1.store(%arg1) {bank = 7 : i64, shape = [32, 64], space = 3 : i64, start = [96, 0], thread = 1 : i64} : memref<128x64xf32>
// CHECK-NEXT: x1.barrier() {mode = 1 : i32}
// CHECK-NEXT: return
