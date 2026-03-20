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
  %0 = nova.load(%arg0) {start = array<i64: 0, 0>} : memref<128x64xf32> -> tensor<64x64xf32, {bank0 = 0 : i64, bank1 = 1 : i64, space = 3 : i64, threading = 32 : i64}>
  %1 = nova.reduce(%0) {mode = 1 : i32} : tensor<64x64xf32, {bank0 = 0 : i64, bank1 = 1 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x1xf32, {bank0 = 2 : i64, bank1 = 3 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier() {mode = 0 : i32}
  %2 = nova.broadcast(%0, %1) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 3 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<64x64xf32, {bank0 = 0 : i64, bank1 = 1 : i64, space = 3 : i64, threading = 32 : i64}>, tensor<64x1xf32, {bank0 = 2 : i64, bank1 = 3 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x64xf32, {bank0 = 4 : i64, bank1 = 5 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier() {mode = 0 : i32}
  %3 = nova.exp(%2) : tensor<64x64xf32, {bank0 = 4 : i64, bank1 = 5 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x64xf32, {bank0 = 2 : i64, space = 3 : i64}>
  nova.barrier() {mode = 0 : i32}
  %4 = nova.reduce(%3) {mode = 0 : i32} : tensor<64x64xf32, {bank0 = 2 : i64, space = 3 : i64}> -> tensor<64x1xf32, {bank0 = 4 : i64, space = 3 : i64}>
  nova.barrier() {mode = 0 : i32}
  %5 = nova.reciprocal(%4) : tensor<64x1xf32, {bank0 = 4 : i64, space = 3 : i64}> -> tensor<64x1xf32, {bank0 = 6 : i64, space = 3 : i64}>
  nova.barrier() {mode = 0 : i32}
  %6 = nova.broadcast(%3, %5) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 2 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<64x64xf32, {bank0 = 2 : i64, space = 3 : i64}>, tensor<64x1xf32, {bank0 = 6 : i64, space = 3 : i64}> -> tensor<64x64xf32, {bank0 = 4 : i64, space = 3 : i64}>
  nova.barrier() {mode = 0 : i32}
  nova.store(%6, %arg1) {start = array<i64: 0, 0>} : (tensor<64x64xf32, {bank0 = 4 : i64, space = 3 : i64}>, memref<128x64xf32>) -> ()
  %7 = nova.load(%arg0) {start = array<i64: 64, 0>} : memref<128x64xf32> -> tensor<64x64xf32, {bank0 = 2 : i64, bank1 = 3 : i64, space = 3 : i64, threading = 32 : i64}>
  %8 = nova.reduce(%7) {mode = 1 : i32} : tensor<64x64xf32, {bank0 = 2 : i64, bank1 = 3 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x1xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier() {mode = 0 : i32}
  %9 = nova.broadcast(%7, %8) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 3 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<64x64xf32, {bank0 = 2 : i64, bank1 = 3 : i64, space = 3 : i64, threading = 32 : i64}>, tensor<64x1xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier() {mode = 0 : i32}
  %10 = nova.exp(%9) : tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x64xf32, {bank0 = 2 : i64, space = 3 : i64}>
  nova.barrier() {mode = 0 : i32}
  %11 = nova.reduce(%10) {mode = 0 : i32} : tensor<64x64xf32, {bank0 = 2 : i64, space = 3 : i64}> -> tensor<64x1xf32, {bank0 = 6 : i64, space = 3 : i64}>
  nova.barrier() {mode = 0 : i32}
  %12 = nova.reciprocal(%11) : tensor<64x1xf32, {bank0 = 6 : i64, space = 3 : i64}> -> tensor<64x1xf32, {bank0 = 8 : i64, space = 3 : i64}>
  nova.barrier() {mode = 0 : i32}
  %13 = nova.broadcast(%10, %12) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 2 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<64x64xf32, {bank0 = 2 : i64, space = 3 : i64}>, tensor<64x1xf32, {bank0 = 8 : i64, space = 3 : i64}> -> tensor<64x64xf32, {bank0 = 6 : i64, space = 3 : i64}>
  nova.barrier() {mode = 0 : i32}
  nova.store(%13, %arg1) {start = array<i64: 64, 0>} : (tensor<64x64xf32, {bank0 = 6 : i64, space = 3 : i64}>, memref<128x64xf32>) -> ()
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
// CHECK-NEXT: x1.exp() {inp0 = 4 : i64, inp1 = 5 : i64, m = 32 : i64, n = 64 : i64, res0 = 2 : i64, res1 = 2 : i64}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.reduce() {inp0 = 2 : i64, inp1 = 2 : i64, m = 32 : i64, mode = 0 : i32, n = 64 : i64, res0 = 4 : i64, res1 = 4 : i64}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.reciprocal() {inp0 = 4 : i64, inp1 = 4 : i64, m = 64 : i64, n = 1 : i64, res0 = 6 : i64, res1 = 6 : i64}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.broadcast() {lhs0 = 2 : i64, lhs1 = 2 : i64, lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 2 : i32, res0 = 4 : i64, res1 = 4 : i64, rhs0 = 6 : i64, rhs1 = 6 : i64, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.store(%arg1) {bank = 4 : i64, shape = [32, 64], space = 3 : i64, start = [0, 0], thread = 0 : i64} : memref<128x64xf32>
// CHECK-NEXT: x1.store(%arg1) {bank = 4 : i64, shape = [32, 64], space = 3 : i64, start = [32, 0], thread = 1 : i64} : memref<128x64xf32>
// CHECK-NEXT: x1.load(%arg0) {bank = 2 : i64, shape = [32, 64], space = 3 : i64, start = [64, 0], thread = 0 : i64} : memref<128x64xf32>
// CHECK-NEXT: x1.load(%arg0) {bank = 3 : i64, shape = [32, 64], space = 3 : i64, start = [96, 0], thread = 1 : i64} : memref<128x64xf32>
// CHECK-NEXT: x1.reduce() {inp0 = 2 : i64, inp1 = 3 : i64, m = 32 : i64, mode = 1 : i32, n = 64 : i64, res0 = 6 : i64, res1 = 7 : i64}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.broadcast() {lhs0 = 2 : i64, lhs1 = 3 : i64, lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 3 : i32, res0 = 8 : i64, res1 = 9 : i64, rhs0 = 6 : i64, rhs1 = 7 : i64, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.exp() {inp0 = 8 : i64, inp1 = 9 : i64, m = 32 : i64, n = 64 : i64, res0 = 2 : i64, res1 = 2 : i64}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.reduce() {inp0 = 2 : i64, inp1 = 2 : i64, m = 32 : i64, mode = 0 : i32, n = 64 : i64, res0 = 6 : i64, res1 = 6 : i64}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.reciprocal() {inp0 = 6 : i64, inp1 = 6 : i64, m = 64 : i64, n = 1 : i64, res0 = 8 : i64, res1 = 8 : i64}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.broadcast() {lhs0 = 2 : i64, lhs1 = 2 : i64, lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 2 : i32, res0 = 6 : i64, res1 = 6 : i64, rhs0 = 8 : i64, rhs1 = 8 : i64, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.store(%arg1) {bank = 6 : i64, shape = [32, 64], space = 3 : i64, start = [64, 0], thread = 0 : i64} : memref<128x64xf32>
// CHECK-NEXT: x1.store(%arg1) {bank = 6 : i64, shape = [32, 64], space = 3 : i64, start = [96, 0], thread = 1 : i64} : memref<128x64xf32>
// CHECK-NEXT: x1.barrier() {mode = 1 : i32}
// CHECK-NEXT: return

func.func @rowwise_layernorm(%arg0: memref<128x64xf32>, %arg1: memref<1x64xf32>,
                             %arg2: memref<1x64xf32>, %arg3: memref<128x64xf32>) attributes {xt.double_buffering = 1 : i32, xt.grid = array<i32: 2, 1, 1>} {
  %0 = nova.load(%arg0) {start = array<i64: 0, 0>} : memref<128x64xf32> -> tensor<64x64xf32, {bank0 = 0 : i64, bank1 = 1 : i64, space = 3 : i64, threading = 32 : i64}>
  %1 = nova.load(%arg1) {shared = 1 : i64, start = array<i64: 0, 0>} : memref<1x64xf32> -> tensor<1x64xf32, {bank0 = 2 : i64, space = 3 : i64, threading = 1 : i64}>
  %2 = nova.load(%arg2) {shared = 1 : i64, start = array<i64: 0, 0>} : memref<1x64xf32> -> tensor<1x64xf32, {bank0 = 4 : i64, space = 3 : i64, threading = 1 : i64}>
  %3 = nova.reduce(%0) {mode = 0 : i32} : tensor<64x64xf32, {bank0 = 0 : i64, bank1 = 1 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x1xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier() {mode = 0 : i32}
  %4 = nova.broadcast(%0, %3) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 3 : i32, rhs_a = 1.562500e-02 : f32, rhs_b = 0.000000e+00 : f32} : tensor<64x64xf32, {bank0 = 0 : i64, bank1 = 1 : i64, space = 3 : i64, threading = 32 : i64}>, tensor<64x1xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier() {mode = 0 : i32}
  %5 = nova.square(%4) : tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x64xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier() {mode = 0 : i32}
  %6 = nova.reduce(%5) {mode = 0 : i32} : tensor<64x64xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x1xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier() {mode = 0 : i32}
  %7 = nova.scalar_fma(%6) {a = 1.562500e-02 : f32, b = 9.99999974E-6 : f32} : tensor<64x1xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x1xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier() {mode = 0 : i32}
  %8 = nova.rsqrt(%7) : tensor<64x1xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x1xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier() {mode = 0 : i32}
  %9 = nova.broadcast(%4, %8) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 2 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}>, tensor<64x1xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x64xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier() {mode = 0 : i32}
  %10 = nova.broadcast(%9, %1) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 2 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<64x64xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}>, tensor<1x64xf32, {bank0 = 2 : i64, space = 3 : i64, threading = 1 : i64}> -> tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier() {mode = 0 : i32}
  %11 = nova.broadcast(%10, %2) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 1 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}>, tensor<1x64xf32, {bank0 = 4 : i64, space = 3 : i64, threading = 1 : i64}> -> tensor<64x64xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier() {mode = 0 : i32}
  nova.store(%11, %arg3) {start = array<i64: 0, 0>} : (tensor<64x64xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}>, memref<128x64xf32>) -> ()
  %12 = nova.load(%arg0) {start = array<i64: 64, 0>} : memref<128x64xf32> -> tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}>
  %13 = nova.reduce(%12) {mode = 0 : i32} : tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x1xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier() {mode = 0 : i32}
  %14 = nova.broadcast(%12, %13) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 3 : i32, rhs_a = 1.562500e-02 : f32, rhs_b = 0.000000e+00 : f32} : tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}>, tensor<64x1xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x64xf32, {bank0 = 12 : i64, bank1 = 13 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier() {mode = 0 : i32}
  %15 = nova.square(%14) : tensor<64x64xf32, {bank0 = 12 : i64, bank1 = 13 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier() {mode = 0 : i32}
  %16 = nova.reduce(%15) {mode = 0 : i32} : tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x1xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier() {mode = 0 : i32}
  %17 = nova.scalar_fma(%16) {a = 1.562500e-02 : f32, b = 9.99999974E-6 : f32} : tensor<64x1xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x1xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier() {mode = 0 : i32}
  %18 = nova.rsqrt(%17) : tensor<64x1xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x1xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier() {mode = 0 : i32}
  %19 = nova.broadcast(%14, %18) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 2 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<64x64xf32, {bank0 = 12 : i64, bank1 = 13 : i64, space = 3 : i64, threading = 32 : i64}>, tensor<64x1xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier() {mode = 0 : i32}
  %20 = nova.broadcast(%19, %1) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 2 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}>, tensor<1x64xf32, {bank0 = 2 : i64, space = 3 : i64, threading = 1 : i64}> -> tensor<64x64xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier() {mode = 0 : i32}
  %21 = nova.broadcast(%20, %2) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 1 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<64x64xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}>, tensor<1x64xf32, {bank0 = 4 : i64, space = 3 : i64, threading = 1 : i64}> -> tensor<64x64xf32, {bank0 = 2 : i64, bank1 = 3 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier() {mode = 0 : i32}
  nova.store(%21, %arg3) {start = array<i64: 64, 0>} : (tensor<64x64xf32, {bank0 = 2 : i64, bank1 = 3 : i64, space = 3 : i64, threading = 32 : i64}>, memref<128x64xf32>) -> ()
  nova.barrier() {mode = 1 : i32}
  return
}

// CHECK-LABEL: func.func @rowwise_layernorm
// CHECK-NEXT: x1.load(%arg0) {bank = 0 : i64, shape = [32, 64], space = 3 : i64, start = [0, 0], thread = 0 : i64} : memref<128x64xf32>
// CHECK-NEXT: x1.load(%arg0) {bank = 1 : i64, shape = [32, 64], space = 3 : i64, start = [32, 0], thread = 1 : i64} : memref<128x64xf32>
// CHECK-NEXT: x1.load(%arg1) {bank = 2 : i64, shape = [1, 64], space = 3 : i64, start = [0, 0], thread = 0 : i64} : memref<1x64xf32>
// CHECK-NEXT: x1.load(%arg2) {bank = 4 : i64, shape = [1, 64], space = 3 : i64, start = [0, 0], thread = 0 : i64} : memref<1x64xf32>
// CHECK-NEXT: x1.reduce() {inp0 = 0 : i64, inp1 = 1 : i64, m = 32 : i64, mode = 0 : i32, n = 64 : i64, res0 = 6 : i64, res1 = 7 : i64}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.broadcast() {lhs0 = 0 : i64, lhs1 = 1 : i64, lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 3 : i32, res0 = 8 : i64, res1 = 9 : i64, rhs0 = 6 : i64, rhs1 = 7 : i64, rhs_a = 1.562500e-02 : f32, rhs_b = 0.000000e+00 : f32}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.square() {inp0 = 8 : i64, inp1 = 9 : i64, m = 32 : i64, n = 64 : i64, res0 = 6 : i64, res1 = 7 : i64}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.reduce() {inp0 = 6 : i64, inp1 = 7 : i64, m = 32 : i64, mode = 0 : i32, n = 64 : i64, res0 = 10 : i64, res1 = 11 : i64}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.scalar_fma() {a = 1.562500e-02 : f32, b = 9.99999974E-6 : f32, inp0 = 10 : i64, inp1 = 11 : i64, m = 32 : i64, n = 1 : i64, res0 = 6 : i64, res1 = 7 : i64}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.rsqrt() {inp0 = 6 : i64, inp1 = 7 : i64, m = 32 : i64, n = 1 : i64, res0 = 10 : i64, res1 = 11 : i64}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.broadcast() {lhs0 = 8 : i64, lhs1 = 9 : i64, lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 2 : i32, res0 = 6 : i64, res1 = 7 : i64, rhs0 = 10 : i64, rhs1 = 11 : i64, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.broadcast() {lhs0 = 6 : i64, lhs1 = 7 : i64, lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 2 : i32, res0 = 8 : i64, res1 = 9 : i64, rhs0 = 2 : i64, rhs1 = 2 : i64, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.broadcast() {lhs0 = 8 : i64, lhs1 = 9 : i64, lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 1 : i32, res0 = 6 : i64, res1 = 7 : i64, rhs0 = 4 : i64, rhs1 = 4 : i64, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.store(%arg3) {bank = 6 : i64, shape = [32, 64], space = 3 : i64, start = [0, 0], thread = 0 : i64} : memref<128x64xf32>
// CHECK-NEXT: x1.store(%arg3) {bank = 7 : i64, shape = [32, 64], space = 3 : i64, start = [32, 0], thread = 1 : i64} : memref<128x64xf32>
// CHECK-NEXT: x1.load(%arg0) {bank = 8 : i64, shape = [32, 64], space = 3 : i64, start = [64, 0], thread = 0 : i64} : memref<128x64xf32>
// CHECK-NEXT: x1.load(%arg0) {bank = 9 : i64, shape = [32, 64], space = 3 : i64, start = [96, 0], thread = 1 : i64} : memref<128x64xf32>
// CHECK-NEXT: x1.reduce() {inp0 = 8 : i64, inp1 = 9 : i64, m = 32 : i64, mode = 0 : i32, n = 64 : i64, res0 = 10 : i64, res1 = 11 : i64}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.broadcast() {lhs0 = 8 : i64, lhs1 = 9 : i64, lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 3 : i32, res0 = 12 : i64, res1 = 13 : i64, rhs0 = 10 : i64, rhs1 = 11 : i64, rhs_a = 1.562500e-02 : f32, rhs_b = 0.000000e+00 : f32}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.square() {inp0 = 12 : i64, inp1 = 13 : i64, m = 32 : i64, n = 64 : i64, res0 = 8 : i64, res1 = 9 : i64}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.reduce() {inp0 = 8 : i64, inp1 = 9 : i64, m = 32 : i64, mode = 0 : i32, n = 64 : i64, res0 = 10 : i64, res1 = 11 : i64}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.scalar_fma() {a = 1.562500e-02 : f32, b = 9.99999974E-6 : f32, inp0 = 10 : i64, inp1 = 11 : i64, m = 32 : i64, n = 1 : i64, res0 = 8 : i64, res1 = 9 : i64}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.rsqrt() {inp0 = 8 : i64, inp1 = 9 : i64, m = 32 : i64, n = 1 : i64, res0 = 10 : i64, res1 = 11 : i64}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.broadcast() {lhs0 = 12 : i64, lhs1 = 13 : i64, lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 2 : i32, res0 = 8 : i64, res1 = 9 : i64, rhs0 = 10 : i64, rhs1 = 11 : i64, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.broadcast() {lhs0 = 8 : i64, lhs1 = 9 : i64, lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 2 : i32, res0 = 10 : i64, res1 = 11 : i64, rhs0 = 2 : i64, rhs1 = 2 : i64, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.broadcast() {lhs0 = 10 : i64, lhs1 = 11 : i64, lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 1 : i32, res0 = 2 : i64, res1 = 3 : i64, rhs0 = 4 : i64, rhs1 = 4 : i64, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32}
// CHECK-NEXT: x1.barrier() {mode = 0 : i32}
// CHECK-NEXT: x1.store(%arg3) {bank = 2 : i64, shape = [32, 64], space = 3 : i64, start = [64, 0], thread = 0 : i64} : memref<128x64xf32>
// CHECK-NEXT: x1.store(%arg3) {bank = 3 : i64, shape = [32, 64], space = 3 : i64, start = [96, 0], thread = 1 : i64} : memref<128x64xf32>
// CHECK-NEXT: x1.barrier() {mode = 1 : i32}
// CHECK-NEXT: return
