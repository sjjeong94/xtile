// RUN: xt-opt --nova-threading %s | FileCheck %s

func.func @load_sets_threading(%src: memref<10x8xf32>) {
  %0 = nova.load(%src) {start = array<i64: 0, 0>} : memref<10x8xf32> -> tensor<5x8xf32>
  nova.free(%0) : tensor<5x8xf32>
  func.return
}

func.func @propagate_unary_and_casts(%src: memref<10x8xf32>) {
  %0 = nova.load(%src) {start = array<i64: 0, 0>} : memref<10x8xf32> -> tensor<5x8xf32>
  %1 = nova.square(%0) : tensor<5x8xf32> -> tensor<5x8xf32>
  %2 = nova.rsqrt(%1) : tensor<5x8xf32> -> tensor<5x8xf32>
  %3 = nova.ftoi(%2) : tensor<5x8xf32> -> tensor<5x8xi8>
  nova.free(%3) : tensor<5x8xi8>
  func.return
}

func.func @propagate_scalar_ops(%src: memref<10x8xf32>) {
  %0 = nova.load(%src) {start = array<i64: 0, 0>} : memref<10x8xf32> -> tensor<5x8xf32>
  %1 = nova.scalar(%0) {mode = 2 : i32, rhs = 1.250000e-01 : f32} : tensor<5x8xf32> -> tensor<5x8xf32>
  %2 = nova.scalar_fma(%1) {a = 2.000000e+00 : f32, b = 3.000000e+00 : f32} : tensor<5x8xf32> -> tensor<5x8xf32>
  nova.free(%2) : tensor<5x8xf32>
  func.return
}

func.func @propagate_reduce_and_binary(%src: memref<10x8xf32>) {
  %0 = nova.load(%src) {start = array<i64: 0, 0>} : memref<10x8xf32> -> tensor<5x8xf32>
  %1 = nova.reduce(%0) {mode = 0 : i32} : tensor<5x8xf32> -> tensor<5x1xf32>
  %2 = nova.broadcast(%0, %1) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 0 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<5x8xf32>, tensor<5x1xf32> -> tensor<5x8xf32>
  %3 = nova.elementwise(%2, %0) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 0 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<5x8xf32>, tensor<5x8xf32> -> tensor<5x8xf32>
  nova.free(%3) : tensor<5x8xf32>
  func.return
}

func.func @broadcast_propagates_max_threading(%src: memref<10x8xf32>) {
  %0 = nova.load(%src) {start = array<i64: 0, 0>} : memref<10x8xf32> -> tensor<5x8xf32>
  %1 = nova.load(%src) {start = array<i64: 0, 0>} : memref<10x8xf32> -> tensor<3x8xf32>
  %2 = nova.broadcast(%0, %1) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 0 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<5x8xf32>, tensor<3x8xf32> -> tensor<5x8xf32>
  nova.free(%2) : tensor<5x8xf32>
  func.return
}

func.func @matmul_propagates_lhs_and_strips_rhs(%lhs_src: memref<10x4xf32>, %rhs_src: memref<6x4xf32>, %scale_src: memref<3x4xf32>, %bias_src: memref<3x4xf32>) {
  %lhs = nova.load(%lhs_src) {start = array<i64: 0, 0>} : memref<10x4xf32> -> tensor<5x4xf32>
  %rhs = nova.load(%rhs_src) {start = array<i64: 0, 0>} : memref<6x4xf32> -> tensor<3x4xf32>
  %scale = nova.load(%scale_src) {start = array<i64: 0, 0>} : memref<3x4xf32> -> tensor<3x4xf32>
  %bias = nova.load(%bias_src) {start = array<i64: 0, 0>} : memref<3x4xf32> -> tensor<3x4xf32>
  %result = nova.matmul(%lhs, %rhs, %scale, %bias) : tensor<5x4xf32>, tensor<3x4xf32>, tensor<3x4xf32>, tensor<3x4xf32> -> tensor<5x4xf32>
  nova.free(%result) : tensor<5x4xf32>
  func.return
}

func.func @overwrite_existing_threading(%src: memref<12x8xf32>) {
  %0 = nova.load(%src) {start = array<i64: 0, 0>} : memref<12x8xf32> -> tensor<6x8xf32, {threading = 99 : i64}>
  nova.free(%0) : tensor<6x8xf32, {threading = 99 : i64}>
  func.return
}

// CHECK-LABEL: func.func @load_sets_threading
// CHECK: %[[LOAD:.*]] = nova.load(%arg0) {start = array<i64: 0, 0>} : memref<10x8xf32> -> tensor<5x8xf32, {threading = 3 : i64}>
// CHECK: nova.free(%[[LOAD]]) : tensor<5x8xf32, {threading = 3 : i64}>
// CHECK: return
// CHECK-LABEL: func.func @propagate_unary_and_casts
// CHECK: %[[LOAD2:.*]] = nova.load(%arg0) {start = array<i64: 0, 0>} : memref<10x8xf32> -> tensor<5x8xf32, {threading = 3 : i64}>
// CHECK: %[[SQUARE:.*]] = nova.square(%[[LOAD2]]) : tensor<5x8xf32, {threading = 3 : i64}> -> tensor<5x8xf32, {threading = 3 : i64}>
// CHECK: %[[RSQRT:.*]] = nova.rsqrt(%[[SQUARE]]) : tensor<5x8xf32, {threading = 3 : i64}> -> tensor<5x8xf32, {threading = 3 : i64}>
// CHECK: %[[FTOI:.*]] = nova.ftoi(%[[RSQRT]]) : tensor<5x8xf32, {threading = 3 : i64}> -> tensor<5x8xi8, {threading = 3 : i64}>
// CHECK: nova.free(%[[FTOI]]) : tensor<5x8xi8, {threading = 3 : i64}>
// CHECK: return
// CHECK-LABEL: func.func @propagate_scalar_ops
// CHECK: %[[LOADS:.*]] = nova.load(%arg0) {start = array<i64: 0, 0>} : memref<10x8xf32> -> tensor<5x8xf32, {threading = 3 : i64}>
// CHECK: %[[SCALAR:.*]] = nova.scalar(%[[LOADS]]) {mode = 2 : i32, rhs = 1.250000e-01 : f32} : tensor<5x8xf32, {threading = 3 : i64}> -> tensor<5x8xf32, {threading = 3 : i64}>
// CHECK: %[[FMA:.*]] = nova.scalar_fma(%[[SCALAR]]) {a = 2.000000e+00 : f32, b = 3.000000e+00 : f32} : tensor<5x8xf32, {threading = 3 : i64}> -> tensor<5x8xf32, {threading = 3 : i64}>
// CHECK: nova.free(%[[FMA]]) : tensor<5x8xf32, {threading = 3 : i64}>
// CHECK: return
// CHECK-LABEL: func.func @propagate_reduce_and_binary
// CHECK: %[[LOADR:.*]] = nova.load(%arg0) {start = array<i64: 0, 0>} : memref<10x8xf32> -> tensor<5x8xf32, {threading = 3 : i64}>
// CHECK: %[[REDUCE:.*]] = nova.reduce(%[[LOADR]]) {mode = 0 : i32} : tensor<5x8xf32, {threading = 3 : i64}> -> tensor<5x1xf32, {threading = 3 : i64}>
// CHECK: %[[BCAST:.*]] = nova.broadcast(%[[LOADR]], %[[REDUCE]]) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 0 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<5x8xf32, {threading = 3 : i64}>, tensor<5x1xf32, {threading = 3 : i64}> -> tensor<5x8xf32, {threading = 3 : i64}>
// CHECK: %[[EW:.*]] = nova.elementwise(%[[BCAST]], %[[LOADR]]) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 0 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<5x8xf32, {threading = 3 : i64}>, tensor<5x8xf32, {threading = 3 : i64}> -> tensor<5x8xf32, {threading = 3 : i64}>
// CHECK: nova.free(%[[EW]]) : tensor<5x8xf32, {threading = 3 : i64}>
// CHECK: return
// CHECK-LABEL: func.func @broadcast_propagates_max_threading
// CHECK: %[[LOADB0:.*]] = nova.load(%arg0) {start = array<i64: 0, 0>} : memref<10x8xf32> -> tensor<5x8xf32, {threading = 3 : i64}>
// CHECK: %[[LOADB1:.*]] = nova.load(%arg0) {start = array<i64: 0, 0>} : memref<10x8xf32> -> tensor<3x8xf32, {threading = 2 : i64}>
// CHECK: %[[BMM:.*]] = nova.broadcast(%[[LOADB0]], %[[LOADB1]]) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 0 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<5x8xf32, {threading = 3 : i64}>, tensor<3x8xf32, {threading = 2 : i64}> -> tensor<5x8xf32, {threading = 3 : i64}>
// CHECK: nova.free(%[[BMM]]) : tensor<5x8xf32, {threading = 3 : i64}>
// CHECK: return
// CHECK-LABEL: func.func @matmul_propagates_lhs_and_strips_rhs
// CHECK: %[[LHS:.*]] = nova.load(%arg0) {start = array<i64: 0, 0>} : memref<10x4xf32> -> tensor<5x4xf32, {threading = 3 : i64}>
// CHECK: %[[RHS:.*]] = nova.load(%arg1) {start = array<i64: 0, 0>} : memref<6x4xf32> -> tensor<3x4xf32>
// CHECK: %[[SCALE:.*]] = nova.load(%arg2) {start = array<i64: 0, 0>} : memref<3x4xf32> -> tensor<3x4xf32>
// CHECK: %[[BIAS:.*]] = nova.load(%arg3) {start = array<i64: 0, 0>} : memref<3x4xf32> -> tensor<3x4xf32>
// CHECK: %[[MM:.*]] = nova.matmul(%[[LHS]], %[[RHS]], %[[SCALE]], %[[BIAS]]) : tensor<5x4xf32, {threading = 3 : i64}>, tensor<3x4xf32>, tensor<3x4xf32>, tensor<3x4xf32> -> tensor<5x4xf32, {threading = 3 : i64}>
// CHECK: nova.free(%[[MM]]) : tensor<5x4xf32, {threading = 3 : i64}>
// CHECK: return
// CHECK-LABEL: func.func @overwrite_existing_threading
// CHECK: %[[LOAD3:.*]] = nova.load(%arg0) {start = array<i64: 0, 0>} : memref<12x8xf32> -> tensor<6x8xf32, {threading = 3 : i64}>
// CHECK: nova.free(%[[LOAD3]]) : tensor<6x8xf32, {threading = 3 : i64}>
// CHECK: return
