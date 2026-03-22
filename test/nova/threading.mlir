// RUN: xt-opt --nova-threading %s | FileCheck %s

func.func @load_sets_threading(%src: memref<10x8xf32>) {
  %0 = nova.load(%src) {start = [0, 0]} : memref<10x8xf32> -> tensor<5x8xf32>
  func.return
}

func.func @propagate_unary_and_casts(%src: memref<10x8xf32>) {
  %0 = nova.load(%src) {start = [0, 0]} : memref<10x8xf32> -> tensor<5x8xf32>
  %1 = nova.square(%0) : tensor<5x8xf32> -> tensor<5x8xf32>
  %2 = nova.exp(%1) : tensor<5x8xf32> -> tensor<5x8xf32>
  %3 = nova.rsqrt(%2) : tensor<5x8xf32> -> tensor<5x8xf32>
  %4 = nova.ftoi(%3) : tensor<5x8xf32> -> tensor<5x8xi8>
  func.return
}

func.func @propagate_scalar_ops(%src: memref<10x8xf32>) {
  %0 = nova.load(%src) {start = [0, 0]} : memref<10x8xf32> -> tensor<5x8xf32>
  %1 = nova.scalar(%0) {mode = 2 : i32, rhs = 1.250000e-01 : f32} : tensor<5x8xf32> -> tensor<5x8xf32>
  %2 = nova.scalar_fma(%1) {a = 2.000000e+00 : f32, b = 3.000000e+00 : f32} : tensor<5x8xf32> -> tensor<5x8xf32>
  func.return
}

func.func @propagate_reduce_and_binary(%src: memref<10x8xf32>) {
  %0 = nova.load(%src) {start = [0, 0]} : memref<10x8xf32> -> tensor<5x8xf32>
  %1 = nova.reduce(%0) {axis = 1 : i64, mode = 0 : i32} : tensor<5x8xf32> -> tensor<5x1xf32>
  %2 = nova.broadcast(%0, %1) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 0 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<5x8xf32>, tensor<5x1xf32> -> tensor<5x8xf32>
  %3 = nova.elementwise(%2, %0) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 0 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<5x8xf32>, tensor<5x8xf32> -> tensor<5x8xf32>
  func.return
}

func.func @broadcast_propagates_max_threading(%src: memref<10x8xf32>) {
  %0 = nova.load(%src) {start = [0, 0]} : memref<10x8xf32> -> tensor<5x8xf32>
  %1 = nova.load(%src) {start = [0, 0]} : memref<10x8xf32> -> tensor<3x8xf32>
  %2 = nova.broadcast(%0, %1) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 0 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<5x8xf32>, tensor<3x8xf32> -> tensor<5x8xf32>
  func.return
}

func.func @reduce_axis0_drops_reduce_threading_but_broadcast_recovers(%src: memref<10x8xf32>) {
  %0 = nova.load(%src) {start = [0, 0]} : memref<10x8xf32> -> tensor<5x8xf32>
  %1 = nova.reduce(%0) {axis = 0 : i64, mode = 0 : i32} : tensor<5x8xf32> -> tensor<1x8xf32>
  %2 = nova.broadcast(%0, %1) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 0 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<5x8xf32>, tensor<1x8xf32> -> tensor<5x8xf32>
  func.return
}

func.func @matmul_propagates_lhs_and_strips_rhs(%lhs_src: memref<10x4xf32>, %rhs_src: memref<6x4xf32>, %scale_src: memref<3x4xf32>, %bias_src: memref<3x4xf32>) {
  %lhs = nova.load(%lhs_src) {start = [0, 0]} : memref<10x4xf32> -> tensor<5x4xf32>
  %rhs = nova.load(%rhs_src) {start = [0, 0]} : memref<6x4xf32> -> tensor<3x4xf32>
  %scale = nova.load(%scale_src) {start = [0, 0]} : memref<3x4xf32> -> tensor<3x4xf32>
  %bias = nova.load(%bias_src) {start = [0, 0]} : memref<3x4xf32> -> tensor<3x4xf32>
  %result = nova.matmul(%lhs, %rhs, %scale, %bias) : tensor<5x4xf32>, tensor<3x4xf32>, tensor<3x4xf32>, tensor<3x4xf32> -> tensor<5x4xf32>
  func.return
}

func.func @overwrite_existing_threading(%src: memref<12x8xf32>) {
  %0 = nova.load(%src) {start = [0, 0]} : memref<12x8xf32> -> tensor<6x8xf32, {threading = 99 : i64}>
  func.return
}

func.func @load_sets_thread_slices(%src: memref<128x128xf32>) {
  %0 = nova.load(%src) {start = [0, 0]} : memref<128x128xf32> -> tensor<64x128xf32>
  func.return
}

// CHECK-LABEL: func.func @load_sets_threading
// CHECK: %[[LOAD:.*]] = nova.load(%arg0) {start = [0, 0]} : memref<10x8xf32> -> tensor<5x8xf32, {shape0 = [3, 8], shape1 = [2, 8], start0 = [0, 0], start1 = [3, 0]}>
// CHECK: return
// CHECK-LABEL: func.func @propagate_unary_and_casts
// CHECK: %[[LOAD2:.*]] = nova.load(%arg0) {start = [0, 0]} : memref<10x8xf32> -> tensor<5x8xf32, {shape0 = [3, 8], shape1 = [2, 8], start0 = [0, 0], start1 = [3, 0]}>
// CHECK: %[[SQUARE:.*]] = nova.square(%[[LOAD2]]) : tensor<5x8xf32, {shape0 = [3, 8], shape1 = [2, 8], start0 = [0, 0], start1 = [3, 0]}> -> tensor<5x8xf32, {shape0 = [3, 8], shape1 = [2, 8], start0 = [0, 0], start1 = [3, 0]}>
// CHECK: %[[EXP:.*]] = nova.exp(%[[SQUARE]]) : tensor<5x8xf32, {shape0 = [3, 8], shape1 = [2, 8], start0 = [0, 0], start1 = [3, 0]}> -> tensor<5x8xf32, {shape0 = [3, 8], shape1 = [2, 8], start0 = [0, 0], start1 = [3, 0]}>
// CHECK: %[[RSQRT:.*]] = nova.rsqrt(%[[EXP]]) : tensor<5x8xf32, {shape0 = [3, 8], shape1 = [2, 8], start0 = [0, 0], start1 = [3, 0]}> -> tensor<5x8xf32, {shape0 = [3, 8], shape1 = [2, 8], start0 = [0, 0], start1 = [3, 0]}>
// CHECK: %[[FTOI:.*]] = nova.ftoi(%[[RSQRT]]) : tensor<5x8xf32, {shape0 = [3, 8], shape1 = [2, 8], start0 = [0, 0], start1 = [3, 0]}> -> tensor<5x8xi8, {shape0 = [3, 8], shape1 = [2, 8], start0 = [0, 0], start1 = [3, 0]}>
// CHECK: return
// CHECK-LABEL: func.func @propagate_scalar_ops
// CHECK: %[[LOADS:.*]] = nova.load(%arg0) {start = [0, 0]} : memref<10x8xf32> -> tensor<5x8xf32, {shape0 = [3, 8], shape1 = [2, 8], start0 = [0, 0], start1 = [3, 0]}>
// CHECK: %[[SCALAR:.*]] = nova.scalar(%[[LOADS]]) {mode = 2 : i32, rhs = 1.250000e-01 : f32} : tensor<5x8xf32, {shape0 = [3, 8], shape1 = [2, 8], start0 = [0, 0], start1 = [3, 0]}> -> tensor<5x8xf32, {shape0 = [3, 8], shape1 = [2, 8], start0 = [0, 0], start1 = [3, 0]}>
// CHECK: %[[FMA:.*]] = nova.scalar_fma(%[[SCALAR]]) {a = 2.000000e+00 : f32, b = 3.000000e+00 : f32} : tensor<5x8xf32, {shape0 = [3, 8], shape1 = [2, 8], start0 = [0, 0], start1 = [3, 0]}> -> tensor<5x8xf32, {shape0 = [3, 8], shape1 = [2, 8], start0 = [0, 0], start1 = [3, 0]}>
// CHECK: return
// CHECK-LABEL: func.func @propagate_reduce_and_binary
// CHECK: %[[LOADR:.*]] = nova.load(%arg0) {start = [0, 0]} : memref<10x8xf32> -> tensor<5x8xf32, {shape0 = [3, 8], shape1 = [2, 8], start0 = [0, 0], start1 = [3, 0]}>
// CHECK: %[[REDUCE:.*]] = nova.reduce(%[[LOADR]]) {axis = 1 : i64, mode = 0 : i32} : tensor<5x8xf32, {shape0 = [3, 8], shape1 = [2, 8], start0 = [0, 0], start1 = [3, 0]}> -> tensor<5x1xf32, {shape0 = [3, 1], shape1 = [2, 1], start0 = [0, 0], start1 = [3, 0]}>
// CHECK: %[[BCAST:.*]] = nova.broadcast(%[[LOADR]], %[[REDUCE]]) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 0 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<5x8xf32, {shape0 = [3, 8], shape1 = [2, 8], start0 = [0, 0], start1 = [3, 0]}>, tensor<5x1xf32, {shape0 = [3, 1], shape1 = [2, 1], start0 = [0, 0], start1 = [3, 0]}> -> tensor<5x8xf32, {shape0 = [3, 8], shape1 = [2, 8], start0 = [0, 0], start1 = [3, 0]}>
// CHECK: %[[EW:.*]] = nova.elementwise(%[[BCAST]], %[[LOADR]]) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 0 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<5x8xf32, {shape0 = [3, 8], shape1 = [2, 8], start0 = [0, 0], start1 = [3, 0]}>, tensor<5x8xf32, {shape0 = [3, 8], shape1 = [2, 8], start0 = [0, 0], start1 = [3, 0]}> -> tensor<5x8xf32, {shape0 = [3, 8], shape1 = [2, 8], start0 = [0, 0], start1 = [3, 0]}>
// CHECK: return
// CHECK-LABEL: func.func @broadcast_propagates_max_threading
// CHECK: %[[LOADB0:.*]] = nova.load(%arg0) {start = [0, 0]} : memref<10x8xf32> -> tensor<5x8xf32, {shape0 = [3, 8], shape1 = [2, 8], start0 = [0, 0], start1 = [3, 0]}>
// CHECK: %[[LOADB1:.*]] = nova.load(%arg0) {start = [0, 0]} : memref<10x8xf32> -> tensor<3x8xf32, {shape0 = [2, 8], shape1 = [1, 8], start0 = [0, 0], start1 = [2, 0]}>
// CHECK: %[[BMM:.*]] = nova.broadcast(%[[LOADB0]], %[[LOADB1]]) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 0 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<5x8xf32, {shape0 = [3, 8], shape1 = [2, 8], start0 = [0, 0], start1 = [3, 0]}>, tensor<3x8xf32, {shape0 = [2, 8], shape1 = [1, 8], start0 = [0, 0], start1 = [2, 0]}> -> tensor<5x8xf32, {shape0 = [3, 8], shape1 = [2, 8], start0 = [0, 0], start1 = [3, 0]}>
// CHECK: return
// CHECK-LABEL: func.func @reduce_axis0_drops_reduce_threading_but_broadcast_recovers
// CHECK: %[[LOADA0:.*]] = nova.load(%arg0) {start = [0, 0]} : memref<10x8xf32> -> tensor<5x8xf32, {shape0 = [3, 8], shape1 = [2, 8], start0 = [0, 0], start1 = [3, 0]}>
// CHECK: %[[REDA0:.*]] = nova.reduce(%[[LOADA0]]) {axis = 0 : i64, mode = 0 : i32} : tensor<5x8xf32, {shape0 = [3, 8], shape1 = [2, 8], start0 = [0, 0], start1 = [3, 0]}> -> tensor<1x8xf32>
// CHECK: %[[BCA0:.*]] = nova.broadcast(%[[LOADA0]], %[[REDA0]]) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 0 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<5x8xf32, {shape0 = [3, 8], shape1 = [2, 8], start0 = [0, 0], start1 = [3, 0]}>, tensor<1x8xf32> -> tensor<5x8xf32, {shape0 = [3, 8], shape1 = [2, 8], start0 = [0, 0], start1 = [3, 0]}>
// CHECK: return
// CHECK-LABEL: func.func @matmul_propagates_lhs_and_strips_rhs
// CHECK: %[[LHS:.*]] = nova.load(%arg0) {start = [0, 0]} : memref<10x4xf32> -> tensor<5x4xf32, {shape0 = [3, 4], shape1 = [2, 4], start0 = [0, 0], start1 = [3, 0]}>
// CHECK: %[[RHS:.*]] = nova.load(%arg1) {start = [0, 0]} : memref<6x4xf32> -> tensor<3x4xf32, {shape0 = [3, 4], start0 = [0, 0]}>
// CHECK: %[[SCALE:.*]] = nova.load(%arg2) {start = [0, 0]} : memref<3x4xf32> -> tensor<3x4xf32, {shape0 = [3, 4], start0 = [0, 0]}>
// CHECK: %[[BIAS:.*]] = nova.load(%arg3) {start = [0, 0]} : memref<3x4xf32> -> tensor<3x4xf32, {shape0 = [3, 4], start0 = [0, 0]}>
// CHECK: %[[MM:.*]] = nova.matmul(%[[LHS]], %[[RHS]], %[[SCALE]], %[[BIAS]]) : tensor<5x4xf32, {shape0 = [3, 4], shape1 = [2, 4], start0 = [0, 0], start1 = [3, 0]}>, tensor<3x4xf32, {shape0 = [3, 4], start0 = [0, 0]}>, tensor<3x4xf32, {shape0 = [3, 4], start0 = [0, 0]}>, tensor<3x4xf32, {shape0 = [3, 4], start0 = [0, 0]}> -> tensor<5x4xf32, {shape0 = [3, 4], shape1 = [2, 4], start0 = [0, 0], start1 = [3, 0]}>
// CHECK: return
// CHECK-LABEL: func.func @overwrite_existing_threading
// CHECK: %[[LOAD3:.*]] = nova.load(%arg0) {start = [0, 0]} : memref<12x8xf32> -> tensor<6x8xf32, {shape0 = [3, 8], shape1 = [3, 8], start0 = [0, 0], start1 = [3, 0]}>
// CHECK: return
// CHECK-LABEL: func.func @load_sets_thread_slices
// CHECK: %[[LOAD4:.*]] = nova.load(%arg0) {start = [0, 0]} : memref<128x128xf32> -> tensor<64x128xf32, {shape0 = [32, 128], shape1 = [32, 128], start0 = [0, 0], start1 = [32, 0]}>
// CHECK: return
