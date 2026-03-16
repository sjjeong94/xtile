// RUN: xt-opt --nova-optimize %s | FileCheck %s

module {
  func.func @fold_rhs_scalar_into_broadcast(%arg0: tensor<16x16xf32>, %arg1: tensor<16x1xf32>) -> tensor<16x16xf32> {
    %0 = nova.scalar(%arg1) {mode = 1 : i32, rhs = 2.500000e-01 : f32} : tensor<16x1xf32> -> tensor<16x1xf32>
    %1 = "nova.broadcast"(%arg0, %0) <{lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 3 : i32, rhs_a = 4.000000e+00 : f32, rhs_b = 5.000000e-01 : f32}> : (tensor<16x16xf32>, tensor<16x1xf32>) -> tensor<16x16xf32>
    return %1 : tensor<16x16xf32>
  }

  func.func @fold_lhs_scalar_into_elementwise(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %0 = nova.scalar(%arg0) {mode = 2 : i32, rhs = 5.000000e-01 : f32} : tensor<16x16xf32> -> tensor<16x16xf32>
    %1 = "nova.elementwise"(%0, %arg1) <{lhs_a = 8.000000e+00 : f32, lhs_b = 1.000000e+00 : f32, mode = 2 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32}> : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    return %1 : tensor<16x16xf32>
  }

  func.func @fold_both_sides(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %0 = nova.scalar(%arg0) {mode = 1 : i32, rhs = 1.250000e-01 : f32} : tensor<16x16xf32> -> tensor<16x16xf32>
    %1 = nova.scalar(%arg1) {mode = 2 : i32, rhs = 2.000000e+00 : f32} : tensor<16x16xf32> -> tensor<16x16xf32>
    %2 = "nova.elementwise"(%0, %1) <{lhs_a = 8.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 1 : i32, rhs_a = 5.000000e-01 : f32, rhs_b = 3.000000e+00 : f32}> : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    return %2 : tensor<16x16xf32>
  }

  func.func @fold_sub_scalar_as_add_with_negated_rhs(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %0 = nova.scalar(%arg1) {mode = 3 : i32, rhs = 7.500000e-01 : f32} : tensor<16x16xf32> -> tensor<16x16xf32>
    %1 = "nova.elementwise"(%arg0, %0) <{lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 2 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32}> : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    return %1 : tensor<16x16xf32>
  }

  func.func @fold_scalar_mul_into_matmul(%lhs: tensor<16x32xf32>, %rhs: tensor<32x8xf32>) -> tensor<16x8xf32> {
    %scale = arith.constant dense<1.000000e+00> : tensor<1x1xf32>
    %bias = arith.constant dense<5.000000e-01> : tensor<1x1xf32>
    %0 = nova.matmul(%lhs, %rhs, %scale, %bias) : tensor<16x32xf32>, tensor<32x8xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xf32>
    %1 = nova.scalar(%0) {mode = 2 : i32, rhs = 2.000000e+00 : f32} : tensor<16x8xf32> -> tensor<16x8xf32>
    return %1 : tensor<16x8xf32>
  }

  func.func @fold_scalar_add_into_matmul(%lhs: tensor<16x32xf32>, %rhs: tensor<32x8xf32>) -> tensor<16x8xf32> {
    %scale = arith.constant dense<4.000000e+00> : tensor<1x1xf32>
    %bias = arith.constant dense<5.000000e-01> : tensor<1x1xf32>
    %0 = nova.matmul(%lhs, %rhs, %scale, %bias) : tensor<16x32xf32>, tensor<32x8xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xf32>
    %1 = nova.scalar(%0) {mode = 1 : i32, rhs = 2.500000e-01 : f32} : tensor<16x8xf32> -> tensor<16x8xf32>
    return %1 : tensor<16x8xf32>
  }

  func.func @fuse_scalar_mul_then_add(%arg0: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %0 = nova.scalar(%arg0) {mode = 2 : i32, rhs = 3.000000e+00 : f32} : tensor<16x16xf32> -> tensor<16x16xf32>
    %1 = nova.scalar(%0) {mode = 1 : i32, rhs = 4.000000e+00 : f32} : tensor<16x16xf32> -> tensor<16x16xf32>
    return %1 : tensor<16x16xf32>
  }

  func.func @do_not_fuse_scalar_add_then_mul(%arg0: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %0 = nova.scalar(%arg0) {mode = 1 : i32, rhs = 4.000000e+00 : f32} : tensor<16x16xf32> -> tensor<16x16xf32>
    %1 = nova.scalar(%0) {mode = 2 : i32, rhs = 3.000000e+00 : f32} : tensor<16x16xf32> -> tensor<16x16xf32>
    return %1 : tensor<16x16xf32>
  }

  func.func @fold_broadcast_mul_into_matmul(%lhs: tensor<16x32xf32>, %rhs: tensor<32x8xf32>, %scale_arg: tensor<1x1xf32>) -> tensor<16x8xf32> {
    %scale = arith.constant dense<1.000000e+00> : tensor<1x1xf32>
    %bias = arith.constant dense<0.000000e+00> : tensor<1x1xf32>
    %0 = nova.matmul(%lhs, %rhs, %scale, %bias) : tensor<16x32xf32>, tensor<32x8xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xf32>
    %1 = "nova.broadcast"(%0, %scale_arg) <{lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 2 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32}> : (tensor<16x8xf32>, tensor<1x1xf32>) -> tensor<16x8xf32>
    return %1 : tensor<16x8xf32>
  }

  func.func @do_not_fold_broadcast_mul_into_multiuse_matmul(%lhs: tensor<16x32xf32>, %rhs: tensor<32x8xf32>, %scale_arg: tensor<1x1xf32>) -> (tensor<16x8xf32>, tensor<16x8xf32>) {
    %scale = arith.constant dense<1.000000e+00> : tensor<1x1xf32>
    %bias = arith.constant dense<0.000000e+00> : tensor<1x1xf32>
    %0 = nova.matmul(%lhs, %rhs, %scale, %bias) : tensor<16x32xf32>, tensor<32x8xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xf32>
    %1 = "nova.broadcast"(%0, %scale_arg) <{lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 2 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32}> : (tensor<16x8xf32>, tensor<1x1xf32>) -> tensor<16x8xf32>
    return %0, %1 : tensor<16x8xf32>, tensor<16x8xf32>
  }

  func.func @fold_broadcast_add_into_matmul(%lhs: tensor<16x32xf32>, %rhs: tensor<32x8xf32>, %bias_arg: tensor<1x1xf32>) -> tensor<16x8xf32> {
    %scale = arith.constant dense<3.000000e+00> : tensor<1x1xf32>
    %bias = arith.constant dense<0.000000e+00> : tensor<1x1xf32>
    %0 = nova.matmul(%lhs, %rhs, %scale, %bias) : tensor<16x32xf32>, tensor<32x8xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xf32>
    %1 = "nova.broadcast"(%0, %bias_arg) <{lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 1 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32}> : (tensor<16x8xf32>, tensor<1x1xf32>) -> tensor<16x8xf32>
    return %1 : tensor<16x8xf32>
  }

  func.func @do_not_fold_broadcast_add_into_multiuse_matmul(%lhs: tensor<16x32xf32>, %rhs: tensor<32x8xf32>, %bias_arg: tensor<1x1xf32>) -> (tensor<16x8xf32>, tensor<16x8xf32>) {
    %scale = arith.constant dense<3.000000e+00> : tensor<1x1xf32>
    %bias = arith.constant dense<0.000000e+00> : tensor<1x1xf32>
    %0 = nova.matmul(%lhs, %rhs, %scale, %bias) : tensor<16x32xf32>, tensor<32x8xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xf32>
    %1 = "nova.broadcast"(%0, %bias_arg) <{lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 1 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32}> : (tensor<16x8xf32>, tensor<1x1xf32>) -> tensor<16x8xf32>
    return %0, %1 : tensor<16x8xf32>, tensor<16x8xf32>
  }
}

// CHECK-LABEL: func.func @fold_rhs_scalar_into_broadcast
// CHECK: nova.broadcast(%arg0, %arg1) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 3 : i32, rhs_a = 4.000000e+00 : f32, rhs_b = 1.500000e+00 : f32} : tensor<16x16xf32>, tensor<16x1xf32> -> tensor<16x16xf32>

// CHECK-LABEL: func.func @fold_lhs_scalar_into_elementwise
// CHECK: nova.elementwise(%arg0, %arg1) {lhs_a = 4.000000e+00 : f32, lhs_b = 1.000000e+00 : f32, mode = 2 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf32>

// CHECK-LABEL: func.func @fold_both_sides
// CHECK: nova.elementwise(%arg0, %arg1) {lhs_a = 8.000000e+00 : f32, lhs_b = 1.000000e+00 : f32, mode = 1 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 3.000000e+00 : f32} : tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf32>

// CHECK-LABEL: func.func @fold_sub_scalar_as_add_with_negated_rhs
// CHECK: nova.elementwise(%arg0, %arg1) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 2 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = -7.500000e-01 : f32} : tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf32>

// CHECK-LABEL: func.func @fold_scalar_mul_into_matmul
// CHECK: %[[SCALE:.*]] = arith.constant dense<2.000000e+00> : tensor<1x1xf32>
// CHECK: %[[BIAS:.*]] = arith.constant dense<1.000000e+00> : tensor<1x1xf32>
// CHECK: nova.matmul(%arg0, %arg1, %[[SCALE]], %[[BIAS]]) : tensor<16x32xf32>, tensor<32x8xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xf32>

// CHECK-LABEL: func.func @fold_scalar_add_into_matmul
// CHECK: %[[SCALE:.*]] = arith.constant dense<4.000000e+00> : tensor<1x1xf32>
// CHECK: %[[BIAS:.*]] = arith.constant dense<7.500000e-01> : tensor<1x1xf32>
// CHECK: nova.matmul(%arg0, %arg1, %[[SCALE]], %[[BIAS]]) : tensor<16x32xf32>, tensor<32x8xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xf32>

// CHECK-LABEL: func.func @fuse_scalar_mul_then_add
// CHECK: nova.scalar_fma(%arg0) {a = 3.000000e+00 : f32, b = 4.000000e+00 : f32} : tensor<16x16xf32> -> tensor<16x16xf32>

// CHECK-LABEL: func.func @do_not_fuse_scalar_add_then_mul
// CHECK: %[[ADD:.*]] = nova.scalar(%arg0) {mode = 1 : i32, rhs = 4.000000e+00 : f32} : tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK: %[[MUL:.*]] = nova.scalar(%[[ADD]]) {mode = 2 : i32, rhs = 3.000000e+00 : f32} : tensor<16x16xf32> -> tensor<16x16xf32>

// CHECK-LABEL: func.func @fold_broadcast_mul_into_matmul
// CHECK-NOT: "nova.broadcast"
// CHECK: nova.matmul(%arg0, %arg1, %arg2, %{{.*}}) : tensor<16x32xf32>, tensor<32x8xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xf32>

// CHECK-LABEL: func.func @do_not_fold_broadcast_mul_into_multiuse_matmul
// CHECK: %[[SCALE:.*]] = arith.constant dense<1.000000e+00> : tensor<1x1xf32>
// CHECK: %[[BIAS:.*]] = arith.constant dense<0.000000e+00> : tensor<1x1xf32>
// CHECK: %[[MM:.*]] = nova.matmul(%arg0, %arg1, %[[SCALE]], %[[BIAS]]) : tensor<16x32xf32>, tensor<32x8xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xf32>
// CHECK: %[[BCAST:.*]] = nova.broadcast(%[[MM]], %arg2) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 2 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<16x8xf32>, tensor<1x1xf32> -> tensor<16x8xf32>

// CHECK-LABEL: func.func @fold_broadcast_add_into_matmul
// CHECK-NOT: "nova.broadcast"
// CHECK: nova.matmul(%arg0, %arg1, %{{.*}}, %arg2) : tensor<16x32xf32>, tensor<32x8xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xf32>

// CHECK-LABEL: func.func @do_not_fold_broadcast_add_into_multiuse_matmul
// CHECK: %[[SCALE2:.*]] = arith.constant dense<3.000000e+00> : tensor<1x1xf32>
// CHECK: %[[BIAS2:.*]] = arith.constant dense<0.000000e+00> : tensor<1x1xf32>
// CHECK: %[[MM2:.*]] = nova.matmul(%arg0, %arg1, %[[SCALE2]], %[[BIAS2]]) : tensor<16x32xf32>, tensor<32x8xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xf32>
// CHECK: %[[BCAST2:.*]] = nova.broadcast(%[[MM2]], %arg2) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 1 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<16x8xf32>, tensor<1x1xf32> -> tensor<16x8xf32>
