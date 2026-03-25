// RUN: xt-opt --nova-optimize %s | FileCheck %s

module {
  func.func @fold_rhs_scalar_into_broadcast(%arg0: tensor<16x16xf32>, %arg1: tensor<16x1xf32>) -> tensor<16x16xf32> {
    %0 = nova.scalar 1 %arg1, 2.500000e-01 : tensor<16x1xf32> -> tensor<16x1xf32>
    %1 = nova.broadcast 1 %arg0, %0 lhs 1.000000e+00 0.000000e+00 rhs 4.000000e+00 0.000000e+00 : tensor<16x16xf32>, tensor<16x1xf32> -> tensor<16x16xf32>
    return %1 : tensor<16x16xf32>
  }

  func.func @fold_lhs_scalar_into_elementwise(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %0 = nova.scalar 2 %arg0, 5.000000e-01 : tensor<16x16xf32> -> tensor<16x16xf32>
    %1 = nova.elementwise 1 %0, %arg1 lhs 8.000000e+00 1.000000e+00 rhs 1.000000e+00 0.000000e+00 : tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf32>
    return %1 : tensor<16x16xf32>
  }

  func.func @fold_both_sides(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %0 = nova.scalar 1 %arg0, 1.250000e-01 : tensor<16x16xf32> -> tensor<16x16xf32>
    %1 = nova.scalar 2 %arg1, 2.000000e+00 : tensor<16x16xf32> -> tensor<16x16xf32>
    %2 = nova.elementwise 1 %0, %1 lhs 8.000000e+00 0.000000e+00 rhs 1.000000e+00 5.000000e-01 : tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf32>
    return %2 : tensor<16x16xf32>
  }

  func.func @fold_sub_scalar_as_add_with_negated_rhs(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %0 = nova.scalar 3 %arg1, 7.500000e-01 : tensor<16x16xf32> -> tensor<16x16xf32>
    %1 = nova.elementwise 1 %arg0, %0 lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00 : tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf32>
    return %1 : tensor<16x16xf32>
  }

  func.func @fold_scalar_mul_into_matmul(%lhs: tensor<16x32xf32>, %rhs: tensor<32x8xf32>) -> tensor<16x8xf32> {
    %scale = arith.constant dense<1.000000e+00> : tensor<1x1xf32>
    %bias = arith.constant dense<5.000000e-01> : tensor<1x1xf32>
    %0 = nova.matmul %lhs, %rhs, %scale, %bias : tensor<16x32xf32>, tensor<32x8xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xf32>
    %1 = nova.scalar 2 %0, 2.000000e+00 : tensor<16x8xf32> -> tensor<16x8xf32>
    return %1 : tensor<16x8xf32>
  }

  func.func @fold_scalar_add_into_matmul(%lhs: tensor<16x32xf32>, %rhs: tensor<32x8xf32>) -> tensor<16x8xf32> {
    %scale = arith.constant dense<4.000000e+00> : tensor<1x1xf32>
    %bias = arith.constant dense<5.000000e-01> : tensor<1x1xf32>
    %0 = nova.matmul %lhs, %rhs, %scale, %bias : tensor<16x32xf32>, tensor<32x8xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xf32>
    %1 = nova.scalar 1 %0, 2.500000e-01 : tensor<16x8xf32> -> tensor<16x8xf32>
    return %1 : tensor<16x8xf32>
  }

  func.func @fuse_scalar_mul_then_add(%arg0: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %0 = nova.scalar 2 %arg0, 3.000000e+00 : tensor<16x16xf32> -> tensor<16x16xf32>
    %1 = nova.scalar 1 %0, 4.000000e+00 : tensor<16x16xf32> -> tensor<16x16xf32>
    return %1 : tensor<16x16xf32>
  }

  func.func @do_not_fuse_scalar_add_then_mul(%arg0: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %0 = nova.scalar 1 %arg0, 4.000000e+00 : tensor<16x16xf32> -> tensor<16x16xf32>
    %1 = nova.scalar 2 %0, 3.000000e+00 : tensor<16x16xf32> -> tensor<16x16xf32>
    return %1 : tensor<16x16xf32>
  }

  func.func @fold_broadcast_mul_into_matmul(%lhs: tensor<16x32xf32>, %rhs: tensor<32x8xf32>, %scale_arg: tensor<1x1xf32>) -> tensor<16x8xf32> {
    %scale = arith.constant dense<1.000000e+00> : tensor<1x1xf32>
    %bias = arith.constant dense<0.000000e+00> : tensor<1x1xf32>
    %0 = nova.matmul %lhs, %rhs, %scale, %bias : tensor<16x32xf32>, tensor<32x8xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xf32>
    %1 = nova.broadcast 2 %0, %scale_arg lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00 : tensor<16x8xf32>, tensor<1x1xf32> -> tensor<16x8xf32>
    return %1 : tensor<16x8xf32>
  }

  func.func @do_not_fold_broadcast_mul_into_multiuse_matmul(%lhs: tensor<16x32xf32>, %rhs: tensor<32x8xf32>, %scale_arg: tensor<1x1xf32>) -> (tensor<16x8xf32>, tensor<16x8xf32>) {
    %scale = arith.constant dense<1.000000e+00> : tensor<1x1xf32>
    %bias = arith.constant dense<0.000000e+00> : tensor<1x1xf32>
    %0 = nova.matmul %lhs, %rhs, %scale, %bias : tensor<16x32xf32>, tensor<32x8xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xf32>
    %1 = nova.broadcast 2 %0, %scale_arg lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00 : tensor<16x8xf32>, tensor<1x1xf32> -> tensor<16x8xf32>
    return %0, %1 : tensor<16x8xf32>, tensor<16x8xf32>
  }

  func.func @fold_broadcast_add_into_matmul(%lhs: tensor<16x32xf32>, %rhs: tensor<32x8xf32>, %bias_arg: tensor<1x1xf32>) -> tensor<16x8xf32> {
    %scale = arith.constant dense<3.000000e+00> : tensor<1x1xf32>
    %bias = arith.constant dense<0.000000e+00> : tensor<1x1xf32>
    %0 = nova.matmul %lhs, %rhs, %scale, %bias : tensor<16x32xf32>, tensor<32x8xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xf32>
    %1 = nova.broadcast 1 %0, %bias_arg lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00 : tensor<16x8xf32>, tensor<1x1xf32> -> tensor<16x8xf32>
    return %1 : tensor<16x8xf32>
  }

  func.func @do_not_fold_broadcast_add_into_multiuse_matmul(%lhs: tensor<16x32xf32>, %rhs: tensor<32x8xf32>, %bias_arg: tensor<1x1xf32>) -> (tensor<16x8xf32>, tensor<16x8xf32>) {
    %scale = arith.constant dense<3.000000e+00> : tensor<1x1xf32>
    %bias = arith.constant dense<0.000000e+00> : tensor<1x1xf32>
    %0 = nova.matmul %lhs, %rhs, %scale, %bias : tensor<16x32xf32>, tensor<32x8xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xf32>
    %1 = nova.broadcast 1 %0, %bias_arg lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00 : tensor<16x8xf32>, tensor<1x1xf32> -> tensor<16x8xf32>
    return %0, %1 : tensor<16x8xf32>, tensor<16x8xf32>
  }

  func.func @do_not_fold_scalar_into_int_matmul(%lhs: tensor<16x32xi8>, %rhs: tensor<32x8xi8>) -> tensor<16x8xi32> {
    %scale = arith.constant dense<1.000000e+00> : tensor<1x1xf32>
    %bias = arith.constant dense<0.000000e+00> : tensor<1x1xf32>
    %0 = nova.matmul %lhs, %rhs, %scale, %bias : tensor<16x32xi8>, tensor<32x8xi8>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xi32>
    %1 = nova.scalar 1 %0, 2.500000e-01 : tensor<16x8xi32> -> tensor<16x8xi32>
    return %1 : tensor<16x8xi32>
  }

  func.func @do_not_fold_broadcast_mul_into_int_matmul(%lhs: tensor<16x32xi8>, %rhs: tensor<32x8xi8>, %scale_arg: tensor<1x1xf32>) -> tensor<16x8xi32> {
    %scale = arith.constant dense<1.000000e+00> : tensor<1x1xf32>
    %bias = arith.constant dense<0.000000e+00> : tensor<1x1xf32>
    %0 = nova.matmul %lhs, %rhs, %scale, %bias : tensor<16x32xi8>, tensor<32x8xi8>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xi32>
    %1 = nova.broadcast 2 %0, %scale_arg lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00 : tensor<16x8xi32>, tensor<1x1xf32> -> tensor<16x8xi32>
    return %1 : tensor<16x8xi32>
  }

  func.func @do_not_fold_broadcast_add_into_int_matmul(%lhs: tensor<16x32xi8>, %rhs: tensor<32x8xi8>, %bias_arg: tensor<1x1xf32>) -> tensor<16x8xi32> {
    %scale = arith.constant dense<3.000000e+00> : tensor<1x1xf32>
    %bias = arith.constant dense<0.000000e+00> : tensor<1x1xf32>
    %0 = nova.matmul %lhs, %rhs, %scale, %bias : tensor<16x32xi8>, tensor<32x8xi8>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xi32>
    %1 = nova.broadcast 1 %0, %bias_arg lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00 : tensor<16x8xi32>, tensor<1x1xf32> -> tensor<16x8xi32>
    return %1 : tensor<16x8xi32>
  }

  func.func @fold_ftoi_into_matmul(%lhs: tensor<16x32xi8>, %rhs: tensor<32x8xi8>) -> tensor<16x8xi32> {
    %scale = arith.constant dense<1.000000e+00> : tensor<1x1xf32>
    %bias = arith.constant dense<0.000000e+00> : tensor<1x1xf32>
    %0 = nova.matmul %lhs, %rhs, %scale, %bias : tensor<16x32xi8>, tensor<32x8xi8>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xf32>
    %1 = nova.ftoi %0 : tensor<16x8xf32> -> tensor<16x8xi32>
    return %1 : tensor<16x8xi32>
  }
}

// CHECK-LABEL: func.func @fold_rhs_scalar_into_broadcast
// CHECK: nova.broadcast 1 %arg0, %arg1 lhs 1.000000e+00 0.000000e+00 rhs 4.000000e+00 1.000000e+00 : tensor<16x16xf32>, tensor<16x1xf32> -> tensor<16x16xf32>

// CHECK-LABEL: func.func @fold_lhs_scalar_into_elementwise
// CHECK: nova.elementwise 1 %arg0, %arg1 lhs 4.000000e+00 1.000000e+00 rhs 1.000000e+00 0.000000e+00 : tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf32>

// CHECK-LABEL: func.func @fold_both_sides
// CHECK: nova.elementwise 1 %arg0, %arg1 lhs 8.000000e+00 1.000000e+00 rhs 2.000000e+00 5.000000e-01 : tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf32>

// CHECK-LABEL: func.func @fold_sub_scalar_as_add_with_negated_rhs
// CHECK: nova.elementwise 1 %arg0, %arg1 lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 -7.500000e-01 : tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf32>

// CHECK-LABEL: func.func @fold_scalar_mul_into_matmul
// CHECK: %[[SCALE:.*]] = arith.constant dense<2.000000e+00> : tensor<1x1xf32>
// CHECK: %[[BIAS:.*]] = arith.constant dense<1.000000e+00> : tensor<1x1xf32>
// CHECK: nova.matmul %arg0, %arg1, %[[SCALE]], %[[BIAS]] : tensor<16x32xf32>, tensor<32x8xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xf32>

// CHECK-LABEL: func.func @fold_scalar_add_into_matmul
// CHECK: %[[SCALE2:.*]] = arith.constant dense<4.000000e+00> : tensor<1x1xf32>
// CHECK: %[[BIAS2:.*]] = arith.constant dense<7.500000e-01> : tensor<1x1xf32>
// CHECK: nova.matmul %arg0, %arg1, %[[SCALE2]], %[[BIAS2]] : tensor<16x32xf32>, tensor<32x8xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xf32>

// CHECK-LABEL: func.func @fuse_scalar_mul_then_add
// CHECK: nova.scalar_fma %arg0, 3.000000e+00, 4.000000e+00 : tensor<16x16xf32> -> tensor<16x16xf32>

// CHECK-LABEL: func.func @do_not_fuse_scalar_add_then_mul
// CHECK: %[[ADD:.*]] = nova.scalar 1 %arg0, 4.000000e+00 : tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK: %[[MUL:.*]] = nova.scalar 2 %[[ADD]], 3.000000e+00 : tensor<16x16xf32> -> tensor<16x16xf32>

// CHECK-LABEL: func.func @fold_broadcast_mul_into_matmul
// CHECK-NOT: nova.broadcast
// CHECK: nova.matmul %arg0, %arg1, %arg2, %{{.*}} : tensor<16x32xf32>, tensor<32x8xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xf32>

// CHECK-LABEL: func.func @do_not_fold_broadcast_mul_into_multiuse_matmul
// CHECK: %[[MM:.*]] = nova.matmul %arg0, %arg1, %{{.*}}, %{{.*}} : tensor<16x32xf32>, tensor<32x8xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xf32>
// CHECK: %[[BCAST:.*]] = nova.broadcast 2 %[[MM]], %arg2 lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00 : tensor<16x8xf32>, tensor<1x1xf32> -> tensor<16x8xf32>

// CHECK-LABEL: func.func @fold_broadcast_add_into_matmul
// CHECK-NOT: nova.broadcast
// CHECK: nova.matmul %arg0, %arg1, %{{.*}}, %arg2 : tensor<16x32xf32>, tensor<32x8xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xf32>

// CHECK-LABEL: func.func @do_not_fold_broadcast_add_into_multiuse_matmul
// CHECK: %[[MM2:.*]] = nova.matmul %arg0, %arg1, %{{.*}}, %{{.*}} : tensor<16x32xf32>, tensor<32x8xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xf32>
// CHECK: %[[BCAST2:.*]] = nova.broadcast 1 %[[MM2]], %arg2 lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00 : tensor<16x8xf32>, tensor<1x1xf32> -> tensor<16x8xf32>

// CHECK-LABEL: func.func @do_not_fold_scalar_into_int_matmul
// CHECK: %[[IMM:.*]] = nova.matmul %arg0, %arg1, %{{.*}}, %{{.*}} : tensor<16x32xi8>, tensor<32x8xi8>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xi32>
// CHECK: nova.scalar 1 %[[IMM]], 2.500000e-01 : tensor<16x8xi32> -> tensor<16x8xi32>

// CHECK-LABEL: func.func @do_not_fold_broadcast_mul_into_int_matmul
// CHECK: %[[IMMMUL:.*]] = nova.matmul %arg0, %arg1, %{{.*}}, %{{.*}} : tensor<16x32xi8>, tensor<32x8xi8>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xi32>
// CHECK: nova.broadcast 2 %[[IMMMUL]], %arg2 lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00 : tensor<16x8xi32>, tensor<1x1xf32> -> tensor<16x8xi32>

// CHECK-LABEL: func.func @do_not_fold_broadcast_add_into_int_matmul
// CHECK: %[[IMMADD:.*]] = nova.matmul %arg0, %arg1, %{{.*}}, %{{.*}} : tensor<16x32xi8>, tensor<32x8xi8>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xi32>
// CHECK: nova.broadcast 1 %[[IMMADD]], %arg2 lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00 : tensor<16x8xi32>, tensor<1x1xf32> -> tensor<16x8xi32>

// CHECK-LABEL: func.func @fold_ftoi_into_matmul
// CHECK-NOT: nova.ftoi
// CHECK: nova.matmul %arg0, %arg1, %{{.*}}, %{{.*}} : tensor<16x32xi8>, tensor<32x8xi8>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xi32>
