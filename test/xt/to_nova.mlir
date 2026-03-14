// RUN: xt-opt --xt-to-nova %s | FileCheck %s

module {
  func.func @reduce_ops(%a: tensor<16x16xf32>) -> (tensor<16x1xf32>, tensor<16x1xf32>) {
    %0 = "xt.reduce_sum"(%a) : (tensor<16x16xf32>) -> tensor<16x1xf32>
    %1 = "xt.reduce_max"(%a) : (tensor<16x16xf32>) -> tensor<16x1xf32>
    return %0, %1 : tensor<16x1xf32>, tensor<16x1xf32>
  }

  func.func @broadcast_sub(%a: tensor<16x16xf32>, %b: tensor<16x1xf32>) -> tensor<16x16xf32> {
    %0 = "xt.sub"(%a, %b) : (tensor<16x16xf32>, tensor<16x1xf32>) -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }

  func.func @elementwise_mul(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %0 = "xt.mul"(%a, %b) : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }

  func.func @scalar_like_mul(%a: tensor<16x1xf32>, %b: tensor<1x1xf32>) -> tensor<16x1xf32> {
    %0 = "xt.mul"(%a, %b) : (tensor<16x1xf32>, tensor<1x1xf32>) -> tensor<16x1xf32>
    return %0 : tensor<16x1xf32>
  }

  func.func @constant_scalar_mul(%a: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %cst = arith.constant dense<1.250000e-01> : tensor<1x1xf32>
    %0 = "xt.mul"(%a, %cst) : (tensor<16x16xf32>, tensor<1x1xf32>) -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }

  func.func @constant_scalar_sub(%a: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %cst = arith.constant dense<1.250000e-01> : tensor<1x1xf32>
    %0 = "xt.sub"(%a, %cst) : (tensor<16x16xf32>, tensor<1x1xf32>) -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }

  func.func @matmul(%a: tensor<16x32xf32>, %b: tensor<32x8xf32>) -> tensor<16x8xf32> {
    %0 = "xt.matmul"(%a, %b) : (tensor<16x32xf32>, tensor<32x8xf32>) -> tensor<16x8xf32>
    return %0 : tensor<16x8xf32>
  }
}

// CHECK-LABEL: func.func @reduce_ops
// CHECK: "nova.reduce"(%arg0)
// CHECK-SAME: mode = 0 : i32
// CHECK: "nova.reduce"(%arg0)
// CHECK-SAME: mode = 1 : i32
// CHECK-LABEL: func.func @broadcast_sub
// CHECK: "nova.broadcast"(%arg0, %arg1)
// CHECK-SAME: lhs_b = 0.000000e+00 : f32
// CHECK-SAME: lhs_s = 1.000000e+00 : f32
// CHECK-SAME: mode = 3 : i32
// CHECK-SAME: rhs_b = 0.000000e+00 : f32
// CHECK-SAME: rhs_s = 1.000000e+00 : f32
// CHECK-LABEL: func.func @elementwise_mul
// CHECK: "nova.elementwise"(%arg0, %arg1)
// CHECK-SAME: lhs_b = 0.000000e+00 : f32
// CHECK-SAME: lhs_s = 1.000000e+00 : f32
// CHECK-SAME: mode = 2 : i32
// CHECK-SAME: rhs_b = 0.000000e+00 : f32
// CHECK-SAME: rhs_s = 1.000000e+00 : f32
// CHECK-LABEL: func.func @scalar_like_mul
// CHECK: xt.mul(%arg0, %arg1) : (tensor<16x1xf32>, tensor<1x1xf32>) -> tensor<16x1xf32>
// CHECK-LABEL: func.func @constant_scalar_mul
// CHECK: "nova.scalar"(%arg0)
// CHECK-SAME: mode = 2 : i32
// CHECK-SAME: rhs = 1.250000e-01 : f32
// CHECK: : (tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK-LABEL: func.func @constant_scalar_sub
// CHECK: "nova.scalar"(%arg0)
// CHECK-SAME: mode = 1 : i32
// CHECK-SAME: rhs = -1.250000e-01 : f32
// CHECK: : (tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK-LABEL: func.func @matmul
// CHECK: %[[SCALE:.*]] = arith.constant dense<1.000000e+00> : tensor<1x1xf32>
// CHECK: %[[BIAS:.*]] = arith.constant dense<0.000000e+00> : tensor<1x1xf32>
// CHECK: "nova.matmul"(%arg0, %arg1, %[[SCALE]], %[[BIAS]]) : (tensor<16x32xf32>, tensor<32x8xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<16x8xf32>
