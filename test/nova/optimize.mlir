// RUN: xt-opt --nova-optimize %s | FileCheck %s

module {
  func.func @fold_rhs_scalar_into_broadcast(%arg0: tensor<16x16xf32>, %arg1: tensor<16x1xf32>) -> tensor<16x16xf32> {
    %0 = "nova.scalar"(%arg1) <{mode = 1 : i32, rhs = 2.500000e-01 : f32}> : (tensor<16x1xf32>) -> tensor<16x1xf32>
    %1 = "nova.broadcast"(%arg0, %0) <{lhs_b = 0.000000e+00 : f32, lhs_s = 1.000000e+00 : f32, mode = 3 : i32, rhs_b = 5.000000e-01 : f32, rhs_s = 4.000000e+00 : f32}> : (tensor<16x16xf32>, tensor<16x1xf32>) -> tensor<16x16xf32>
    return %1 : tensor<16x16xf32>
  }

  func.func @fold_lhs_scalar_into_elementwise(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %0 = "nova.scalar"(%arg0) <{mode = 2 : i32, rhs = 5.000000e-01 : f32}> : (tensor<16x16xf32>) -> tensor<16x16xf32>
    %1 = "nova.elementwise"(%0, %arg1) <{lhs_b = 1.000000e+00 : f32, lhs_s = 8.000000e+00 : f32, mode = 2 : i32, rhs_b = 0.000000e+00 : f32, rhs_s = 1.000000e+00 : f32}> : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    return %1 : tensor<16x16xf32>
  }

  func.func @fold_both_sides(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %0 = "nova.scalar"(%arg0) <{mode = 1 : i32, rhs = 1.250000e-01 : f32}> : (tensor<16x16xf32>) -> tensor<16x16xf32>
    %1 = "nova.scalar"(%arg1) <{mode = 2 : i32, rhs = 2.000000e+00 : f32}> : (tensor<16x16xf32>) -> tensor<16x16xf32>
    %2 = "nova.elementwise"(%0, %1) <{lhs_b = 0.000000e+00 : f32, lhs_s = 8.000000e+00 : f32, mode = 1 : i32, rhs_b = 3.000000e+00 : f32, rhs_s = 5.000000e-01 : f32}> : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    return %2 : tensor<16x16xf32>
  }

  func.func @fold_sub_scalar_as_add_with_negated_rhs(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %0 = "nova.scalar"(%arg1) <{mode = 3 : i32, rhs = 7.500000e-01 : f32}> : (tensor<16x16xf32>) -> tensor<16x16xf32>
    %1 = "nova.elementwise"(%arg0, %0) <{lhs_b = 0.000000e+00 : f32, lhs_s = 1.000000e+00 : f32, mode = 2 : i32, rhs_b = 0.000000e+00 : f32, rhs_s = 1.000000e+00 : f32}> : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    return %1 : tensor<16x16xf32>
  }
}

// CHECK-LABEL: func.func @fold_rhs_scalar_into_broadcast
// CHECK: "nova.broadcast"(%arg0, %arg1)
// CHECK-SAME: lhs_b = 0.000000e+00 : f32
// CHECK-SAME: lhs_s = 1.000000e+00 : f32
// CHECK-SAME: mode = 3 : i32
// CHECK-SAME: rhs_b = 1.500000e+00 : f32
// CHECK-SAME: rhs_s = 4.000000e+00 : f32

// CHECK-LABEL: func.func @fold_lhs_scalar_into_elementwise
// CHECK: "nova.elementwise"(%arg0, %arg1)
// CHECK-SAME: lhs_b = 1.000000e+00 : f32
// CHECK-SAME: lhs_s = 4.000000e+00 : f32
// CHECK-SAME: mode = 2 : i32

// CHECK-LABEL: func.func @fold_both_sides
// CHECK: "nova.elementwise"(%arg0, %arg1)
// CHECK-SAME: lhs_b = 1.000000e+00 : f32
// CHECK-SAME: lhs_s = 8.000000e+00 : f32
// CHECK-SAME: mode = 1 : i32
// CHECK-SAME: rhs_b = 3.000000e+00 : f32
// CHECK-SAME: rhs_s = 1.000000e+00 : f32

// CHECK-LABEL: func.func @fold_sub_scalar_as_add_with_negated_rhs
// CHECK: "nova.elementwise"(%arg0, %arg1)
// CHECK-SAME: lhs_b = 0.000000e+00 : f32
// CHECK-SAME: lhs_s = 1.000000e+00 : f32
// CHECK-SAME: mode = 2 : i32
// CHECK-SAME: rhs_b = -7.500000e-01 : f32
// CHECK-SAME: rhs_s = 1.000000e+00 : f32
