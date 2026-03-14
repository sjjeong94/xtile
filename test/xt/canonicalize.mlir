// RUN: xt-opt --canonicalize %s | FileCheck %s

func.func @fold_add_zero(%arg0: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %zero = arith.constant dense<0.0> : tensor<16x16xf32>
  %0 = "xt.add"(%arg0, %zero) : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}

func.func @fold_add_zero_1d(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  %zero = arith.constant dense<0.0> : tensor<16xf32>
  %0 = "xt.add"(%arg0, %zero) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}

// CHECK-LABEL: func.func @fold_add_zero
// CHECK-NOT: xt.add
// CHECK: return %arg0 : tensor<16x16xf32>
// CHECK-LABEL: func.func @fold_add_zero_1d
// CHECK-NOT: xt.add
// CHECK: return %arg0 : tensor<16xf32>
