// RUN: xt-opt --xt-liveness %s | FileCheck %s

func.func @basic_liveness(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %0 = xt.add(%arg0, %arg1) : tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf32>
  %1 = xt.exp(%0) : tensor<16x16xf32> -> tensor<16x16xf32>
  %2 = xt.mul(%1, %arg1) : tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf32>
  func.return %2 : tensor<16x16xf32>
}

// CHECK-LABEL: liveness @basic_liveness
// CHECK: op: xt.add
// CHECK-SAME: live_in=[%arg0, %arg1]
// CHECK-SAME: live_out=[%0, %arg1]
// CHECK: op: xt.exp
// CHECK-SAME: live_in=[%0, %arg1]
// CHECK-SAME: live_out=[%1, %arg1]
// CHECK: op: xt.mul
// CHECK-SAME: live_in=[%1, %arg1]
// CHECK-SAME: live_out=[%2]
