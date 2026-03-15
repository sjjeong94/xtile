// RUN: xt-opt %s | FileCheck %s

func.func @parse_square(%arg0: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %0 = nova.square(%arg0) : tensor<16x16xf32> -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}

// CHECK-LABEL: func.func @parse_square
// CHECK: %[[RES:.*]] = nova.square(%arg0) : tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK: return %[[RES]] : tensor<16x16xf32>
