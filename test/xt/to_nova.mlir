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

  func.func @square_mul(%a: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %0 = "xt.mul"(%a, %a) : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
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

  func.func @load_store_constant_index(%src: memref<128x16xf32>, %dst: memref<128x16xf32>) {
    %c0 = arith.constant 0 : i32
    %c2 = arith.constant 2 : i32
    %c1 = arith.constant 1 : i32
    %0 = xt.load(%src, %c0, %c2) {shared = 1 : i64} : (memref<128x16xf32>, i32, i32) -> tensor<16x16xf32>
    xt.store(%0, %dst, %c1, %c2) : (tensor<16x16xf32>, memref<128x16xf32>, i32, i32) -> ()
    return
  }

  func.func @load_store_dynamic_index(%src: memref<128x16xf32>, %dst: memref<128x16xf32>, %i: i32) {
    %c2 = arith.constant 2 : i32
    %0 = xt.load(%src, %i, %c2) : (memref<128x16xf32>, i32, i32) -> tensor<16x16xf32>
    xt.store(%0, %dst, %i, %c2) : (tensor<16x16xf32>, memref<128x16xf32>, i32, i32) -> ()
    return
  }

  func.func @free_op(%arg0: tensor<16x16xf32>) {
    xt.free(%arg0) : tensor<16x16xf32>
    return
  }
}

// CHECK-LABEL: func.func @reduce_ops
// CHECK: nova.reduce(%arg0) {mode = 0 : i32} : tensor<16x16xf32> -> tensor<16x1xf32>
// CHECK: nova.reduce(%arg0) {mode = 1 : i32} : tensor<16x16xf32> -> tensor<16x1xf32>
// CHECK-LABEL: func.func @broadcast_sub
// CHECK: nova.broadcast(%arg0, %arg1) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 3 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<16x16xf32>, tensor<16x1xf32> -> tensor<16x16xf32>
// CHECK-LABEL: func.func @elementwise_mul
// CHECK: nova.elementwise(%arg0, %arg1) {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 2 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-LABEL: func.func @square_mul
// CHECK: nova.square(%arg0) : tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-LABEL: func.func @scalar_like_mul
// CHECK: xt.mul(%arg0, %arg1) : tensor<16x1xf32>, tensor<1x1xf32> -> tensor<16x1xf32>
// CHECK-LABEL: func.func @constant_scalar_mul
// CHECK: nova.scalar(%arg0) {mode = 2 : i32, rhs = 1.250000e-01 : f32} : tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-LABEL: func.func @constant_scalar_sub
// CHECK: nova.scalar(%arg0) {mode = 1 : i32, rhs = -1.250000e-01 : f32} : tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-LABEL: func.func @matmul
// CHECK: %[[SCALE:.*]] = arith.constant dense<1.000000e+00> : tensor<1x1xf32>
// CHECK: %[[BIAS:.*]] = arith.constant dense<0.000000e+00> : tensor<1x1xf32>
// CHECK: nova.matmul(%arg0, %arg1, %[[SCALE]], %[[BIAS]]) : tensor<16x32xf32>, tensor<32x8xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xf32>
// CHECK-LABEL: func.func @load_store_constant_index
// CHECK: %[[LOAD:.*]] = nova.load(%arg0) {index = array<i64: 0, 2>, shared = 1 : i64} : memref<128x16xf32> -> tensor<16x16xf32>
// CHECK: nova.store(%[[LOAD]], %arg1) {index = array<i64: 1, 2>} : (tensor<16x16xf32>, memref<128x16xf32>) -> ()
// CHECK-LABEL: func.func @load_store_dynamic_index
// CHECK: %[[DYNLOAD:.*]] = xt.load(%arg0, %arg2, %{{.*}}) : (memref<128x16xf32>, i32, i32) -> tensor<16x16xf32>
// CHECK: xt.store(%[[DYNLOAD]], %arg1, %arg2, %{{.*}}) : (tensor<16x16xf32>, memref<128x16xf32>, i32, i32) -> ()
// CHECK-LABEL: func.func @free_op
// CHECK: nova.free(%arg0) : tensor<16x16xf32>
