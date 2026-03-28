// RUN: xt-opt --xt-to-nova %s | FileCheck %s

module {
  func.func @reduce_ops(%a: tensor<16x16xf32>) -> (tensor<16x1xf32>, tensor<16x1xf32>) {
    %0 = xt.reduce_sum(%a) {axis = 1 : i64} : tensor<16x16xf32> -> tensor<16x1xf32>
    %1 = xt.reduce_max(%a) {axis = 1 : i64} : tensor<16x16xf32> -> tensor<16x1xf32>
    return %0, %1 : tensor<16x1xf32>, tensor<16x1xf32>
  }

  func.func @reduce_axis0(%a: tensor<16x16xf32>) -> tensor<1x16xf32> {
    %0 = xt.reduce_sum(%a) {axis = 0 : i64} : tensor<16x16xf32> -> tensor<1x16xf32>
    return %0 : tensor<1x16xf32>
  }

  func.func @broadcast_sub(%a: tensor<16x16xf32>, %b: tensor<16x1xf32>) -> tensor<16x16xf32> {
    %0 = xt.sub(%a, %b) : tensor<16x16xf32>, tensor<16x1xf32> -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }

  func.func @elementwise_mul(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %0 = xt.mul(%a, %b) : tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }

  func.func @square_mul(%a: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %0 = xt.mul(%a, %a) : tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }

  func.func @rsqrt(%a: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %0 = xt.rsqrt(%a) : tensor<16x16xf32> -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }

  func.func @exp(%a: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %0 = xt.exp(%a) : tensor<16x16xf32> -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }

  func.func @reciprocal(%a: tensor<16x1xf32>) -> tensor<16x1xf32> {
    %0 = xt.reciprocal(%a) : tensor<16x1xf32> -> tensor<16x1xf32>
    return %0 : tensor<16x1xf32>
  }

  func.func @scalar_like_mul(%a: tensor<16x1xf32>, %b: tensor<1x1xf32>) -> tensor<16x1xf32> {
    %0 = xt.mul(%a, %b) : tensor<16x1xf32>, tensor<1x1xf32> -> tensor<16x1xf32>
    return %0 : tensor<16x1xf32>
  }

  func.func @constant_scalar_mul(%a: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %cst = arith.constant dense<1.250000e-01> : tensor<1x1xf32>
    %0 = xt.mul(%a, %cst) : tensor<16x16xf32>, tensor<1x1xf32> -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }

  func.func @constant_scalar_sub(%a: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %cst = arith.constant dense<1.250000e-01> : tensor<1x1xf32>
    %0 = xt.sub(%a, %cst) : tensor<16x16xf32>, tensor<1x1xf32> -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }

  func.func @matmul(%a: tensor<16x32xf32>, %b: tensor<32x8xf32>) -> tensor<16x8xf32> {
    %0 = xt.matmul(%a, %b) : tensor<16x32xf32>, tensor<32x8xf32> -> tensor<16x8xf32>
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

  func.func @load_conv2d_interior(%src: memref<1x96x192x128xi8>, %filter: tensor<3x3x128x64xi8>) -> tensor<1x32x64x64xf32> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %0 = xt.load_conv2d(%src, %filter, %c0, %c1, %c1, %c0) {dilation = array<i64: 1, 1>, group = 1 : i64, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (memref<1x96x192x128xi8>, tensor<3x3x128x64xi8>, i32, i32, i32, i32) -> tensor<1x32x64x64xf32>
    return %0 : tensor<1x32x64x64xf32>
  }

  func.func @load_conv2d_boundary(%src: memref<1x96x192x128xi8>, %filter: tensor<3x3x128x64xi8>) -> tensor<1x32x64x64xf32> {
    %c0 = arith.constant 0 : i32
    %0 = xt.load_conv2d(%src, %filter, %c0, %c0, %c0, %c0) {dilation = array<i64: 1, 1>, group = 1 : i64, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (memref<1x96x192x128xi8>, tensor<3x3x128x64xi8>, i32, i32, i32, i32) -> tensor<1x32x64x64xf32>
    return %0 : tensor<1x32x64x64xf32>
  }

  func.func @load_conv2d_bottom_right_boundary(%src: memref<1x96x192x128xi8>, %filter: tensor<3x3x128x64xi8>) -> tensor<1x32x64x64xf32> {
    %c0 = arith.constant 0 : i32
    %c2 = arith.constant 2 : i32
    %0 = xt.load_conv2d(%src, %filter, %c0, %c2, %c2, %c0) {dilation = array<i64: 1, 1>, group = 1 : i64, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (memref<1x96x192x128xi8>, tensor<3x3x128x64xi8>, i32, i32, i32, i32) -> tensor<1x32x64x64xf32>
    return %0 : tensor<1x32x64x64xf32>
  }

  func.func @cast_ops(%a: tensor<5x16xi8>, %b: tensor<5x16xf32>) -> (tensor<5x16xf32>, tensor<5x16xi8>) {
    %0 = xt.itof(%a) : tensor<5x16xi8> -> tensor<5x16xf32>
    %1 = xt.ftoi(%b) : tensor<5x16xf32> -> tensor<5x16xi8>
    return %0, %1 : tensor<5x16xf32>, tensor<5x16xi8>
  }

  func.func @permute(%a: tensor<2x3x5xf32>) -> tensor<5x2x3xf32> {
    %0 = xt.permute(%a) {permutation = [2, 0, 1]} : tensor<2x3x5xf32> -> tensor<5x2x3xf32>
    return %0 : tensor<5x2x3xf32>
  }
}

// CHECK-LABEL: func.func @reduce_ops
// CHECK: nova.reduce 0 %arg0 {axis = 1 : i64} : tensor<16x16xf32> -> tensor<16x1xf32>
// CHECK: nova.reduce 1 %arg0 {axis = 1 : i64} : tensor<16x16xf32> -> tensor<16x1xf32>
// CHECK-LABEL: func.func @reduce_axis0
// CHECK: nova.reduce 0 %arg0 {axis = 0 : i64} : tensor<16x16xf32> -> tensor<1x16xf32>
// CHECK-LABEL: func.func @broadcast_sub
// CHECK: nova.broadcast 3 %arg0, %arg1 lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00 : tensor<16x16xf32>, tensor<16x1xf32> -> tensor<16x16xf32>
// CHECK-LABEL: func.func @elementwise_mul
// CHECK: nova.elementwise 2 %arg0, %arg1 lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00 : tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-LABEL: func.func @square_mul
// CHECK: nova.square %arg0 : tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-LABEL: func.func @rsqrt
// CHECK: nova.rsqrt %arg0 : tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-LABEL: func.func @exp
// CHECK: nova.exp %arg0 : tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-LABEL: func.func @reciprocal
// CHECK: nova.reciprocal %arg0 : tensor<16x1xf32> -> tensor<16x1xf32>
// CHECK-LABEL: func.func @scalar_like_mul
// CHECK: xt.mul(%arg0, %arg1) : tensor<16x1xf32>, tensor<1x1xf32> -> tensor<16x1xf32>
// CHECK-LABEL: func.func @constant_scalar_mul
// CHECK: nova.scalar 2 %arg0, 1.250000e-01 : tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-LABEL: func.func @constant_scalar_sub
// CHECK: nova.scalar 1 %arg0, -1.250000e-01 : tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-LABEL: func.func @matmul
// CHECK: %[[SCALE:.*]] = arith.constant dense<1.000000e+00> : tensor<1x1xf32>
// CHECK: %[[BIAS:.*]] = arith.constant dense<0.000000e+00> : tensor<1x1xf32>
// CHECK: nova.matmul %arg0, %arg1, %[[SCALE]], %[[BIAS]] : tensor<16x32xf32>, tensor<32x8xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xf32>
// CHECK-LABEL: func.func @load_store_constant_index
// CHECK: %[[LOAD:.*]] = nova.load %arg0 [0, 32] {shared = 1 : i64} : memref<128x16xf32> -> tensor<16x16xf32>
// CHECK: nova.store %[[LOAD]], %arg1 [16, 32] : (tensor<16x16xf32>, memref<128x16xf32>) -> ()
// CHECK-LABEL: func.func @load_store_dynamic_index
// CHECK: %[[DYNLOAD:.*]] = xt.load(%arg0, %arg2, %{{.*}}) : (memref<128x16xf32>, i32, i32) -> tensor<16x16xf32>
// CHECK: xt.store(%[[DYNLOAD]], %arg1, %arg2, %{{.*}}) : (tensor<16x16xf32>, memref<128x16xf32>, i32, i32) -> ()
// CHECK-LABEL: func.func @load_conv2d_interior
// CHECK: %[[LOADCONV0:.*]] = nova.load %arg0 [0, 31, 63, 0] : memref<1x96x192x128xi8> -> tensor<1x34x66x128xi8>
// CHECK: %[[CONV0:.*]] = nova.conv2d %[[LOADCONV0]], %arg1 group 1 pad [0, 0, 0, 0] stride [1, 1] dilation [1, 1] : tensor<1x34x66x128xi8>, tensor<3x3x128x64xi8> -> tensor<1x32x64x64xf32>
// CHECK-LABEL: func.func @load_conv2d_boundary
// CHECK: %[[LOADCONV1:.*]] = nova.load %arg0 [0, 0, 0, 0] : memref<1x96x192x128xi8> -> tensor<1x33x65x128xi8>
// CHECK: %[[CONV1:.*]] = nova.conv2d %[[LOADCONV1]], %arg1 group 1 pad [1, 1, 0, 0] stride [1, 1] dilation [1, 1] : tensor<1x33x65x128xi8>, tensor<3x3x128x64xi8> -> tensor<1x32x64x64xf32>
// CHECK-LABEL: func.func @load_conv2d_bottom_right_boundary
// CHECK: %[[LOADCONV2:.*]] = nova.load %arg0 [0, 63, 127, 0] : memref<1x96x192x128xi8> -> tensor<1x33x65x128xi8>
// CHECK: %[[CONV2:.*]] = nova.conv2d %[[LOADCONV2]], %arg1 group 1 pad [0, 0, 1, 1] stride [1, 1] dilation [1, 1] : tensor<1x33x65x128xi8>, tensor<3x3x128x64xi8> -> tensor<1x32x64x64xf32>
// CHECK-LABEL: func.func @cast_ops
// CHECK: %[[ITOF:.*]] = nova.itof %arg0 : tensor<5x16xi8> -> tensor<5x16xf32>
// CHECK: %[[FTOI:.*]] = nova.ftoi %arg1 : tensor<5x16xf32> -> tensor<5x16xi8>
// CHECK: return %[[ITOF]], %[[FTOI]] : tensor<5x16xf32>, tensor<5x16xi8>
// CHECK-LABEL: func.func @permute
// CHECK: nova.permute %arg0 [2, 0, 1] : tensor<2x3x5xf32> -> tensor<5x2x3xf32>
// CHECK: return %{{.*}} : tensor<5x2x3xf32>
