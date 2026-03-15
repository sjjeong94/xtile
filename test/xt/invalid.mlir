// RUN: not xt-opt --split-input-file %s 2>&1 | FileCheck %s --check-prefix=ERR

func.func @bad_rank_mismatch(%arg0: memref<2048xf32>) {
  %zero = arith.constant 0 : i32
  %0 = "xt.load"(%arg0, %zero, %zero) : (memref<2048xf32>, i32, i32) -> tensor<16xf32>
  func.return
}

// ERR: coordinate count must match tensor rank

// -----

func.func @bad_element_type(%arg0: memref<2048x16xf32>) {
  %zero = arith.constant 0 : i32
  %0 = "xt.load"(%arg0, %zero, %zero) : (memref<2048x16xf32>, i32, i32) -> tensor<16x16xi32>
  func.return
}

// ERR: requires matching memref/tensor element types

// -----

func.func @bad_matmul_shape(%a: tensor<16x32xf32>, %b: tensor<16x8xf32>) {
  %0 = "xt.matmul"(%a, %b) : (tensor<16x32xf32>, tensor<16x8xf32>) -> tensor<16x8xf32>
  func.return
}

// ERR: matmul requires lhs inner dimension to match rhs outer dimension

// -----

func.func @bad_mma_input_type(%a: tensor<16x32xf32>, %b: tensor<32x8xi8>, %acc: tensor<16x8xf32>) {
  %0 = "xt.mma"(%a, %b, %acc) : (tensor<16x32xf32>, tensor<32x8xi8>, tensor<16x8xf32>) -> tensor<16x8xf32>
  func.return
}

// ERR: mma requires i8 input tensors

// -----

func.func @bad_mma_acc_type(%a: tensor<16x32xi8>, %b: tensor<32x8xi8>, %acc: tensor<16x8xi32>) {
  %0 = "xt.mma"(%a, %b, %acc) : (tensor<16x32xi8>, tensor<32x8xi8>, tensor<16x8xi32>) -> tensor<16x8xi32>
  func.return
}

// ERR: mma requires f32 or bf16 accumulator and result tensors

// -----

func.func @bad_shared_hint(%arg0: memref<256x512xi8>) {
  %zero = arith.constant 0 : i32
  %0 = "xt.load"(%arg0, %zero, %zero) <{shared = 2 : i64}> : (memref<256x512xi8>, i32, i32) -> tensor<256x64xi8>
  func.return
}

// ERR: shared attribute must be 0 or 1

// -----

func.func @bad_broadcast_shape(%arg0: tensor<16x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<16x16xf32> {
  %0 = "xt.add"(%arg0, %arg1) : (tensor<16x16xf32>, tensor<8x16xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}

// ERR: operands are not broadcast-compatible with result tensor type

// -----

func.func @bad_conv2d_pad_attr(%arg0: tensor<1x32x64x128xi8>, %arg1: tensor<3x3x128x64xi8>) -> tensor<1x32x64x64xf32> {
  %0 = "xt.conv2d"(%arg0, %arg1) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x32x64x128xi8>, tensor<3x3x128x64xi8>) -> tensor<1x32x64x64xf32>
  func.return %0 : tensor<1x32x64x64xf32>
}

// ERR: pad attribute must have exactly 4 entries

// -----

func.func @bad_conv2d_shape(%arg0: tensor<1x32x64x128xi8>, %arg1: tensor<3x3x64x64xi8>) -> tensor<1x32x64x64xf32> {
  %0 = "xt.conv2d"(%arg0, %arg1) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x32x64x128xi8>, tensor<3x3x64x64xi8>) -> tensor<1x32x64x64xf32>
  func.return %0 : tensor<1x32x64x64xf32>
}

// ERR: conv2d requires input and filter channel dimensions to match

// -----

func.func @bad_depthwise_filter_shape(%arg0: tensor<1x32x64x64xi8>, %arg1: tensor<3x3x2x64xi8>) -> tensor<1x32x64x64xf32> {
  %0 = "xt.depthwise_conv2d"(%arg0, %arg1) <{dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x32x64x64xi8>, tensor<3x3x2x64xi8>) -> tensor<1x32x64x64xf32>
  func.return %0 : tensor<1x32x64x64xf32>
}

// ERR: depthwise_conv2d requires filter input-channel dimension to be 1

// -----

func.func @bad_reduce_shape(%arg0: tensor<16x16xf32>) -> tensor<8x1xf32> {
  %0 = "xt.reduce_sum"(%arg0) : (tensor<16x16xf32>) -> tensor<8x1xf32>
  func.return %0 : tensor<8x1xf32>
}

// ERR: reduce result shape must match input shape except for the last dimension, which must be 1

// -----

func.func @bad_reshape_shape(%arg0: tensor<64x16xf32>) -> tensor<2x16x16xf32> {
  %0 = "xt.reshape"(%arg0) : (tensor<64x16xf32>) -> tensor<2x16x16xf32>
  func.return %0 : tensor<2x16x16xf32>
}

// ERR: reshape requires operand and result to have the same number of elements

// -----

func.func @bad_transpose_shape(%arg0: tensor<2x32x16xf32>) -> tensor<2x32x16xf32> {
  %0 = "xt.transpose"(%arg0) : (tensor<2x32x16xf32>) -> tensor<2x32x16xf32>
  func.return %0 : tensor<2x32x16xf32>
}

// ERR: transpose result shape must preserve dim 0 and swap dims 1 and 2

// -----

func.func @bad_grid_rank() attributes {xt.grid = array<i32: 32, 8>} {
  func.return
}

// ERR: xt.grid must have exactly 3 entries

// -----

module attributes {xt.grid = array<i32: 1, 1, 1>} {
}

// ERR: xt.grid is only valid on func.func operations
