// RUN: not xt-opt --split-input-file %s 2>&1 | FileCheck %s --check-prefix=ERR

func.func @bad_rank_mismatch(%arg0: memref<2048xf32>) {
  %zero = arith.constant 0 : i32
  %0 = xt.load(%arg0, %zero, %zero) : (memref<2048xf32>, i32, i32) -> tensor<16xf32>
  func.return
}

// ERR: coordinate count must match tensor rank

// -----

func.func @bad_element_type(%arg0: memref<2048x16xf32>) {
  %zero = arith.constant 0 : i32
  %0 = xt.load(%arg0, %zero, %zero) : (memref<2048x16xf32>, i32, i32) -> tensor<16x16xi32>
  func.return
}

// ERR: requires matching memref/tensor element types

// -----

func.func @bad_matmul_shape(%a: tensor<16x32xf32>, %b: tensor<16x8xf32>) {
  %0 = xt.matmul(%a, %b) : tensor<16x32xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
  func.return
}

// ERR: matmul requires lhs inner dimension to match rhs outer dimension

// -----

func.func @bad_mma_input_type(%a: tensor<16x32xf32>, %b: tensor<32x8xi8>, %acc: tensor<16x8xf32>) {
  %0 = xt.mma(%a, %b, %acc) : tensor<16x32xf32>, tensor<32x8xi8>, tensor<16x8xf32> -> tensor<16x8xf32>
  func.return
}

// ERR: mma requires i8 input tensors

// -----

func.func @bad_mma_acc_type(%a: tensor<16x32xi8>, %b: tensor<32x8xi8>, %acc: tensor<16x8xi32>) {
  %0 = xt.mma(%a, %b, %acc) : tensor<16x32xi8>, tensor<32x8xi8>, tensor<16x8xi32> -> tensor<16x8xi32>
  func.return
}

// ERR: mma requires f32 or bf16 accumulator and result tensors

// -----

func.func @bad_shared_hint(%arg0: memref<256x512xi8>) {
  %zero = arith.constant 0 : i32
  %0 = xt.load(%arg0, %zero, %zero) {shared = 3 : i64} : (memref<256x512xi8>, i32, i32) -> tensor<256x64xi8>
  func.return
}

// ERR: shared attribute must be 0, 1, or 2

// -----

func.func @bad_broadcast_shape(%arg0: tensor<16x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<16x16xf32> {
  %0 = xt.add(%arg0, %arg1) : tensor<16x16xf32>, tensor<8x16xf32> -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}

// ERR: operands are not broadcast-compatible with result tensor type

// -----

func.func @bad_load_conv2d_shape(%arg0: memref<1x34x66x128xi8>, %arg1: tensor<3x3x64x64xi8>) -> tensor<1x32x64x32xf32> {
  %c0 = arith.constant 0 : i32
  %0 = xt.load_conv2d(%arg0, %arg1, %c0, %c0, %c0, %c0) {dilation = array<i64: 1, 1>, group = 1 : i64, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (memref<1x34x66x128xi8>, tensor<3x3x64x64xi8>, i32, i32, i32, i32) -> tensor<1x32x64x32xf32>
  func.return %0 : tensor<1x32x64x32xf32>
}

// ERR: load_conv2d requires source and filter channel dimensions to match

// -----

func.func @bad_load_conv2d_group(%arg0: memref<1x34x66x128xi8>, %arg1: tensor<3x3x128x64xi8>) -> tensor<1x32x64x32xf32> {
  %c0 = arith.constant 0 : i32
  %0 = xt.load_conv2d(%arg0, %arg1, %c0, %c0, %c0, %c0) {dilation = array<i64: 1, 1>, group = 0 : i64, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (memref<1x34x66x128xi8>, tensor<3x3x128x64xi8>, i32, i32, i32, i32) -> tensor<1x32x64x32xf32>
  func.return %0 : tensor<1x32x64x32xf32>
}

// ERR: group attribute must be positive

// -----

func.func @bad_reduce_shape(%arg0: tensor<16x16xf32>) -> tensor<8x1xf32> {
  %0 = xt.reduce_sum(%arg0) {axis = 1 : i64} : tensor<16x16xf32> -> tensor<8x1xf32>
  func.return %0 : tensor<8x1xf32>
}

// ERR: reduce result shape must match input shape except for the reduced dimension, which must be 1

// -----

func.func @bad_reduce_axis(%arg0: tensor<16x16xf32>) -> tensor<16x1xf32> {
  %0 = xt.reduce_sum(%arg0) {axis = 2 : i64} : tensor<16x16xf32> -> tensor<16x1xf32>
  func.return %0 : tensor<16x1xf32>
}

// ERR: axis must be in range [-rank, rank)

// -----

func.func @bad_reshape_shape(%arg0: tensor<64x16xf32>) -> tensor<2x16x16xf32> {
  %0 = xt.reshape(%arg0) : tensor<64x16xf32> -> tensor<2x16x16xf32>
  func.return %0 : tensor<2x16x16xf32>
}

// ERR: reshape requires operand and result to have the same number of elements

// -----

func.func @bad_transpose_shape(%arg0: tensor<2x32x16xf32>) -> tensor<2x32x16xf32> {
  %0 = xt.transpose(%arg0) : tensor<2x32x16xf32> -> tensor<2x32x16xf32>
  func.return %0 : tensor<2x32x16xf32>
}

// ERR: transpose result shape must preserve dim 0 and swap dims 1 and 2

// -----

func.func @bad_permute_shape(%arg0: tensor<2x3x5xf32>) -> tensor<2x5x3xf32> {
  %0 = xt.permute(%arg0) {permutation = [2, 0, 1]} : tensor<2x3x5xf32> -> tensor<2x5x3xf32>
  func.return %0 : tensor<2x5x3xf32>
}

// ERR: permute result shape must match the input shape reordered by permutation

// -----

func.func @bad_permute_attr(%arg0: tensor<2x3x5xf32>) -> tensor<5x2x3xf32> {
  %0 = xt.permute(%arg0) {permutation = [2, 0, 0]} : tensor<2x3x5xf32> -> tensor<5x2x3xf32>
  func.return %0 : tensor<5x2x3xf32>
}

// ERR: permutation attribute must contain each dimension exactly once

// -----

func.func @bad_grid_rank() attributes {xt.grid = array<i32: 32, 8>} {
  func.return
}

// ERR: xt.grid must have exactly 3 entries

// -----

module attributes {xt.grid = array<i32: 1, 1, 1>} {
}

// ERR: xt.grid is only valid on func.func operations

// -----

func.func @bad_itof_types(%arg0: tensor<5x16xf32>) {
  %0 = xt.itof(%arg0) : tensor<5x16xf32> -> tensor<5x16xf32>
  func.return
}

// ERR: 'xt.itof' op requires integer input and floating-point result element types

// -----

func.func @bad_ftoi_types(%arg0: tensor<5x16xi8>) {
  %0 = xt.ftoi(%arg0) : tensor<5x16xi8> -> tensor<5x16xi8>
  func.return
}

// ERR: 'xt.ftoi' op requires floating-point input and integer result element types

// -----

func.func @removed_free_op(%arg0: tensor<16x16xf32>) {
  xt.free(%arg0) : tensor<16x16xf32>
  func.return
}

// ERR: custom op 'xt.free' is unknown
