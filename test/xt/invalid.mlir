// RUN: not xt-opt %s 2>&1 | FileCheck %s --check-prefix=ERR

func.func @bad_rank_mismatch(%arg0: memref<2048xf32>) {
  %zero = arith.constant 0 : i32
  %0 = xt.load(%arg0, %zero, %zero) {tile = [16]} : memref<2048xf32> -> tensor<16xf32>
  func.return
}

// ERR: coordinate count must match tile rank

func.func @bad_tile_shape(%arg0: memref<2048x16xf32>) {
  %zero = arith.constant 0 : i32
  %0 = xt.load(%arg0, %zero, %zero) {tile = [8, 16]} : memref<2048x16xf32> -> tensor<16x16xf32>
  func.return
}

// ERR: tile attribute must match tensor shape

func.func @bad_element_type(%arg0: memref<2048x16xf32>) {
  %zero = arith.constant 0 : i32
  %0 = xt.load(%arg0, %zero, %zero) {tile = [16, 16]} : memref<2048x16xf32> -> tensor<16x16xi32>
  func.return
}

// ERR: requires matching memref/tensor element types

func.func @bad_3d_shape(%arg0: memref<2048x32x32xf32>) {
  %zero = arith.constant 0 : i32
  %0 = xt.load(%arg0, %zero, %zero, %zero) {tile = [16, 16, 8]} : memref<2048x32x32xf32> -> tensor<16x16x16xf32>
  func.return
}

// ERR: tile attribute must match tensor shape

func.func @bad_matmul_shape(%a: tensor<16x32xf32>, %b: tensor<16x8xf32>) {
  %0 = xt.matmul(%a, %b) : (tensor<16x32xf32>, tensor<16x8xf32>) -> tensor<16x8xf32>
  func.return
}

// ERR: matmul requires lhs inner dimension to match rhs outer dimension

func.func @bad_mma_input_type(%a: tensor<16x32xf32>, %b: tensor<32x8xi8>, %acc: tensor<16x8xf32>) {
  %0 = xt.mma(%a, %b, %acc) : (tensor<16x32xf32>, tensor<32x8xi8>, tensor<16x8xf32>) -> tensor<16x8xf32>
  func.return
}

// ERR: mma requires i8 input tensors

func.func @bad_mma_acc_type(%a: tensor<16x32xi8>, %b: tensor<32x8xi8>, %acc: tensor<16x8xi32>) {
  %0 = xt.mma(%a, %b, %acc) : (tensor<16x32xi8>, tensor<32x8xi8>, tensor<16x8xi32>) -> tensor<16x8xi32>
  func.return
}

// ERR: mma requires f32 or bf16 accumulator and result tensors

func.func @bad_shared_hint(%arg0: memref<256x512xi8>) {
  %zero = arith.constant 0 : i32
  %0 = xt.load(%arg0, %zero, %zero) {tile = [256, 64], shared = 2} : memref<256x512xi8> -> tensor<256x64xi8>
  func.return
}

// ERR: shared attribute must be 0 or 1
