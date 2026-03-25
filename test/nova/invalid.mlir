// RUN: not xt-opt --split-input-file %s 2>&1 | FileCheck %s --check-prefix=ERR

func.func @removed_free_op(%arg0: tensor<16x16xf32>) {
  nova.free(%arg0) : tensor<16x16xf32>
  func.return
}

// ERR: custom op 'nova.free' is unknown

// -----

func.func @bad_permute_shape(%arg0: tensor<2x3x5xf32>) -> tensor<2x5x3xf32> {
  %0 = nova.permute %arg0 [2, 0, 1] : tensor<2x3x5xf32> -> tensor<2x5x3xf32>
  func.return %0 : tensor<2x5x3xf32>
}

// ERR: permute result shape must match the input shape reordered by permutation

// -----

func.func @bad_permute_attr(%arg0: tensor<2x3x5xf32>) -> tensor<5x2x3xf32> {
  %0 = nova.permute %arg0 [2, 0, 0] : tensor<2x3x5xf32> -> tensor<5x2x3xf32>
  func.return %0 : tensor<5x2x3xf32>
}

// ERR: permutation attribute must contain each dimension exactly once

// -----

func.func @bad_conv2d_shape(%arg0: tensor<1x32x64x128xi8>, %arg1: tensor<3x3x64x64xi8>) -> tensor<1x32x64x64xf32> {
  %0 = nova.conv2d %arg0, %arg1 {dilation = array<i64: 1, 1>, group = 1 : i64, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : tensor<1x32x64x128xi8>, tensor<3x3x64x64xi8> -> tensor<1x32x64x64xf32>
  func.return %0 : tensor<1x32x64x64xf32>
}

// ERR: conv2d requires input and filter channel dimensions to match

// -----

func.func @bad_conv2d_group(%arg0: tensor<1x32x64x128xi8>, %arg1: tensor<3x3x128x64xi8>) -> tensor<1x32x64x64xf32> {
  %0 = nova.conv2d %arg0, %arg1 {dilation = array<i64: 1, 1>, group = 0 : i64, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : tensor<1x32x64x128xi8>, tensor<3x3x128x64xi8> -> tensor<1x32x64x64xf32>
  func.return %0 : tensor<1x32x64x64xf32>
}

// ERR: group attribute must be positive

// -----
