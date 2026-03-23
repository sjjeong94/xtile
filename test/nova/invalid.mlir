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
