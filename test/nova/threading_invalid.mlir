// RUN: not xt-opt --nova-threading --split-input-file %s 2>&1 | FileCheck %s --check-prefix=ERR

func.func @broadcast_threading_one_sided(%src: memref<10x8xf32>, %rhs: tensor<3x8xf32>) {
  %0 = nova.load %src [0, 0] : memref<10x8xf32> -> tensor<5x8xf32>
  %1 = nova.broadcast 0 %0, %rhs {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<5x8xf32>, tensor<3x8xf32> -> tensor<5x8xf32>
  func.return
}

// -----

func.func @elementwise_threading_mismatch(%src: memref<10x8xf32>) {
  %0 = nova.load %src [0, 0] : memref<10x8xf32> -> tensor<5x8xf32>
  %1 = nova.load %src [0, 0] : memref<10x8xf32> -> tensor<3x8xf32>
  %2 = nova.elementwise 0 %0, %1 {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<5x8xf32>, tensor<3x8xf32> -> tensor<5x8xf32>
  func.return
}

// ERR: error: 'nova.elementwise' op lhs and rhs slice metadata must match
