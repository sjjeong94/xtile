// RUN: xt-opt %s | FileCheck %s

module {
  func.func @parse_x1(%src: memref<1024x1024xf32>, %dst: memref<1024x1024xf32>) {
    x1.barrier 0
    x1.load %src 3 [0, 64] [64, 64] space 4 thread 0 : memref<1024x1024xf32>
    x1.store %dst 3 [128, 64] [64, 64] space 4 thread 0 : memref<1024x1024xf32>
    x1.matmul lhs 0 1 rhs 2 3 out 4 5 mnk 32 128 64
    x1.conv2d inp 0 filter 1 out 2 input [1, 32, 64, 128] kernel [3, 3, 128, 64] result [1, 32, 64, 64] group 1 pad [1, 1, 1, 1] stride [1, 1] dilation [1, 1]
    x1.reduce 1 inp 0 1 out 2 3 [32, 64] axis 1
    x1.broadcast 3 lhs 0 1 rhs 2 3 out 4 5 [32, 64] axis 1 lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00
    x1.square inp 4 5 out 6 7 [32, 64]
    x1.exp inp 4 5 out 2 3 [32, 64]
    x1.scalar_fma inp 2 3 out 4 5 [32, 1] {a = 1.562500e-02 : f32, b = 9.99999974E-6 : f32}
    x1.rsqrt inp 4 5 out 2 3 [32, 1]
    x1.reciprocal inp 2 3 out 4 5 [64, 1]
    func.return
  }
}

// CHECK-LABEL: func.func @parse_x1
// CHECK: x1.barrier 0
// CHECK: x1.load %arg0 3 [0, 64] [64, 64] space 4 thread 0 : memref<1024x1024xf32>
// CHECK: x1.store %arg1 3 [128, 64] [64, 64] space 4 thread 0 : memref<1024x1024xf32>
// CHECK: x1.matmul lhs 0 1 rhs 2 3 out 4 5 mnk 32 128 64
// CHECK: x1.conv2d inp 0 filter 1 out 2 input [1, 32, 64, 128] kernel [3, 3, 128, 64] result [1, 32, 64, 64] group 1 pad [1, 1, 1, 1] stride [1, 1] dilation [1, 1]
// CHECK: x1.reduce 1 inp 0 1 out 2 3 [32, 64] axis 1
// CHECK: x1.broadcast 3 lhs 0 1 rhs 2 3 out 4 5 [32, 64] axis 1 lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00
// CHECK: x1.square inp 4 5 out 6 7 [32, 64]
// CHECK: x1.exp inp 4 5 out 2 3 [32, 64]
// CHECK: x1.scalar_fma inp 2 3 out 4 5 [32, 1] {a = 1.562500e-02 : f32, b = 9.99999974E-6 : f32}
// CHECK: x1.rsqrt inp 4 5 out 2 3 [32, 1]
// CHECK: x1.reciprocal inp 2 3 out 4 5 [64, 1]
