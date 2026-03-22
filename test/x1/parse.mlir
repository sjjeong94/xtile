// RUN: xt-opt %s | FileCheck %s

module {
  func.func @parse_x1(%src: memref<1024x1024xf32>, %dst: memref<1024x1024xf32>) {
    x1.barrier {mode = 0}
    x1.load %src {bank = 3, space = 4, thread = 0, start = [0, 64], shape = [64, 64]} : memref<1024x1024xf32>
    x1.store %dst {bank = 3, space = 4, thread = 0, start = [128, 64], shape = [64, 64]} : memref<1024x1024xf32>
    x1.matmul {lhs0 = 0, lhs1 = 1, rhs0 = 2, rhs1 = 3, out0 = 4, out1 = 5, m = 32, n = 128, k = 64}
    x1.reduce {inp0 = 0, inp1 = 1, out0 = 2, out1 = 3, shape = [32, 64], axis = 1, mode = 1}
    x1.broadcast {lhs0 = 0, lhs1 = 1, rhs0 = 2, rhs1 = 3, out0 = 4, out1 = 5, shape = [32, 64], axis = 1, lhs_a = 1.000000e+00, lhs_b = 0.000000e+00, rhs_a = 1.000000e+00, rhs_b = 0.000000e+00, mode = 3}
    x1.square {inp0 = 4, inp1 = 5, out0 = 6, out1 = 7, shape = [32, 64]}
    x1.exp {inp0 = 4, inp1 = 5, out0 = 2, out1 = 3, shape = [32, 64]}
    x1.scalar_fma {inp0 = 2, inp1 = 3, out0 = 4, out1 = 5, shape = [32, 1], a = 1.562500e-02, b = 9.99999974E-6}
    x1.rsqrt {inp0 = 4, inp1 = 5, out0 = 2, out1 = 3, shape = [32, 1]}
    x1.reciprocal {inp0 = 2, inp1 = 3, out0 = 4, out1 = 5, shape = [64, 1]}
    func.return
  }
}

// CHECK-LABEL: func.func @parse_x1
// CHECK: x1.barrier {mode = 0}
// CHECK: x1.load %arg0 {bank = 3, space = 4, thread = 0, start = [0, 64], shape = [64, 64]} : memref<1024x1024xf32>
// CHECK: x1.store %arg1 {bank = 3, space = 4, thread = 0, start = [128, 64], shape = [64, 64]} : memref<1024x1024xf32>
// CHECK: x1.matmul {lhs0 = 0, lhs1 = 1, rhs0 = 2, rhs1 = 3, out0 = 4, out1 = 5, m = 32, n = 128, k = 64}
// CHECK: x1.reduce {inp0 = 0, inp1 = 1, out0 = 2, out1 = 3, shape = [32, 64], axis = 1, mode = 1}
// CHECK: x1.broadcast {lhs0 = 0, lhs1 = 1, rhs0 = 2, rhs1 = 3, out0 = 4, out1 = 5, shape = [32, 64], axis = 1, lhs_a = 1.000000e+00, lhs_b = 0.000000e+00, rhs_a = 1.000000e+00, rhs_b = 0.000000e+00, mode = 3}
// CHECK: x1.square {inp0 = 4, inp1 = 5, out0 = 6, out1 = 7, shape = [32, 64]}
// CHECK: x1.exp {inp0 = 4, inp1 = 5, out0 = 2, out1 = 3, shape = [32, 64]}
// CHECK: x1.scalar_fma {inp0 = 2, inp1 = 3, out0 = 4, out1 = 5, shape = [32, 1], a = 1.562500e-02, b = 9.99999974E-6}
// CHECK: x1.rsqrt {inp0 = 4, inp1 = 5, out0 = 2, out1 = 3, shape = [32, 1]}
// CHECK: x1.reciprocal {inp0 = 2, inp1 = 3, out0 = 4, out1 = 5, shape = [64, 1]}
