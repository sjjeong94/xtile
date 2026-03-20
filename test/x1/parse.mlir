// RUN: xt-opt %s | FileCheck %s

module {
  func.func @parse_x1(%src: memref<1024x1024xf32>, %dst: memref<1024x1024xf32>) {
    x1.barrier() {mode = 0 : i32}
    x1.load(%src) {bank = 3 : i64, space = 4 : i64, thread = 0 : i64, start = [0, 64], shape = [64, 64]} : memref<1024x1024xf32>
    x1.store(%dst) {bank = 3 : i64, space = 4 : i64, thread = 0 : i64, start = [128, 64], shape = [64, 64]} : memref<1024x1024xf32>
    x1.matmul() {lhs0_bank = 0 : i64, lhs1_bank = 1 : i64, rhs0_bank = 2 : i64, rhs1_bank = 3 : i64, res0_bank = 4 : i64, res1_bank = 5 : i64, m = 32 : i64, n = 128 : i64, k = 64 : i64}
    x1.reduce() {inp0 = 0 : i64, inp1 = 1 : i64, res0 = 2 : i64, res1 = 3 : i64, m = 32 : i64, n = 64 : i64, mode = 1 : i32}
    x1.broadcast() {lhs0 = 0 : i64, lhs1 = 1 : i64, rhs0 = 2 : i64, rhs1 = 3 : i64, res0 = 4 : i64, res1 = 5 : i64, lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 3 : i32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32}
    x1.exp() {inp0 = 4 : i64, inp1 = 5 : i64, res0 = 2 : i64, res1 = 3 : i64, m = 32 : i64, n = 64 : i64}
    x1.reciprocal() {inp0 = 2 : i64, inp1 = 3 : i64, res0 = 4 : i64, res1 = 5 : i64, m = 64 : i64, n = 1 : i64}
    func.return
  }
}

// CHECK-LABEL: func.func @parse_x1
// CHECK: x1.barrier() {mode = 0 : i32}
// CHECK: x1.load(%arg0) {bank = 3 : i64, shape = [64, 64], space = 4 : i64, start = [0, 64], thread = 0 : i64} : memref<1024x1024xf32>
// CHECK: x1.store(%arg1) {bank = 3 : i64, shape = [64, 64], space = 4 : i64, start = [128, 64], thread = 0 : i64} : memref<1024x1024xf32>
// CHECK: x1.matmul() {k = 64 : i64, lhs0_bank = 0 : i64, lhs1_bank = 1 : i64, m = 32 : i64, n = 128 : i64, res0_bank = 4 : i64, res1_bank = 5 : i64, rhs0_bank = 2 : i64, rhs1_bank = 3 : i64}
// CHECK: x1.reduce() {inp0 = 0 : i64, inp1 = 1 : i64, m = 32 : i64, mode = 1 : i32, n = 64 : i64, res0 = 2 : i64, res1 = 3 : i64}
// CHECK: x1.broadcast() {lhs0 = 0 : i64, lhs1 = 1 : i64, lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, mode = 3 : i32, res0 = 4 : i64, res1 = 5 : i64, rhs0 = 2 : i64, rhs1 = 3 : i64, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32}
// CHECK: x1.exp() {inp0 = 4 : i64, inp1 = 5 : i64, m = 32 : i64, n = 64 : i64, res0 = 2 : i64, res1 = 3 : i64}
// CHECK: x1.reciprocal() {inp0 = 2 : i64, inp1 = 3 : i64, m = 64 : i64, n = 1 : i64, res0 = 4 : i64, res1 = 5 : i64}
