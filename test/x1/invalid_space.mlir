// RUN: not xt-opt %s 2>&1 | FileCheck %s

module {
  func.func @invalid_store_space(%dst: memref<1024x1024xf32>) {
    x1.store(%dst) {bank = 3 : i64, space = -1 : i64, thread = 0 : i64, start = [128, 64], shape = [64, 64]} : memref<1024x1024xf32>
    func.return
  }
}

// CHECK: space attribute must be non-negative
