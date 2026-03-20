// RUN: not xt-opt %s 2>&1 | FileCheck %s

module {
  func.func @invalid_load_bank(%src: memref<1024x1024xf32>) {
    x1.load(%src) {bank = -1 : i64, space = 4 : i64, thread = 0 : i64, start = [0, 64], shape = [64, 64]} : memref<1024x1024xf32>
    func.return
  }
}

// CHECK: bank attribute must be non-negative
