// RUN: not xt-opt %s 2>&1 | FileCheck %s

module {
  func.func @invalid_load_bank(%src: memref<1024x1024xf32>) {
    x1.load %src -1 [0, 64] [64, 64] space 4 thread 0 : memref<1024x1024xf32>
    func.return
  }
}

// CHECK: bank attribute must be non-negative
