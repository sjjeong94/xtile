// RUN: not xt-opt %s 2>&1 | FileCheck %s

module {
  func.func @invalid_store_space(%dst: memref<1024x1024xf32>) {
    x1.store %dst 3 [128, 64] [64, 64] space -1 thread 0 : memref<1024x1024xf32>
    func.return
  }
}

// CHECK: space attribute must be non-negative
