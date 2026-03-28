// RUN: not xt-opt --split-input-file %s 2>&1 | FileCheck %s --check-prefix=ERR

module {
  func.func @invalid_load_bank(%src: memref<1024x1024xf32>) {
    x1.load %src -1 [0, 64] [64, 64] space 4 thread 0 : memref<1024x1024xf32>
    func.return
  }
}

// ERR: bank attribute must be non-negative

// -----

module {
  func.func @invalid_conv2d_group() {
    x1.conv2d inp 0 filter 1 out 2 input [1, 32, 64, 128] kernel [3, 3, 128, 64] result [1, 32, 64, 64] group 0 pad [1, 1, 1, 1] stride [1, 1] dilation [1, 1]
    func.return
  }
}

// ERR: group attribute must be positive
