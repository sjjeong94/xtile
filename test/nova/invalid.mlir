// RUN: not xt-opt --split-input-file %s 2>&1 | FileCheck %s --check-prefix=ERR

func.func @removed_free_op(%arg0: tensor<16x16xf32>) {
  nova.free(%arg0) : tensor<16x16xf32>
  func.return
}

// ERR: custom op 'nova.free' is unknown
