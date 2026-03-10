// RUN: not xt-opt %s 2>&1 | FileCheck %s --check-prefix=ERR

func.func @bad_tile_shape(%arg0: memref<2048x16xf32>) {
  %zero = arith.constant 0 : i32
  %0 = xt.load(%arg0, %zero, %zero) {tile = [8, 16]} : memref<2048x16xf32> -> tensor<16x16xf32>
  func.return
}

// ERR: tile attribute must match tensor shape

func.func @bad_element_type(%arg0: memref<2048x16xf32>) {
  %zero = arith.constant 0 : i32
  %0 = xt.load(%arg0, %zero, %zero) {tile = [16, 16]} : memref<2048x16xf32> -> tensor<16x16xi32>
  func.return
}

// ERR: requires matching memref/tensor element types
