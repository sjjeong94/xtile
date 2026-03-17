// RUN: not xt-opt --xt-serialize --split-input-file %s 2>&1 | FileCheck %s --check-prefix=ERR

func.func @missing_grid(%src: memref<128x16xf32>, %dst: memref<128x16xf32>) {
  %bid:3 = xt.get_tile_block_id : i32, i32, i32
  %0 = xt.load(%src, %bid#0, %bid#1) : (memref<128x16xf32>, i32, i32) -> tensor<16x16xf32>
  xt.store(%0, %dst, %bid#0, %bid#1) : (tensor<16x16xf32>, memref<128x16xf32>, i32, i32) -> ()
  func.return
}

// ERR: xt-serialize requires func.func xt.grid attribute

// -----

func.func @non_void(%src: memref<128x16xf32>) -> i32 attributes {xt.grid = array<i32: 1, 1, 1>} {
  %bid:3 = xt.get_tile_block_id : i32, i32, i32
  func.return %bid#0 : i32
}

// ERR: xt-serialize only supports void functions

// -----

func.func @multi_block(%flag: i1) attributes {xt.grid = array<i32: 1, 1, 1>} {
  cf.cond_br %flag, ^bb1, ^bb2
^bb1:
  func.return
^bb2:
  func.return
}

// ERR: xt-serialize only supports single-block functions
