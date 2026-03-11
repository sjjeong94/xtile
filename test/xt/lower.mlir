// RUN: xt-opt --xt-lower-to-loops %s | FileCheck %s

func.func @lower_1d(%arg0: memref<2048xf32>, %arg1: memref<2048xf32>, %arg2: memref<2048xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %0 = xt.load(%arg0, %bid_x) {tile = [16]} : memref<2048xf32> -> tensor<16xf32>
  %1 = xt.load(%arg1, %bid_x) {tile = [16]} : memref<2048xf32> -> tensor<16xf32>
  %2 = xt.add(%0, %1) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
  xt.store(%2, %arg2, %bid_x) {tile = [16]} : tensor<16xf32> -> memref<2048xf32>
  func.return
}

func.func @lower_all(%arg0: memref<2048x16xf32>, %arg1: memref<2048x16xf32>, %arg2: memref<2048x16xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %zero = arith.constant 0 : i32
  %0 = xt.load(%arg0, %bid_x, %zero) {tile = [16, 16]} : memref<2048x16xf32> -> tensor<16x16xf32>
  %1 = xt.load(%arg1, %bid_x, %zero) {tile = [16, 16]} : memref<2048x16xf32> -> tensor<16x16xf32>
  %2 = xt.add(%0, %1) : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
  %3 = xt.exp(%2) : tensor<16x16xf32>
  xt.store(%3, %arg2, %bid_x, %zero) {tile = [16, 16]} : tensor<16x16xf32> -> memref<2048x16xf32>
  func.return
}

func.func @lower_3d(%arg0: memref<2048x32x32xf32>, %arg1: memref<2048x32x32xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %0 = xt.load(%arg0, %bid_x, %bid_y, %bid_z) {tile = [16, 16, 16]} : memref<2048x32x32xf32> -> tensor<16x16x16xf32>
  %1 = xt.exp(%0) : tensor<16x16x16xf32>
  xt.store(%1, %arg1, %bid_x, %bid_y, %bid_z) {tile = [16, 16, 16]} : tensor<16x16x16xf32> -> memref<2048x32x32xf32>
  func.return
}

func.func @lower_extra_elementwise(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
  %0 = xt.sub(%arg0, %arg1) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
  %1 = xt.mul(%0, %arg1) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
  %2 = xt.sigmoid(%1) : tensor<16xf32>
  %3 = xt.silu(%2) : tensor<16xf32>
  func.return %3 : tensor<16xf32>
}

func.func @lower_matmul(%a: tensor<16x32xf32>, %b: tensor<32x8xf32>) -> tensor<16x8xf32> {
  %0 = xt.matmul(%a, %b) : (tensor<16x32xf32>, tensor<32x8xf32>) -> tensor<16x8xf32>
  func.return %0 : tensor<16x8xf32>
}

func.func @lower_mma(%a: tensor<16x32xi8>, %b: tensor<32x8xi8>, %acc: tensor<16x8xf32>) -> tensor<16x8xf32> {
  %0 = xt.mma(%a, %b, %acc) : (tensor<16x32xi8>, tensor<32x8xi8>, tensor<16x8xf32>) -> tensor<16x8xf32>
  func.return %0 : tensor<16x8xf32>
}

func.func @lower_shared(%arg0: memref<128x256xi8>, %arg1: memref<256x512xi8>, %arg2: memref<128x512xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %zero = arith.constant 0 : i32
  %0 = xt.load(%arg0, %bid_x, %zero) {tile = [64, 256]} : memref<128x256xi8> -> tensor<64x256xi8>
  %1 = xt.load(%arg1, %zero, %bid_y) {tile = [256, 64], shared = 1} : memref<256x512xi8> -> tensor<256x64xi8>
  %2 = xt.matmul(%0, %1) : (tensor<64x256xi8>, tensor<256x64xi8>) -> tensor<64x64xf32>
  xt.store(%2, %arg2, %bid_x, %bid_y) {tile = [64, 64]} : tensor<64x64xf32> -> memref<128x512xf32>
  func.return
}

func.func @lower_dynamic_shared(%arg0: memref<?x256xi8>, %arg1: memref<256x?xi8>, %arg2: memref<?x?xbf16>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %zero = arith.constant 0 : i32
  %0 = xt.load(%arg0, %bid_x, %zero) {tile = [64, 256]} : memref<?x256xi8> -> tensor<64x256xi8>
  %1 = xt.load(%arg1, %zero, %bid_y) {tile = [256, 64], shared = 1} : memref<256x?xi8> -> tensor<256x64xi8>
  %2 = xt.matmul(%0, %1) : (tensor<64x256xi8>, tensor<256x64xi8>) -> tensor<64x64xbf16>
  xt.store(%2, %arg2, %bid_x, %bid_y) {tile = [64, 64]} : tensor<64x64xbf16> -> memref<?x?xbf16>
  func.return
}

func.func @lower_4d(%arg0: memref<64x32x32x32xf32>, %arg1: memref<64x32x32x32xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %zero = arith.constant 0 : i32
  %0 = xt.load(%arg0, %bid_x, %bid_y, %bid_z, %zero) {tile = [16, 16, 16, 16]} : memref<64x32x32x32xf32> -> tensor<16x16x16x16xf32>
  %1 = xt.exp(%0) : tensor<16x16x16x16xf32>
  xt.store(%1, %arg1, %bid_x, %bid_y, %bid_z, %zero) {tile = [16, 16, 16, 16]} : tensor<16x16x16x16xf32> -> memref<64x32x32x32xf32>
  func.return
}

func.func @lower_broadcast(%arg0: memref<2048x16xf32>, %arg1: memref<1x16xf32>, %arg2: memref<2048x16xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %zero = arith.constant 0 : i32
  %0 = xt.load(%arg0, %bid_x, %zero) {tile = [16, 16]} : memref<2048x16xf32> -> tensor<16x16xf32>
  %1 = xt.load(%arg1, %zero, %zero) {tile = [1, 16], shared = 1} : memref<1x16xf32> -> tensor<1x16xf32>
  %2 = xt.add(%0, %1) : (tensor<16x16xf32>, tensor<1x16xf32>) -> tensor<16x16xf32>
  xt.store(%2, %arg2, %bid_x, %zero) {tile = [16, 16]} : tensor<16x16xf32> -> memref<2048x16xf32>
  func.return
}

func.func @lower_conv2d(%arg0: memref<8x32x64x128xi8>, %arg1: memref<3x3x128x256xi8>, %arg2: memref<8x32x64x256xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %zero = arith.constant 0 : i32
  %0 = xt.load(%arg0, %bid_x, %zero, %zero, %zero) {tile = [1, 32, 64, 128]} : memref<8x32x64x128xi8> -> tensor<1x32x64x128xi8>
  %1 = xt.load(%arg1, %zero, %zero, %zero, %bid_y) {tile = [3, 3, 128, 64], shared = 1} : memref<3x3x128x256xi8> -> tensor<3x3x128x64xi8>
  %2 = xt.conv2d(%0, %1) {pad = [1, 1, 1, 1], stride = [1, 1], dilation = [1, 1]} : (tensor<1x32x64x128xi8>, tensor<3x3x128x64xi8>) -> tensor<1x32x64x64xf32>
  xt.store(%2, %arg2, %bid_x, %zero, %zero, %bid_y) {tile = [1, 32, 64, 64]} : tensor<1x32x64x64xf32> -> memref<8x32x64x256xf32>
  func.return
}

// CHECK-LABEL: func.func @lower_1d
// CHECK-NOT: xt.
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.store
// CHECK-LABEL: func.func @lower_all
// CHECK-NOT: xt.
// CHECK: arith.constant 0 : i32
// CHECK: scf.for
// CHECK: memref.load
// CHECK: math.exp
// CHECK: memref.store
// CHECK-LABEL: func.func @lower_3d
// CHECK-NOT: xt.
// CHECK: scf.for
// CHECK: math.exp
// CHECK: memref.store
// CHECK-LABEL: func.func @lower_extra_elementwise
// CHECK-NOT: xt.
// CHECK: arith.subf
// CHECK: arith.mulf
// CHECK: math.exp
// CHECK: arith.divf
// CHECK-LABEL: func.func @lower_matmul
// CHECK-NOT: xt.
// CHECK: scf.for
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK-LABEL: func.func @lower_mma
// CHECK-NOT: xt.
// CHECK: arith.sitofp
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK-LABEL: func.func @lower_shared
// CHECK-NOT: xt.
// CHECK: memref.load
// CHECK: arith.sitofp
// CHECK: arith.addf
// CHECK: memref.store
// CHECK-LABEL: func.func @lower_dynamic_shared
// CHECK-NOT: xt.
// CHECK: memref.load
// CHECK: arith.sitofp
// CHECK: arith.addf
// CHECK: memref.store
// CHECK-LABEL: func.func @lower_4d
// CHECK-NOT: xt.
// CHECK: scf.for
// CHECK: math.exp
// CHECK: memref.store
// CHECK-LABEL: func.func @lower_broadcast
// CHECK-NOT: xt.
// CHECK: arith.constant 0 : index
// CHECK: %[[LHS:.+]] = tensor.extract %{{.+}}[%{{.+}}, %{{.+}}] : tensor<16x16xf32>
// CHECK: %[[RHS:.+]] = tensor.extract %{{.+}}[%{{c0.*}}, %{{.+}}] : tensor<1x16xf32>
// CHECK: arith.addf %[[LHS]], %[[RHS]] : f32
// CHECK: memref.store
// CHECK-LABEL: func.func @lower_conv2d
// CHECK-NOT: xt.
// CHECK: scf.for
// CHECK: arith.cmpi
// CHECK: scf.if
// CHECK: arith.sitofp
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: memref.store
