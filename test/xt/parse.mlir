// RUN: xt-opt %s | FileCheck %s

func.func @exp_1d(%arg0: memref<2048xf32>, %arg1: memref<2048xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %0 = xt.load(%arg0, %bid_x) : memref<2048xf32> -> tensor<16xf32>
  %1 = xt.exp(%0) : tensor<16xf32>
  xt.store(%1, %arg1, %bid_x) : tensor<16xf32> -> memref<2048xf32>
  func.return
}

func.func @exp(%arg0: memref<2048x16xf32>, %arg1: memref<2048x16xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %zero = arith.constant 0 : i32

  %0 = xt.load(%arg0, %bid_x, %zero) : memref<2048x16xf32> -> tensor<16x16xf32>
  %1 = xt.exp(%0) : tensor<16x16xf32>
  xt.store(%1, %arg1, %bid_x, %zero) : tensor<16x16xf32> -> memref<2048x16xf32>
  func.return
}

func.func @add(%arg0: memref<2048x16xf32>, %arg1: memref<2048x16xf32>, %arg2: memref<2048x16xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %zero = arith.constant 0 : i32

  %0 = xt.load(%arg0, %bid_x, %zero) : memref<2048x16xf32> -> tensor<16x16xf32>
  %1 = xt.load(%arg1, %bid_x, %zero) : memref<2048x16xf32> -> tensor<16x16xf32>
  %2 = xt.add(%0, %1) : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
  xt.store(%2, %arg2, %bid_x, %zero) : tensor<16x16xf32> -> memref<2048x16xf32>
  func.return
}

func.func @exp_3d(%arg0: memref<2048x32x32xf32>, %arg1: memref<2048x32x32xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %0 = xt.load(%arg0, %bid_x, %bid_y, %bid_z) : memref<2048x32x32xf32> -> tensor<16x16x16xf32>
  %1 = xt.exp(%0) : tensor<16x16x16xf32>
  xt.store(%1, %arg1, %bid_x, %bid_y, %bid_z) : tensor<16x16x16xf32> -> memref<2048x32x32xf32>
  func.return
}

func.func @elementwise_ops(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
  %0 = xt.sub(%arg0, %arg1) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
  %1 = xt.mul(%0, %arg1) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
  %2 = xt.cos(%1) : tensor<16xf32>
  %3 = xt.sin(%2) : tensor<16xf32>
  %4 = xt.reciprocal(%3) : tensor<16xf32>
  %5 = xt.rsqrt(%4) : tensor<16xf32>
  %6 = xt.sigmoid(%5) : tensor<16xf32>
  %7 = xt.tanh(%6) : tensor<16xf32>
  %8 = xt.silu(%7) : tensor<16xf32>
  func.return %8 : tensor<16xf32>
}

func.func @contract_ops(%a: tensor<16x32xf32>, %b: tensor<32x8xf32>, %ai8: tensor<16x32xi8>, %bi8: tensor<32x8xi8>, %acc: tensor<16x8xf32>) -> (tensor<16x8xf32>, tensor<16x8xf32>) {
  %0 = xt.matmul(%a, %b) : (tensor<16x32xf32>, tensor<32x8xf32>) -> tensor<16x8xf32>
  %1 = xt.mma(%ai8, %bi8, %acc) : (tensor<16x32xi8>, tensor<32x8xi8>, tensor<16x8xf32>) -> tensor<16x8xf32>
  func.return %0, %1 : tensor<16x8xf32>, tensor<16x8xf32>
}

func.func @shared(%arg0: memref<128x256xi8>, %arg1: memref<256x512xi8>, %arg2: memref<128x512xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %zero = arith.constant 0 : i32

  %0 = xt.load(%arg0, %bid_x, %zero) : memref<128x256xi8> -> tensor<64x256xi8>
  %1 = xt.load(%arg1, %zero, %bid_y) {shared = 1} : memref<256x512xi8> -> tensor<256x64xi8>
  %2 = xt.matmul(%0, %1) : (tensor<64x256xi8>, tensor<256x64xi8>) -> tensor<64x64xf32>
  xt.store(%2, %arg2, %bid_x, %bid_y) : tensor<64x64xf32> -> memref<128x512xf32>
  func.return
}

func.func @dynamic_shared(%arg0: memref<?x256xi8>, %arg1: memref<256x?xi8>, %arg2: memref<128x512xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %zero = arith.constant 0 : i32

  %0 = xt.load(%arg0, %bid_x, %zero) : memref<?x256xi8> -> tensor<64x256xi8>
  %1 = xt.load(%arg1, %zero, %bid_y) {shared = 1} : memref<256x?xi8> -> tensor<256x64xi8>
  %2 = xt.matmul(%0, %1) : (tensor<64x256xi8>, tensor<256x64xi8>) -> tensor<64x64xf32>
  xt.store(%2, %arg2, %bid_x, %bid_y) : tensor<64x64xf32> -> memref<128x512xf32>
  func.return
}

func.func @exp_4d(%arg0: memref<64x32x32x32xf32>, %arg1: memref<64x32x32x32xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %zero = arith.constant 0 : i32

  %0 = xt.load(%arg0, %bid_x, %bid_y, %bid_z, %zero) : memref<64x32x32x32xf32> -> tensor<16x16x16x16xf32>
  %1 = xt.exp(%0) : tensor<16x16x16x16xf32>
  xt.store(%1, %arg1, %bid_x, %bid_y, %bid_z, %zero) : tensor<16x16x16x16xf32> -> memref<64x32x32x32xf32>
  func.return
}

func.func @broadcast(%arg0: memref<2048x16xf32>, %arg1: memref<1x16xf32>, %arg2: memref<2048x16xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %zero = arith.constant 0 : i32

  %0 = xt.load(%arg0, %bid_x, %zero) : memref<2048x16xf32> -> tensor<16x16xf32>
  %1 = xt.load(%arg1, %zero, %zero) {shared = 1} : memref<1x16xf32> -> tensor<1x16xf32>
  %2 = xt.add(%0, %1) : (tensor<16x16xf32>, tensor<1x16xf32>) -> tensor<16x16xf32>
  xt.store(%2, %arg2, %bid_x, %zero) : tensor<16x16xf32> -> memref<2048x16xf32>
  func.return
}

func.func @conv2d(%arg0: memref<8x32x64x128xi8>, %arg1: memref<3x3x128x256xi8>, %arg2: memref<8x32x64x256xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %zero = arith.constant 0 : i32

  %0 = xt.load(%arg0, %bid_x, %zero, %zero, %zero) : memref<8x32x64x128xi8> -> tensor<1x32x64x128xi8>
  %1 = xt.load(%arg1, %zero, %zero, %zero, %bid_y) {shared = 1} : memref<3x3x128x256xi8> -> tensor<3x3x128x64xi8>
  %2 = xt.conv2d(%0, %1) {pad = [1, 1, 1, 1], stride = [1, 1], dilation = [1, 1]} : (tensor<1x32x64x128xi8>, tensor<3x3x128x64xi8>) -> tensor<1x32x64x64xf32>
  xt.store(%2, %arg2, %bid_x, %zero, %zero, %bid_y) : tensor<1x32x64x64xf32> -> memref<8x32x64x256xf32>
  func.return
}

func.func @depthwise_conv2d(%arg0: memref<8x32x64x256xi8>, %arg1: memref<3x3x1x256xi8>, %arg2: memref<8x32x64x256xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %zero = arith.constant 0 : i32

  %0 = xt.load(%arg0, %bid_x, %zero, %zero, %bid_y) : memref<8x32x64x256xi8> -> tensor<1x32x64x64xi8>
  %1 = xt.load(%arg1, %zero, %zero, %zero, %bid_y) {shared = 1} : memref<3x3x1x256xi8> -> tensor<3x3x1x64xi8>
  %2 = xt.depthwise_conv2d(%0, %1) {pad = [1, 1, 1, 1], stride = [1, 1], dilation = [1, 1]} : (tensor<1x32x64x64xi8>, tensor<3x3x1x64xi8>) -> tensor<1x32x64x64xf32>
  xt.store(%2, %arg2, %bid_x, %zero, %zero, %bid_y) : tensor<1x32x64x64xf32> -> memref<8x32x64x256xf32>
  func.return
}

func.func @reduce_ops(%arg0: memref<2048x16xf32>, %arg1: memref<2048x16xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %zero = arith.constant 0 : i32

  %0 = xt.load(%arg0, %bid_x, %zero) : memref<2048x16xf32> -> tensor<16x16xf32>
  %1 = xt.reduce_sum(%0) : (tensor<16x16xf32>) -> tensor<16x1xf32>
  %2 = xt.reduce_max(%0) : (tensor<16x16xf32>) -> tensor<16x1xf32>
  %3 = xt.sub(%0, %1) : (tensor<16x16xf32>, tensor<16x1xf32>) -> tensor<16x16xf32>
  %4 = xt.sub(%3, %2) : (tensor<16x16xf32>, tensor<16x1xf32>) -> tensor<16x16xf32>
  xt.store(%4, %arg1, %bid_x, %zero) : tensor<16x16xf32> -> memref<2048x16xf32>
  func.return
}

func.func @reshape_transpose(%arg0: memref<2048x16xf32>, %arg1: memref<2048x16xf32>) {
  %bid_x, %bid_y, %bid_z = xt.get_tile_block_id() : i32
  %zero = arith.constant 0 : i32

  %0 = xt.load(%arg0, %bid_x, %zero) : memref<2048x16xf32> -> tensor<64x16xf32>
  %1 = xt.reshape(%0) : (tensor<64x16xf32>) -> tensor<2x32x16xf32>
  %2 = xt.transpose(%1) : (tensor<2x32x16xf32>) -> tensor<2x16x32xf32>
  %3 = xt.reshape(%2) : (tensor<2x16x32xf32>) -> tensor<64x16xf32>
  xt.store(%3, %arg1, %bid_x, %zero) : tensor<64x16xf32> -> memref<2048x16xf32>
  func.return
}

// CHECK-LABEL: func.func @exp_1d
// CHECK: %[[BIDX1:.*]], %[[BIDY1:.*]], %[[BIDZ1:.*]] = xt.get_tile_block_id() : i32
// CHECK: %[[LOAD1:.*]] = xt.load(%arg0, %[[BIDX1]]) : memref<2048xf32> -> tensor<16xf32>
// CHECK: %[[EXP1:.*]] = xt.exp(%[[LOAD1]]) : tensor<16xf32>
// CHECK: xt.store(%[[EXP1]], %arg1, %[[BIDX1]]) : tensor<16xf32> -> memref<2048xf32>
// CHECK: %[[BIDX:.*]], %[[BIDY:.*]], %[[BIDZ:.*]] = xt.get_tile_block_id() : i32
// CHECK: %[[LOAD:.*]] = xt.load(%arg0, %[[BIDX]], %[[ZERO:.*]]) : memref<2048x16xf32> -> tensor<16x16xf32>
// CHECK: %[[EXP:.*]] = xt.exp(%[[LOAD]]) : tensor<16x16xf32>
// CHECK: xt.store(%[[EXP]], %arg1, %[[BIDX]], %[[ZERO]]) : tensor<16x16xf32> -> memref<2048x16xf32>
// CHECK-LABEL: func.func @exp_3d
// CHECK: %[[BIDX3:.*]], %[[BIDY3:.*]], %[[BIDZ3:.*]] = xt.get_tile_block_id() : i32
// CHECK: %[[LOAD3:.*]] = xt.load(%arg0, %[[BIDX3]], %[[BIDY3]], %[[BIDZ3]]) : memref<2048x32x32xf32> -> tensor<16x16x16xf32>
// CHECK: %[[EXP3:.*]] = xt.exp(%[[LOAD3]]) : tensor<16x16x16xf32>
// CHECK: xt.store(%[[EXP3]], %arg1, %[[BIDX3]], %[[BIDY3]], %[[BIDZ3]]) : tensor<16x16x16xf32> -> memref<2048x32x32xf32>
// CHECK-LABEL: func.func @elementwise_ops
// CHECK: xt.sub(%arg0, %arg1) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
// CHECK: xt.mul(%{{.*}}, %arg1) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
// CHECK: xt.cos
// CHECK: xt.sin
// CHECK: xt.reciprocal
// CHECK: xt.rsqrt
// CHECK: xt.sigmoid
// CHECK: xt.tanh
// CHECK: xt.silu
// CHECK-LABEL: func.func @contract_ops
// CHECK: xt.matmul(%arg0, %arg1) : (tensor<16x32xf32>, tensor<32x8xf32>) -> tensor<16x8xf32>
// CHECK: xt.mma(%arg2, %arg3, %arg4) : (tensor<16x32xi8>, tensor<32x8xi8>, tensor<16x8xf32>) -> tensor<16x8xf32>
// CHECK-LABEL: func.func @shared
// CHECK: %[[BIDXS:.*]], %[[BIDYS:.*]], %[[BIDZS:.*]] = xt.get_tile_block_id() : i32
// CHECK: %[[LOADS0:.*]] = xt.load(%arg0, %[[BIDXS]], %[[ZERO_SHARED:.*]]) : memref<128x256xi8> -> tensor<64x256xi8>
// CHECK: %[[LOADS1:.*]] = xt.load(%arg1, %[[ZERO_SHARED]], %[[BIDYS]]) {shared = 1} : memref<256x512xi8> -> tensor<256x64xi8>
// CHECK: %[[MM:.*]] = xt.matmul(%[[LOADS0]], %[[LOADS1]]) : (tensor<64x256xi8>, tensor<256x64xi8>) -> tensor<64x64xf32>
// CHECK: xt.store(%[[MM]], %arg2, %[[BIDXS]], %[[BIDYS]]) : tensor<64x64xf32> -> memref<128x512xf32>
// CHECK-LABEL: func.func @dynamic_shared
// CHECK: %[[BIDXD:.*]], %[[BIDYD:.*]], %[[BIDZD:.*]] = xt.get_tile_block_id() : i32
// CHECK: %[[LOADD0:.*]] = xt.load(%arg0, %[[BIDXD]], %[[ZERO_DYNAMIC:.*]]) : memref<?x256xi8> -> tensor<64x256xi8>
// CHECK: %[[LOADD1:.*]] = xt.load(%arg1, %[[ZERO_DYNAMIC]], %[[BIDYD]]) {shared = 1} : memref<256x?xi8> -> tensor<256x64xi8>
// CHECK: %[[MMD:.*]] = xt.matmul(%[[LOADD0]], %[[LOADD1]]) : (tensor<64x256xi8>, tensor<256x64xi8>) -> tensor<64x64xf32>
// CHECK: xt.store(%[[MMD]], %arg2, %[[BIDXD]], %[[BIDYD]]) : tensor<64x64xf32> -> memref<128x512xf32>
// CHECK-LABEL: func.func @exp_4d
// CHECK: %[[BIDX4:.*]], %[[BIDY4:.*]], %[[BIDZ4:.*]] = xt.get_tile_block_id() : i32
// CHECK: %[[LOAD4:.*]] = xt.load(%arg0, %[[BIDX4]], %[[BIDY4]], %[[BIDZ4]], %[[ZERO4:.*]]) : memref<64x32x32x32xf32> -> tensor<16x16x16x16xf32>
// CHECK: %[[EXP4:.*]] = xt.exp(%[[LOAD4]]) : tensor<16x16x16x16xf32>
// CHECK: xt.store(%[[EXP4]], %arg1, %[[BIDX4]], %[[BIDY4]], %[[BIDZ4]], %[[ZERO4]]) : tensor<16x16x16x16xf32> -> memref<64x32x32x32xf32>
// CHECK-LABEL: func.func @broadcast
// CHECK: %[[BIDXB:.*]], %[[BIDYB:.*]], %[[BIDZB:.*]] = xt.get_tile_block_id() : i32
// CHECK: %[[LOADB0:.*]] = xt.load(%arg0, %[[BIDXB]], %[[ZEROB:.*]]) : memref<2048x16xf32> -> tensor<16x16xf32>
// CHECK: %[[LOADB1:.*]] = xt.load(%arg1, %[[ZEROB]], %[[ZEROB]]) {shared = 1} : memref<1x16xf32> -> tensor<1x16xf32>
// CHECK: %[[ADDB:.*]] = xt.add(%[[LOADB0]], %[[LOADB1]]) : (tensor<16x16xf32>, tensor<1x16xf32>) -> tensor<16x16xf32>
// CHECK: xt.store(%[[ADDB]], %arg2, %[[BIDXB]], %[[ZEROB]]) : tensor<16x16xf32> -> memref<2048x16xf32>
// CHECK-LABEL: func.func @conv2d
// CHECK: %[[BIDXC:.*]], %[[BIDYC:.*]], %[[BIDZC:.*]] = xt.get_tile_block_id() : i32
// CHECK: %[[LOADC0:.*]] = xt.load(%arg0, %[[BIDXC]], %[[ZEROC:.*]], %[[ZEROC]], %[[ZEROC]]) : memref<8x32x64x128xi8> -> tensor<1x32x64x128xi8>
// CHECK: %[[LOADC1:.*]] = xt.load(%arg1, %[[ZEROC]], %[[ZEROC]], %[[ZEROC]], %[[BIDYC]]) {shared = 1} : memref<3x3x128x256xi8> -> tensor<3x3x128x64xi8>
// CHECK: %[[CONV:.*]] = xt.conv2d(%[[LOADC0]], %[[LOADC1]]) {pad = [1, 1, 1, 1], stride = [1, 1], dilation = [1, 1]} : (tensor<1x32x64x128xi8>, tensor<3x3x128x64xi8>) -> tensor<1x32x64x64xf32>
// CHECK: xt.store(%[[CONV]], %arg2, %[[BIDXC]], %[[ZEROC]], %[[ZEROC]], %[[BIDYC]]) : tensor<1x32x64x64xf32> -> memref<8x32x64x256xf32>
// CHECK-LABEL: func.func @depthwise_conv2d
// CHECK: %[[BIDXD:.*]], %[[BIDYD:.*]], %[[BIDZD:.*]] = xt.get_tile_block_id() : i32
// CHECK: %[[LOADD0:.*]] = xt.load(%arg0, %[[BIDXD]], %[[ZERODW:.*]], %[[ZERODW]], %[[BIDYD]]) : memref<8x32x64x256xi8> -> tensor<1x32x64x64xi8>
// CHECK: %[[LOADD1:.*]] = xt.load(%arg1, %[[ZERODW]], %[[ZERODW]], %[[ZERODW]], %[[BIDYD]]) {shared = 1} : memref<3x3x1x256xi8> -> tensor<3x3x1x64xi8>
// CHECK: %[[DWCONV:.*]] = xt.depthwise_conv2d(%[[LOADD0]], %[[LOADD1]]) {pad = [1, 1, 1, 1], stride = [1, 1], dilation = [1, 1]} : (tensor<1x32x64x64xi8>, tensor<3x3x1x64xi8>) -> tensor<1x32x64x64xf32>
// CHECK: xt.store(%[[DWCONV]], %arg2, %[[BIDXD]], %[[ZERODW]], %[[ZERODW]], %[[BIDYD]]) : tensor<1x32x64x64xf32> -> memref<8x32x64x256xf32>
// CHECK-LABEL: func.func @reduce_ops
// CHECK: %[[BIDXR:.*]], %[[BIDYR:.*]], %[[BIDZR:.*]] = xt.get_tile_block_id() : i32
// CHECK: %[[LOADR:.*]] = xt.load(%arg0, %[[BIDXR]], %[[ZEROR:.*]]) : memref<2048x16xf32> -> tensor<16x16xf32>
// CHECK: %[[SUM:.*]] = xt.reduce_sum(%[[LOADR]]) : (tensor<16x16xf32>) -> tensor<16x1xf32>
// CHECK: %[[MAX:.*]] = xt.reduce_max(%[[LOADR]]) : (tensor<16x16xf32>) -> tensor<16x1xf32>
// CHECK: %[[SUB0:.*]] = xt.sub(%[[LOADR]], %[[SUM]]) : (tensor<16x16xf32>, tensor<16x1xf32>) -> tensor<16x16xf32>
// CHECK: %[[SUB1:.*]] = xt.sub(%[[SUB0]], %[[MAX]]) : (tensor<16x16xf32>, tensor<16x1xf32>) -> tensor<16x16xf32>
// CHECK: xt.store(%[[SUB1]], %arg1, %[[BIDXR]], %[[ZEROR]]) : tensor<16x16xf32> -> memref<2048x16xf32>
// CHECK-LABEL: func.func @reshape_transpose
// CHECK: %[[BIDXT:.*]], %[[BIDYT:.*]], %[[BIDZT:.*]] = xt.get_tile_block_id() : i32
// CHECK: %[[LOADT:.*]] = xt.load(%arg0, %[[BIDXT]], %[[ZEROT:.*]]) : memref<2048x16xf32> -> tensor<64x16xf32>
// CHECK: %[[RESHAPE0:.*]] = xt.reshape(%[[LOADT]]) : (tensor<64x16xf32>) -> tensor<2x32x16xf32>
// CHECK: %[[TRANSPOSE:.*]] = xt.transpose(%[[RESHAPE0]]) : (tensor<2x32x16xf32>) -> tensor<2x16x32xf32>
// CHECK: %[[RESHAPE1:.*]] = xt.reshape(%[[TRANSPOSE]]) : (tensor<2x16x32xf32>) -> tensor<64x16xf32>
// CHECK: xt.store(%[[RESHAPE1]], %arg1, %[[BIDXT]], %[[ZEROT]]) : tensor<64x16xf32> -> memref<2048x16xf32>
