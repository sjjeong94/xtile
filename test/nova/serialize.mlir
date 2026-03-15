// RUN: xt-opt --nova-serialize %s | FileCheck %s

func.func @serialize_grid(%src: memref<128x16xf32>, %dst: memref<128x16xf32>) attributes {xt.grid = array<i32: 2, 2, 1>} {
  %bid0:3 = "xt.get_tile_block_id"() : () -> (i32, i32, i32)
  %bid1:3 = "xt.get_tile_block_id"() : () -> (i32, i32, i32)
  %0 = "xt.load"(%src, %bid0#0, %bid1#1) : (memref<128x16xf32>, i32, i32) -> tensor<16x16xf32>
  %1 = "xt.exp"(%0) : (tensor<16x16xf32>) -> tensor<16x16xf32>
  "xt.store"(%1, %dst, %bid1#0, %bid0#1) : (tensor<16x16xf32>, memref<128x16xf32>, i32, i32) -> ()
  func.return
}

func.func @serialize_shared_x(%src: memref<128x16xf32>, %dst: memref<128x16xf32>) attributes {xt.grid = array<i32: 2, 2, 1>} {
  %bid:3 = "xt.get_tile_block_id"() : () -> (i32, i32, i32)
  %0 = "xt.load"(%src, %bid#0, %bid#1) <{shared = 1 : i64}> : (memref<128x16xf32>, i32, i32) -> tensor<16x16xf32>
  "xt.store"(%0, %dst, %bid#0, %bid#1) : (tensor<16x16xf32>, memref<128x16xf32>, i32, i32) -> ()
  func.return
}

func.func @serialize_shared_xy(%src: memref<128x16xf32>, %dst: memref<128x16xf32>) attributes {xt.grid = array<i32: 2, 2, 2>} {
  %bid:3 = "xt.get_tile_block_id"() : () -> (i32, i32, i32)
  %0 = "xt.load"(%src, %bid#0, %bid#1) <{shared = 2 : i64}> : (memref<128x16xf32>, i32, i32) -> tensor<16x16xf32>
  "xt.store"(%0, %dst, %bid#0, %bid#1) : (tensor<16x16xf32>, memref<128x16xf32>, i32, i32) -> ()
  func.return
}

// CHECK-LABEL: func.func @serialize_grid
// CHECK-NOT: xt.get_tile_block_id
// CHECK: %[[X00:.*]] = arith.constant 0 : i32
// CHECK: %[[Y00:.*]] = arith.constant 0 : i32
// CHECK: %[[Z00:.*]] = arith.constant 0 : i32
// CHECK: "xt.load"(%arg0, %[[X00]], %[[Y00]])
// CHECK: "xt.store"(%{{.*}}, %arg1, %[[X00]], %[[Y00]])
// CHECK: %[[X10:.*]] = arith.constant 1 : i32
// CHECK: %[[Y10:.*]] = arith.constant 0 : i32
// CHECK: %[[Z10:.*]] = arith.constant 0 : i32
// CHECK: "xt.load"(%arg0, %[[X10]], %[[Y10]])
// CHECK: "xt.store"(%{{.*}}, %arg1, %[[X10]], %[[Y10]])
// CHECK: %[[X01:.*]] = arith.constant 0 : i32
// CHECK: %[[Y01:.*]] = arith.constant 1 : i32
// CHECK: %[[Z01:.*]] = arith.constant 0 : i32
// CHECK: "xt.load"(%arg0, %[[X01]], %[[Y01]])
// CHECK: "xt.store"(%{{.*}}, %arg1, %[[X01]], %[[Y01]])
// CHECK: %[[X11:.*]] = arith.constant 1 : i32
// CHECK: %[[Y11:.*]] = arith.constant 1 : i32
// CHECK: %[[Z11:.*]] = arith.constant 0 : i32
// CHECK: "xt.load"(%arg0, %[[X11]], %[[Y11]])
// CHECK: "xt.store"(%{{.*}}, %arg1, %[[X11]], %[[Y11]])

// CHECK-LABEL: func.func @serialize_shared_x
// CHECK-NOT: xt.get_tile_block_id
// CHECK: %[[SX0:.*]] = "xt.load"(%arg0, %{{.*}}, %{{.*}}) <{shared = 1 : i64}>
// CHECK: "xt.store"(%[[SX0]], %arg1, %{{.*}}, %{{.*}})
// CHECK: "xt.store"(%[[SX0]], %arg1, %{{.*}}, %{{.*}})
// CHECK: %[[SX1:.*]] = "xt.load"(%arg0, %{{.*}}, %{{.*}}) <{shared = 1 : i64}>
// CHECK: "xt.store"(%[[SX1]], %arg1, %{{.*}}, %{{.*}})
// CHECK: "xt.store"(%[[SX1]], %arg1, %{{.*}}, %{{.*}})

// CHECK-LABEL: func.func @serialize_shared_xy
// CHECK-NOT: xt.get_tile_block_id
// CHECK: %[[SXY0:.*]] = "xt.load"(%arg0, %{{.*}}, %{{.*}}) <{shared = 2 : i64}>
// CHECK: "xt.store"(%[[SXY0]], %arg1, %{{.*}}, %{{.*}})
// CHECK: "xt.store"(%[[SXY0]], %arg1, %{{.*}}, %{{.*}})
// CHECK: "xt.store"(%[[SXY0]], %arg1, %{{.*}}, %{{.*}})
// CHECK: "xt.store"(%[[SXY0]], %arg1, %{{.*}}, %{{.*}})
// CHECK: %[[SXY1:.*]] = "xt.load"(%arg0, %{{.*}}, %{{.*}}) <{shared = 2 : i64}>
// CHECK: "xt.store"(%[[SXY1]], %arg1, %{{.*}}, %{{.*}})
// CHECK: "xt.store"(%[[SXY1]], %arg1, %{{.*}}, %{{.*}})
// CHECK: "xt.store"(%[[SXY1]], %arg1, %{{.*}}, %{{.*}})
// CHECK: "xt.store"(%[[SXY1]], %arg1, %{{.*}}, %{{.*}})
