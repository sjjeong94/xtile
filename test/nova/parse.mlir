// RUN: xt-opt %s | FileCheck %s

func.func @parse_square(%arg0: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %0 = nova.square %arg0 : tensor<16x16xf32> -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}

func.func @parse_rsqrt(%arg0: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %0 = nova.rsqrt %arg0 : tensor<16x16xf32> -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}

func.func @parse_scalar_fma(%arg0: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %0 = nova.scalar_fma %arg0, 2.300000e+00, 3.500000e+00 : tensor<16x16xf32> -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}

func.func @parse_keep_alive(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>, %arg2: tensor<16x16xf32>) {
  nova.keep_alive %arg0 : tensor<16x16xf32>
  nova.keep_alive %arg0, %arg1, %arg2 : tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>
  func.return
}

func.func @parse_load_store(%arr: memref<128x16xf32>,
                            %tile: tensor<16x16xf32, #nova.layout<bank0 = 3, space = 3>>)
    -> tensor<16x16xf32, #nova.layout<bank0 = 7, space = 3>> {
  %0 = nova.load %arr [0, 2] {shared = 1 : i64} : memref<128x16xf32> -> tensor<16x16xf32, #nova.layout<bank0 = 7, space = 3>>
  nova.store %tile, %arr [1, 2] {shared = 1 : i64} : (tensor<16x16xf32, #nova.layout<bank0 = 3, space = 3>>, memref<128x16xf32>) -> ()
  func.return %0 : tensor<16x16xf32, #nova.layout<bank0 = 7, space = 3>>
}

func.func @parse_casts(%arg0: tensor<5x16xi8>, %arg1: tensor<5x16xf32>) -> (tensor<5x16xf32>, tensor<5x16xi8>) {
  %0 = nova.itof %arg0 : tensor<5x16xi8> -> tensor<5x16xf32>
  %1 = nova.ftoi %arg1 : tensor<5x16xf32> -> tensor<5x16xi8>
  func.return %0, %1 : tensor<5x16xf32>, tensor<5x16xi8>
}

func.func @parse_permute(%arg0: tensor<2x3x5xf32>) -> tensor<5x2x3xf32> {
  %0 = nova.permute %arg0 [2, 0, 1] : tensor<2x3x5xf32> -> tensor<5x2x3xf32>
  func.return %0 : tensor<5x2x3xf32>
}

func.func @parse_int_result_matmul(%arg0: tensor<16x32xi8>, %arg1: tensor<32x8xi8>, %scale: tensor<1x1xf32>, %bias: tensor<1x1xf32>) -> tensor<16x8xi32> {
  %0 = nova.matmul %arg0, %arg1, %scale, %bias : tensor<16x32xi8>, tensor<32x8xi8>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xi32>
  func.return %0 : tensor<16x8xi32>
}

// CHECK-LABEL: func.func @parse_square
// CHECK: %[[RES:.*]] = nova.square %arg0 : tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK: return %[[RES]] : tensor<16x16xf32>
// CHECK-LABEL: func.func @parse_rsqrt
// CHECK: %[[RSQRT:.*]] = nova.rsqrt %arg0 : tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK: return %[[RSQRT]] : tensor<16x16xf32>
// CHECK-LABEL: func.func @parse_scalar_fma
// CHECK: %[[FMA:.*]] = nova.scalar_fma %arg0, 2.300000e+00, 3.500000e+00 : tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK: return %[[FMA]] : tensor<16x16xf32>
// CHECK-LABEL: func.func @parse_keep_alive
// CHECK: nova.keep_alive %arg0 : tensor<16x16xf32>
// CHECK: nova.keep_alive %arg0, %arg1, %arg2 : tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>
// CHECK: return
// CHECK-LABEL: func.func @parse_load_store
// CHECK: %[[LOAD:.*]] = nova.load %arg0 [0, 2] {shared = 1 : i64} : memref<128x16xf32> -> tensor<16x16xf32, #nova.layout<bank0 = 7, space = 3>>
// CHECK: nova.store %arg1, %arg0 [1, 2] {shared = 1 : i64} : (tensor<16x16xf32, #nova.layout<bank0 = 3, space = 3>>, memref<128x16xf32>) -> ()
// CHECK: return %[[LOAD]] : tensor<16x16xf32, #nova.layout<bank0 = 7, space = 3>>
// CHECK-LABEL: func.func @parse_casts
// CHECK: %[[ITOF:.*]] = nova.itof %arg0 : tensor<5x16xi8> -> tensor<5x16xf32>
// CHECK: %[[FTOI:.*]] = nova.ftoi %arg1 : tensor<5x16xf32> -> tensor<5x16xi8>
// CHECK: return %[[ITOF]], %[[FTOI]] : tensor<5x16xf32>, tensor<5x16xi8>
// CHECK-LABEL: func.func @parse_permute
// CHECK: %[[PERMUTE:.*]] = nova.permute %arg0 [2, 0, 1] : tensor<2x3x5xf32> -> tensor<5x2x3xf32>
// CHECK: return %[[PERMUTE]] : tensor<5x2x3xf32>
// CHECK-LABEL: func.func @parse_int_result_matmul
// CHECK: %[[MM:.*]] = nova.matmul %arg0, %arg1, %arg2, %arg3 : tensor<16x32xi8>, tensor<32x8xi8>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<16x8xi32>
// CHECK: return %[[MM]] : tensor<16x8xi32>
