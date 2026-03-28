// RUN: xt-opt --nova-to-x1 %s | FileCheck %s

func.func @barrier_only() {
  nova.barrier 1
  func.return
}

// CHECK-LABEL: func.func @barrier_only
// CHECK: x1.barrier 1
// CHECK-NEXT: return

func.func @rowwise_softmax(%arg0: memref<128x64xf32>,
                           %arg1: memref<128x64xf32>) attributes {xt.double_buffering = 1 : i32, xt.grid = array<i32: 2, 1, 1>} {
  %0 = nova.load %arg0 [0, 0] : memref<128x64xf32> -> tensor<64x64xf32, {bank0 = 0 : i64, bank1 = 1 : i64, space = 3 : i64, threading = 32 : i64}>
  %1 = nova.reduce 1 %0 {axis = 1 : i64} : tensor<64x64xf32, {bank0 = 0 : i64, bank1 = 1 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x1xf32, {bank0 = 2 : i64, bank1 = 3 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %2 = nova.broadcast 3 %0, %1 lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00 : tensor<64x64xf32, {bank0 = 0 : i64, bank1 = 1 : i64, space = 3 : i64, threading = 32 : i64}>, tensor<64x1xf32, {bank0 = 2 : i64, bank1 = 3 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x64xf32, {bank0 = 4 : i64, bank1 = 5 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %3 = nova.exp %2 : tensor<64x64xf32, {bank0 = 4 : i64, bank1 = 5 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x64xf32, {bank0 = 2 : i64, space = 3 : i64}>
  nova.barrier 0
  %4 = nova.reduce 0 %3 {axis = 1 : i64} : tensor<64x64xf32, {bank0 = 2 : i64, space = 3 : i64}> -> tensor<64x1xf32, {bank0 = 4 : i64, space = 3 : i64}>
  nova.barrier 0
  %5 = nova.reciprocal %4 : tensor<64x1xf32, {bank0 = 4 : i64, space = 3 : i64}> -> tensor<64x1xf32, {bank0 = 6 : i64, space = 3 : i64}>
  nova.barrier 0
  %6 = nova.broadcast 2 %3, %5 lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00 : tensor<64x64xf32, {bank0 = 2 : i64, space = 3 : i64}>, tensor<64x1xf32, {bank0 = 6 : i64, space = 3 : i64}> -> tensor<64x64xf32, {bank0 = 4 : i64, space = 3 : i64}>
  nova.barrier 0
  nova.store %6, %arg1 [0, 0] : (tensor<64x64xf32, {bank0 = 4 : i64, space = 3 : i64}>, memref<128x64xf32>) -> ()
  %7 = nova.load %arg0 [64, 0] : memref<128x64xf32> -> tensor<64x64xf32, {bank0 = 2 : i64, bank1 = 3 : i64, space = 3 : i64, threading = 32 : i64}>
  %8 = nova.reduce 1 %7 {axis = 1 : i64} : tensor<64x64xf32, {bank0 = 2 : i64, bank1 = 3 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x1xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %9 = nova.broadcast 3 %7, %8 lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00 : tensor<64x64xf32, {bank0 = 2 : i64, bank1 = 3 : i64, space = 3 : i64, threading = 32 : i64}>, tensor<64x1xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %10 = nova.exp %9 : tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x64xf32, {bank0 = 2 : i64, space = 3 : i64}>
  nova.barrier 0
  %11 = nova.reduce 0 %10 {axis = 1 : i64} : tensor<64x64xf32, {bank0 = 2 : i64, space = 3 : i64}> -> tensor<64x1xf32, {bank0 = 6 : i64, space = 3 : i64}>
  nova.barrier 0
  %12 = nova.reciprocal %11 : tensor<64x1xf32, {bank0 = 6 : i64, space = 3 : i64}> -> tensor<64x1xf32, {bank0 = 8 : i64, space = 3 : i64}>
  nova.barrier 0
  %13 = nova.broadcast 2 %10, %12 lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00 : tensor<64x64xf32, {bank0 = 2 : i64, space = 3 : i64}>, tensor<64x1xf32, {bank0 = 8 : i64, space = 3 : i64}> -> tensor<64x64xf32, {bank0 = 6 : i64, space = 3 : i64}>
  nova.barrier 0
  nova.store %13, %arg1 [64, 0] : (tensor<64x64xf32, {bank0 = 6 : i64, space = 3 : i64}>, memref<128x64xf32>) -> ()
  nova.barrier 1
  return
}

// CHECK-LABEL: func.func @rowwise_softmax
// CHECK-NEXT: x1.load %arg0 0 [0, 0] [32, 64] space 3 thread 0 : memref<128x64xf32>
// CHECK-NEXT: x1.load %arg0 1 [32, 0] [32, 64] space 3 thread 1 : memref<128x64xf32>
// CHECK-NEXT: x1.reduce 1 inp 0 1 out 2 3 [32, 64] axis 1
// CHECK-NEXT: x1.barrier 0
// CHECK-NEXT: x1.broadcast 3 lhs 0 1 rhs 2 3 out 4 5 [32, 64] axis 1 lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00
// CHECK-NEXT: x1.barrier 0
// CHECK-NEXT: x1.exp inp 4 5 out 2 3 [32, 64]
// CHECK-NEXT: x1.barrier 0
// CHECK-NEXT: x1.reduce 0 inp 2 3 out 4 5 [32, 64] axis 1
// CHECK-NEXT: x1.barrier 0
// CHECK-NEXT: x1.reciprocal inp 4 5 out 6 7 [64, 1]
// CHECK-NEXT: x1.barrier 0
// CHECK-NEXT: x1.broadcast 2 lhs 2 3 rhs 6 7 out 4 5 [32, 64] axis 1 lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00
// CHECK-NEXT: x1.barrier 0
// CHECK-NEXT: x1.store %arg1 4 [0, 0] [32, 64] space 3 thread 0 : memref<128x64xf32>
// CHECK-NEXT: x1.store %arg1 5 [32, 0] [32, 64] space 3 thread 1 : memref<128x64xf32>
// CHECK-NEXT: x1.load %arg0 2 [64, 0] [32, 64] space 3 thread 0 : memref<128x64xf32>
// CHECK-NEXT: x1.load %arg0 3 [96, 0] [32, 64] space 3 thread 1 : memref<128x64xf32>
// CHECK-NEXT: x1.reduce 1 inp 2 3 out 6 7 [32, 64] axis 1
// CHECK-NEXT: x1.barrier 0
// CHECK-NEXT: x1.broadcast 3 lhs 2 3 rhs 6 7 out 8 9 [32, 64] axis 1 lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00
// CHECK-NEXT: x1.barrier 0
// CHECK-NEXT: x1.exp inp 8 9 out 2 3 [32, 64]
// CHECK-NEXT: x1.barrier 0
// CHECK-NEXT: x1.reduce 0 inp 2 3 out 6 7 [32, 64] axis 1
// CHECK-NEXT: x1.barrier 0
// CHECK-NEXT: x1.reciprocal inp 6 7 out 8 9 [64, 1]
// CHECK-NEXT: x1.barrier 0
// CHECK-NEXT: x1.broadcast 2 lhs 2 3 rhs 8 9 out 6 7 [32, 64] axis 1 lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00
// CHECK-NEXT: x1.barrier 0
// CHECK-NEXT: x1.store %arg1 6 [64, 0] [32, 64] space 3 thread 0 : memref<128x64xf32>
// CHECK-NEXT: x1.store %arg1 7 [96, 0] [32, 64] space 3 thread 1 : memref<128x64xf32>
// CHECK-NEXT: x1.barrier 1
// CHECK-NEXT: return

func.func @rowwise_layernorm(%arg0: memref<128x64xf32>, %arg1: memref<1x64xf32>,
                             %arg2: memref<1x64xf32>, %arg3: memref<128x64xf32>) attributes {xt.double_buffering = 1 : i32, xt.grid = array<i32: 2, 1, 1>} {
  %0 = nova.load %arg0 [0, 0] : memref<128x64xf32> -> tensor<64x64xf32, {bank0 = 0 : i64, bank1 = 1 : i64, space = 3 : i64, threading = 32 : i64}>
  %1 = nova.load %arg1 [0, 0] {shared = 1 : i64} : memref<1x64xf32> -> tensor<1x64xf32, {bank0 = 2 : i64, space = 3 : i64, threading = 1 : i64}>
  %2 = nova.load %arg2 [0, 0] {shared = 1 : i64} : memref<1x64xf32> -> tensor<1x64xf32, {bank0 = 4 : i64, space = 3 : i64, threading = 1 : i64}>
  %3 = nova.reduce 0 %0 {axis = 1 : i64} : tensor<64x64xf32, {bank0 = 0 : i64, bank1 = 1 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x1xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %4 = nova.broadcast 3 %0, %3 lhs 1.000000e+00 0.000000e+00 rhs 1.562500e-02 0.000000e+00 : tensor<64x64xf32, {bank0 = 0 : i64, bank1 = 1 : i64, space = 3 : i64, threading = 32 : i64}>, tensor<64x1xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %5 = nova.square %4 : tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x64xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %6 = nova.reduce 0 %5 {axis = 1 : i64} : tensor<64x64xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x1xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %7 = nova.scalar_fma %6, 1.562500e-02, 9.99999974E-6 : tensor<64x1xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x1xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %8 = nova.rsqrt %7 : tensor<64x1xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x1xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %9 = nova.broadcast 2 %4, %8 lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00 : tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}>, tensor<64x1xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x64xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %10 = nova.broadcast 2 %9, %1 lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00 : tensor<64x64xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}>, tensor<1x64xf32, {bank0 = 2 : i64, space = 3 : i64, threading = 1 : i64}> -> tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %11 = nova.broadcast 1 %10, %2 lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00 : tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}>, tensor<1x64xf32, {bank0 = 4 : i64, space = 3 : i64, threading = 1 : i64}> -> tensor<64x64xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  nova.store %11, %arg3 [0, 0] : (tensor<64x64xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}>, memref<128x64xf32>) -> ()
  %12 = nova.load %arg0 [64, 0] : memref<128x64xf32> -> tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}>
  %13 = nova.reduce 0 %12 {axis = 1 : i64} : tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x1xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %14 = nova.broadcast 3 %12, %13 lhs 1.000000e+00 0.000000e+00 rhs 1.562500e-02 0.000000e+00 : tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}>, tensor<64x1xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x64xf32, {bank0 = 12 : i64, bank1 = 13 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %15 = nova.square %14 : tensor<64x64xf32, {bank0 = 12 : i64, bank1 = 13 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %16 = nova.reduce 0 %15 {axis = 1 : i64} : tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x1xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %17 = nova.scalar_fma %16, 1.562500e-02, 9.99999974E-6 : tensor<64x1xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x1xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %18 = nova.rsqrt %17 : tensor<64x1xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x1xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %19 = nova.broadcast 2 %14, %18 lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00 : tensor<64x64xf32, {bank0 = 12 : i64, bank1 = 13 : i64, space = 3 : i64, threading = 32 : i64}>, tensor<64x1xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %20 = nova.broadcast 2 %19, %1 lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00 : tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}>, tensor<1x64xf32, {bank0 = 2 : i64, space = 3 : i64, threading = 1 : i64}> -> tensor<64x64xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %21 = nova.broadcast 1 %20, %2 lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00 : tensor<64x64xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}>, tensor<1x64xf32, {bank0 = 4 : i64, space = 3 : i64, threading = 1 : i64}> -> tensor<64x64xf32, {bank0 = 2 : i64, bank1 = 3 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  nova.store %21, %arg3 [64, 0] : (tensor<64x64xf32, {bank0 = 2 : i64, bank1 = 3 : i64, space = 3 : i64, threading = 32 : i64}>, memref<128x64xf32>) -> ()
  nova.barrier 1
  return
}

func.func @column_reduce(%arg0: memref<64x64xf32>) {
  %0 = nova.load %arg0 [0, 0] : memref<64x64xf32> -> tensor<64x64xf32, {bank0 = 0 : i64, bank1 = 1 : i64, space = 3 : i64, threading = 32 : i64}>
  %1 = nova.reduce 0 %0 {axis = 0 : i64} : tensor<64x64xf32, {bank0 = 0 : i64, bank1 = 1 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<1x64xf32, {bank0 = 2 : i64, space = 3 : i64}>
  return
}

func.func @legacy_split_bank_inference(%arg0: memref<128x128xf32>) {
  %0 = nova.load %arg0 [0, 0] : memref<128x128xf32> -> tensor<64x128xf32, {bank0 = 0 : i64, shape0 = array<i64: 32, 128>, shape1 = array<i64: 32, 128>, space = 3 : i64, start0 = array<i64: 0, 0>, start1 = array<i64: 32, 0>}>
  return
}

func.func @elementwise_add(%arg0: memref<128x128xf32>, %arg1: memref<128x128xf32>,
                           %arg2: memref<128x128xf32>) attributes {xt.double_buffering = 1 : i32, xt.grid = array<i32: 2, 1, 1>} {
  %0 = nova.load %arg0 [0, 0] : memref<128x128xf32> -> tensor<64x128xf32, #nova.layout<range0 [0, 0] [32, 128], range1 [32, 0] [32, 128], bank0 = 0, bank1 = 1, space = 3>>
  %1 = nova.load %arg1 [0, 0] : memref<128x128xf32> -> tensor<64x128xf32, #nova.layout<range0 [0, 0] [32, 128], range1 [32, 0] [32, 128], bank0 = 2, bank1 = 3, space = 3>>
  %2 = nova.elementwise 1 %0, %1 lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00 : tensor<64x128xf32, #nova.layout<range0 [0, 0] [32, 128], range1 [32, 0] [32, 128], bank0 = 0, bank1 = 1, space = 3>>, tensor<64x128xf32, #nova.layout<range0 [0, 0] [32, 128], range1 [32, 0] [32, 128], bank0 = 2, bank1 = 3, space = 3>> -> tensor<64x128xf32, #nova.layout<range0 [0, 0] [32, 128], range1 [32, 0] [32, 128], bank0 = 4, bank1 = 5, space = 3>>
  nova.store %2, %arg2 [0, 0] : (tensor<64x128xf32, #nova.layout<range0 [0, 0] [32, 128], range1 [32, 0] [32, 128], bank0 = 4, bank1 = 5, space = 3>>, memref<128x128xf32>) -> ()
  return
}

func.func @conv2d(%arg0: memref<1x32x64x128xi8>, %arg1: memref<3x3x128x64xi8>,
                  %arg2: memref<1x32x64x64xf32>) {
  %0 = nova.load %arg0 [0, 0, 0, 0] : memref<1x32x64x128xi8> -> tensor<1x32x64x128xi8, #nova.layout<range0 [0, 0, 0, 0] [1, 32, 64, 128], bank0 = 0, space = 3>>
  %1 = nova.load %arg1 [0, 0, 0, 0] : memref<3x3x128x64xi8> -> tensor<3x3x128x64xi8, #nova.layout<range0 [0, 0, 0, 0] [3, 3, 128, 64], bank0 = 1, space = 3>>
  %2 = nova.conv2d %0, %1 group 1 pad [1, 1, 1, 1] stride [1, 1] dilation [1, 1] : tensor<1x32x64x128xi8, #nova.layout<range0 [0, 0, 0, 0] [1, 32, 64, 128], bank0 = 0, space = 3>>, tensor<3x3x128x64xi8, #nova.layout<range0 [0, 0, 0, 0] [3, 3, 128, 64], bank0 = 1, space = 3>> -> tensor<1x32x64x64xf32, #nova.layout<range0 [0, 0, 0, 0] [1, 32, 64, 64], bank0 = 2, space = 3>>
  nova.store %2, %arg2 [0, 0, 0, 0] : (tensor<1x32x64x64xf32, #nova.layout<range0 [0, 0, 0, 0] [1, 32, 64, 64], bank0 = 2, space = 3>>, memref<1x32x64x64xf32>) -> ()
  return
}

// CHECK-LABEL: func.func @rowwise_layernorm
// CHECK-NEXT: x1.load %arg0 0 [0, 0] [32, 64] space 3 thread 0 : memref<128x64xf32>
// CHECK-NEXT: x1.load %arg0 1 [32, 0] [32, 64] space 3 thread 1 : memref<128x64xf32>
// CHECK-NEXT: x1.load %arg1 2 [0, 0] [1, 64] space 3 thread 0 : memref<1x64xf32>
// CHECK-NEXT: x1.load %arg2 4 [0, 0] [1, 64] space 3 thread 0 : memref<1x64xf32>
// CHECK-NEXT: x1.reduce 0 inp 0 1 out 6 7 [32, 64] axis 1
// CHECK-NEXT: x1.barrier 0
// CHECK-NEXT: x1.broadcast 3 lhs 0 1 rhs 6 7 out 8 9 [32, 64] axis 1 lhs 1.000000e+00 0.000000e+00 rhs 1.562500e-02 0.000000e+00
// CHECK-NEXT: x1.barrier 0
// CHECK-NEXT: x1.square inp 8 9 out 6 7 [32, 64]
// CHECK-NEXT: x1.barrier 0
// CHECK-NEXT: x1.reduce 0 inp 6 7 out 10 11 [32, 64] axis 1
// CHECK-NEXT: x1.barrier 0
// CHECK-NEXT: x1.scalar_fma inp 10 11 out 6 7 [32, 1] {a = 1.562500e-02 : f32, b = 9.99999974E-6 : f32}
// CHECK-NEXT: x1.barrier 0
// CHECK-NEXT: x1.rsqrt inp 6 7 out 10 11 [32, 1]
// CHECK-NEXT: x1.barrier 0
// CHECK-NEXT: x1.broadcast 2 lhs 8 9 rhs 10 11 out 6 7 [32, 64] axis 1 lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00
// CHECK-NEXT: x1.barrier 0
// CHECK-NEXT: x1.broadcast 2 lhs 6 7 rhs 2 2 out 8 9 [32, 64] axis 0 lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00
// CHECK-NEXT: x1.barrier 0
// CHECK-NEXT: x1.broadcast 1 lhs 8 9 rhs 4 4 out 6 7 [32, 64] axis 0 lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00
// CHECK-NEXT: x1.barrier 0
// CHECK-NEXT: x1.store %arg3 6 [0, 0] [32, 64] space 3 thread 0 : memref<128x64xf32>
// CHECK-NEXT: x1.store %arg3 7 [32, 0] [32, 64] space 3 thread 1 : memref<128x64xf32>
// CHECK-NEXT: x1.load %arg0 8 [64, 0] [32, 64] space 3 thread 0 : memref<128x64xf32>
// CHECK-NEXT: x1.load %arg0 9 [96, 0] [32, 64] space 3 thread 1 : memref<128x64xf32>
// CHECK-NEXT: x1.reduce 0 inp 8 9 out 10 11 [32, 64] axis 1
// CHECK-NEXT: x1.barrier 0
// CHECK-NEXT: x1.broadcast 3 lhs 8 9 rhs 10 11 out 12 13 [32, 64] axis 1 lhs 1.000000e+00 0.000000e+00 rhs 1.562500e-02 0.000000e+00
// CHECK-NEXT: x1.barrier 0
// CHECK-NEXT: x1.square inp 12 13 out 8 9 [32, 64]
// CHECK-NEXT: x1.barrier 0
// CHECK-NEXT: x1.reduce 0 inp 8 9 out 10 11 [32, 64] axis 1
// CHECK-NEXT: x1.barrier 0
// CHECK-NEXT: x1.scalar_fma inp 10 11 out 8 9 [32, 1] {a = 1.562500e-02 : f32, b = 9.99999974E-6 : f32}
// CHECK-NEXT: x1.barrier 0
// CHECK-NEXT: x1.rsqrt inp 8 9 out 10 11 [32, 1]
// CHECK-NEXT: x1.barrier 0
// CHECK-NEXT: x1.broadcast 2 lhs 12 13 rhs 10 11 out 8 9 [32, 64] axis 1 lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00
// CHECK-NEXT: x1.barrier 0
// CHECK-NEXT: x1.broadcast 2 lhs 8 9 rhs 2 2 out 10 11 [32, 64] axis 0 lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00
// CHECK-NEXT: x1.barrier 0
// CHECK-NEXT: x1.broadcast 1 lhs 10 11 rhs 4 4 out 2 3 [32, 64] axis 0 lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00
// CHECK-NEXT: x1.barrier 0
// CHECK-NEXT: x1.store %arg3 2 [64, 0] [32, 64] space 3 thread 0 : memref<128x64xf32>
// CHECK-NEXT: x1.store %arg3 3 [96, 0] [32, 64] space 3 thread 1 : memref<128x64xf32>
// CHECK-NEXT: x1.barrier 1
// CHECK-NEXT: return

// CHECK-LABEL: func.func @column_reduce
// CHECK: x1.load %arg0 0 [0, 0] [32, 64] space 3 thread 0 : memref<64x64xf32>
// CHECK: x1.load %arg0 1 [32, 0] [32, 64] space 3 thread 1 : memref<64x64xf32>
// CHECK-NEXT: x1.reduce 0 inp 0 1 out 2 2 [32, 64] axis 0
// CHECK-NEXT: return

// CHECK-LABEL: func.func @legacy_split_bank_inference
// CHECK: x1.load %arg0 0 [0, 0] [32, 128] space 3 thread 0 : memref<128x128xf32>
// CHECK-NEXT: x1.load %arg0 1 [32, 0] [32, 128] space 3 thread 1 : memref<128x128xf32>
// CHECK-NEXT: return

// CHECK-LABEL: func.func @elementwise_add
// CHECK-NEXT: x1.load %arg0 0 [0, 0] [32, 128] space 3 thread 0 : memref<128x128xf32>
// CHECK-NEXT: x1.load %arg0 1 [32, 0] [32, 128] space 3 thread 1 : memref<128x128xf32>
// CHECK-NEXT: x1.load %arg1 2 [0, 0] [32, 128] space 3 thread 0 : memref<128x128xf32>
// CHECK-NEXT: x1.load %arg1 3 [32, 0] [32, 128] space 3 thread 1 : memref<128x128xf32>
// CHECK-NEXT: x1.elementwise 1 lhs 0 1 rhs 2 3 out 4 5 [32, 128] lhs 1.000000e+00 0.000000e+00 rhs 1.000000e+00 0.000000e+00
// CHECK-NEXT: x1.store %arg2 4 [0, 0] [32, 128] space 3 thread 0 : memref<128x128xf32>
// CHECK-NEXT: x1.store %arg2 5 [32, 0] [32, 128] space 3 thread 1 : memref<128x128xf32>
// CHECK-NEXT: return

// CHECK-LABEL: func.func @conv2d
// CHECK-NEXT: x1.load %arg0 0 [0, 0, 0, 0] [1, 32, 64, 128] space 3 thread 0 : memref<1x32x64x128xi8>
// CHECK-NEXT: x1.load %arg1 1 [0, 0, 0, 0] [3, 3, 128, 64] space 3 thread 0 : memref<3x3x128x64xi8>
// CHECK-NEXT: x1.conv2d inp 0 filter 1 out 2 input [1, 32, 64, 128] kernel [3, 3, 128, 64] result [1, 32, 64, 64] group 1 pad [1, 1, 1, 1] stride [1, 1] dilation [1, 1]
// CHECK-NEXT: x1.store %arg2 2 [0, 0, 0, 0] [1, 32, 64, 64] space 3 thread 0 : memref<1x32x64x64xf32>
// CHECK-NEXT: return
