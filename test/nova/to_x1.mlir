// RUN: xt-opt --nova-to-x1 %s | FileCheck %s

func.func @barrier_only() {
  nova.barrier 1
  func.return
}

// CHECK-LABEL: func.func @barrier_only
// CHECK: x1.barrier {mode = 1}
// CHECK-NEXT: return

func.func @rowwise_softmax(%arg0: memref<128x64xf32>,
                           %arg1: memref<128x64xf32>) attributes {xt.double_buffering = 1 : i32, xt.grid = array<i32: 2, 1, 1>} {
  %0 = nova.load %arg0 [0, 0] : memref<128x64xf32> -> tensor<64x64xf32, {bank0 = 0 : i64, bank1 = 1 : i64, space = 3 : i64, threading = 32 : i64}>
  %1 = nova.reduce 1 %0 {axis = 1 : i64} : tensor<64x64xf32, {bank0 = 0 : i64, bank1 = 1 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x1xf32, {bank0 = 2 : i64, bank1 = 3 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %2 = nova.broadcast 3 %0, %1 {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<64x64xf32, {bank0 = 0 : i64, bank1 = 1 : i64, space = 3 : i64, threading = 32 : i64}>, tensor<64x1xf32, {bank0 = 2 : i64, bank1 = 3 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x64xf32, {bank0 = 4 : i64, bank1 = 5 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %3 = nova.exp %2 : tensor<64x64xf32, {bank0 = 4 : i64, bank1 = 5 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x64xf32, {bank0 = 2 : i64, space = 3 : i64}>
  nova.barrier 0
  %4 = nova.reduce 0 %3 {axis = 1 : i64} : tensor<64x64xf32, {bank0 = 2 : i64, space = 3 : i64}> -> tensor<64x1xf32, {bank0 = 4 : i64, space = 3 : i64}>
  nova.barrier 0
  %5 = nova.reciprocal %4 : tensor<64x1xf32, {bank0 = 4 : i64, space = 3 : i64}> -> tensor<64x1xf32, {bank0 = 6 : i64, space = 3 : i64}>
  nova.barrier 0
  %6 = nova.broadcast 2 %3, %5 {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<64x64xf32, {bank0 = 2 : i64, space = 3 : i64}>, tensor<64x1xf32, {bank0 = 6 : i64, space = 3 : i64}> -> tensor<64x64xf32, {bank0 = 4 : i64, space = 3 : i64}>
  nova.barrier 0
  nova.store %6, %arg1 [0, 0] : (tensor<64x64xf32, {bank0 = 4 : i64, space = 3 : i64}>, memref<128x64xf32>) -> ()
  %7 = nova.load %arg0 [64, 0] : memref<128x64xf32> -> tensor<64x64xf32, {bank0 = 2 : i64, bank1 = 3 : i64, space = 3 : i64, threading = 32 : i64}>
  %8 = nova.reduce 1 %7 {axis = 1 : i64} : tensor<64x64xf32, {bank0 = 2 : i64, bank1 = 3 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x1xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %9 = nova.broadcast 3 %7, %8 {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<64x64xf32, {bank0 = 2 : i64, bank1 = 3 : i64, space = 3 : i64, threading = 32 : i64}>, tensor<64x1xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %10 = nova.exp %9 : tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x64xf32, {bank0 = 2 : i64, space = 3 : i64}>
  nova.barrier 0
  %11 = nova.reduce 0 %10 {axis = 1 : i64} : tensor<64x64xf32, {bank0 = 2 : i64, space = 3 : i64}> -> tensor<64x1xf32, {bank0 = 6 : i64, space = 3 : i64}>
  nova.barrier 0
  %12 = nova.reciprocal %11 : tensor<64x1xf32, {bank0 = 6 : i64, space = 3 : i64}> -> tensor<64x1xf32, {bank0 = 8 : i64, space = 3 : i64}>
  nova.barrier 0
  %13 = nova.broadcast 2 %10, %12 {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<64x64xf32, {bank0 = 2 : i64, space = 3 : i64}>, tensor<64x1xf32, {bank0 = 8 : i64, space = 3 : i64}> -> tensor<64x64xf32, {bank0 = 6 : i64, space = 3 : i64}>
  nova.barrier 0
  nova.store %13, %arg1 [64, 0] : (tensor<64x64xf32, {bank0 = 6 : i64, space = 3 : i64}>, memref<128x64xf32>) -> ()
  nova.barrier 1
  return
}

// CHECK-LABEL: func.func @rowwise_softmax
// CHECK-NEXT: x1.load %arg0 {bank = 0, space = 3, thread = 0, start = [0, 0], shape = [32, 64]} : memref<128x64xf32>
// CHECK-NEXT: x1.load %arg0 {bank = 1, space = 3, thread = 1, start = [32, 0], shape = [32, 64]} : memref<128x64xf32>
// CHECK-NEXT: x1.reduce {inp0 = 0, inp1 = 1, out0 = 2, out1 = 3, shape = [32, 64], axis = 1, mode = 1}
// CHECK-NEXT: x1.barrier {mode = 0}
// CHECK-NEXT: x1.broadcast {lhs0 = 0, lhs1 = 1, rhs0 = 2, rhs1 = 3, out0 = 4, out1 = 5, shape = [32, 64], axis = 1, lhs_a = 1.000000e+00, lhs_b = 0.000000e+00, rhs_a = 1.000000e+00, rhs_b = 0.000000e+00, mode = 3}
// CHECK-NEXT: x1.barrier {mode = 0}
// CHECK-NEXT: x1.exp {inp0 = 4, inp1 = 5, out0 = 2, out1 = 3, shape = [32, 64]}
// CHECK-NEXT: x1.barrier {mode = 0}
// CHECK-NEXT: x1.reduce {inp0 = 2, inp1 = 3, out0 = 4, out1 = 5, shape = [32, 64], axis = 1, mode = 0}
// CHECK-NEXT: x1.barrier {mode = 0}
// CHECK-NEXT: x1.reciprocal {inp0 = 4, inp1 = 5, out0 = 6, out1 = 7, shape = [64, 1]}
// CHECK-NEXT: x1.barrier {mode = 0}
// CHECK-NEXT: x1.broadcast {lhs0 = 2, lhs1 = 3, rhs0 = 6, rhs1 = 7, out0 = 4, out1 = 5, shape = [32, 64], axis = 1, lhs_a = 1.000000e+00, lhs_b = 0.000000e+00, rhs_a = 1.000000e+00, rhs_b = 0.000000e+00, mode = 2}
// CHECK-NEXT: x1.barrier {mode = 0}
// CHECK-NEXT: x1.store %arg1 {bank = 4, space = 3, thread = 0, start = [0, 0], shape = [32, 64]} : memref<128x64xf32>
// CHECK-NEXT: x1.store %arg1 {bank = 5, space = 3, thread = 1, start = [32, 0], shape = [32, 64]} : memref<128x64xf32>
// CHECK-NEXT: x1.load %arg0 {bank = 2, space = 3, thread = 0, start = [64, 0], shape = [32, 64]} : memref<128x64xf32>
// CHECK-NEXT: x1.load %arg0 {bank = 3, space = 3, thread = 1, start = [96, 0], shape = [32, 64]} : memref<128x64xf32>
// CHECK-NEXT: x1.reduce {inp0 = 2, inp1 = 3, out0 = 6, out1 = 7, shape = [32, 64], axis = 1, mode = 1}
// CHECK-NEXT: x1.barrier {mode = 0}
// CHECK-NEXT: x1.broadcast {lhs0 = 2, lhs1 = 3, rhs0 = 6, rhs1 = 7, out0 = 8, out1 = 9, shape = [32, 64], axis = 1, lhs_a = 1.000000e+00, lhs_b = 0.000000e+00, rhs_a = 1.000000e+00, rhs_b = 0.000000e+00, mode = 3}
// CHECK-NEXT: x1.barrier {mode = 0}
// CHECK-NEXT: x1.exp {inp0 = 8, inp1 = 9, out0 = 2, out1 = 3, shape = [32, 64]}
// CHECK-NEXT: x1.barrier {mode = 0}
// CHECK-NEXT: x1.reduce {inp0 = 2, inp1 = 3, out0 = 6, out1 = 7, shape = [32, 64], axis = 1, mode = 0}
// CHECK-NEXT: x1.barrier {mode = 0}
// CHECK-NEXT: x1.reciprocal {inp0 = 6, inp1 = 7, out0 = 8, out1 = 9, shape = [64, 1]}
// CHECK-NEXT: x1.barrier {mode = 0}
// CHECK-NEXT: x1.broadcast {lhs0 = 2, lhs1 = 3, rhs0 = 8, rhs1 = 9, out0 = 6, out1 = 7, shape = [32, 64], axis = 1, lhs_a = 1.000000e+00, lhs_b = 0.000000e+00, rhs_a = 1.000000e+00, rhs_b = 0.000000e+00, mode = 2}
// CHECK-NEXT: x1.barrier {mode = 0}
// CHECK-NEXT: x1.store %arg1 {bank = 6, space = 3, thread = 0, start = [64, 0], shape = [32, 64]} : memref<128x64xf32>
// CHECK-NEXT: x1.store %arg1 {bank = 7, space = 3, thread = 1, start = [96, 0], shape = [32, 64]} : memref<128x64xf32>
// CHECK-NEXT: x1.barrier {mode = 1}
// CHECK-NEXT: return

func.func @rowwise_layernorm(%arg0: memref<128x64xf32>, %arg1: memref<1x64xf32>,
                             %arg2: memref<1x64xf32>, %arg3: memref<128x64xf32>) attributes {xt.double_buffering = 1 : i32, xt.grid = array<i32: 2, 1, 1>} {
  %0 = nova.load %arg0 [0, 0] : memref<128x64xf32> -> tensor<64x64xf32, {bank0 = 0 : i64, bank1 = 1 : i64, space = 3 : i64, threading = 32 : i64}>
  %1 = nova.load %arg1 [0, 0] {shared = 1 : i64} : memref<1x64xf32> -> tensor<1x64xf32, {bank0 = 2 : i64, space = 3 : i64, threading = 1 : i64}>
  %2 = nova.load %arg2 [0, 0] {shared = 1 : i64} : memref<1x64xf32> -> tensor<1x64xf32, {bank0 = 4 : i64, space = 3 : i64, threading = 1 : i64}>
  %3 = nova.reduce 0 %0 {axis = 1 : i64} : tensor<64x64xf32, {bank0 = 0 : i64, bank1 = 1 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x1xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %4 = nova.broadcast 3 %0, %3 {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, rhs_a = 1.562500e-02 : f32, rhs_b = 0.000000e+00 : f32} : tensor<64x64xf32, {bank0 = 0 : i64, bank1 = 1 : i64, space = 3 : i64, threading = 32 : i64}>, tensor<64x1xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %5 = nova.square %4 : tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x64xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %6 = nova.reduce 0 %5 {axis = 1 : i64} : tensor<64x64xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x1xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %7 = nova.scalar_fma %6, 1.562500e-02, 9.99999974E-6 : tensor<64x1xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x1xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %8 = nova.rsqrt %7 : tensor<64x1xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x1xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %9 = nova.broadcast 2 %4, %8 {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}>, tensor<64x1xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x64xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %10 = nova.broadcast 2 %9, %1 {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<64x64xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}>, tensor<1x64xf32, {bank0 = 2 : i64, space = 3 : i64, threading = 1 : i64}> -> tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %11 = nova.broadcast 1 %10, %2 {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}>, tensor<1x64xf32, {bank0 = 4 : i64, space = 3 : i64, threading = 1 : i64}> -> tensor<64x64xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  nova.store %11, %arg3 [0, 0] : (tensor<64x64xf32, {bank0 = 6 : i64, bank1 = 7 : i64, space = 3 : i64, threading = 32 : i64}>, memref<128x64xf32>) -> ()
  %12 = nova.load %arg0 [64, 0] : memref<128x64xf32> -> tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}>
  %13 = nova.reduce 0 %12 {axis = 1 : i64} : tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x1xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %14 = nova.broadcast 3 %12, %13 {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, rhs_a = 1.562500e-02 : f32, rhs_b = 0.000000e+00 : f32} : tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}>, tensor<64x1xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x64xf32, {bank0 = 12 : i64, bank1 = 13 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %15 = nova.square %14 : tensor<64x64xf32, {bank0 = 12 : i64, bank1 = 13 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %16 = nova.reduce 0 %15 {axis = 1 : i64} : tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x1xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %17 = nova.scalar_fma %16, 1.562500e-02, 9.99999974E-6 : tensor<64x1xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x1xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %18 = nova.rsqrt %17 : tensor<64x1xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x1xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %19 = nova.broadcast 2 %14, %18 {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<64x64xf32, {bank0 = 12 : i64, bank1 = 13 : i64, space = 3 : i64, threading = 32 : i64}>, tensor<64x1xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}> -> tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %20 = nova.broadcast 2 %19, %1 {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<64x64xf32, {bank0 = 8 : i64, bank1 = 9 : i64, space = 3 : i64, threading = 32 : i64}>, tensor<1x64xf32, {bank0 = 2 : i64, space = 3 : i64, threading = 1 : i64}> -> tensor<64x64xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}>
  nova.barrier 0
  %21 = nova.broadcast 1 %20, %2 {lhs_a = 1.000000e+00 : f32, lhs_b = 0.000000e+00 : f32, rhs_a = 1.000000e+00 : f32, rhs_b = 0.000000e+00 : f32} : tensor<64x64xf32, {bank0 = 10 : i64, bank1 = 11 : i64, space = 3 : i64, threading = 32 : i64}>, tensor<1x64xf32, {bank0 = 4 : i64, space = 3 : i64, threading = 1 : i64}> -> tensor<64x64xf32, {bank0 = 2 : i64, bank1 = 3 : i64, space = 3 : i64, threading = 32 : i64}>
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

// CHECK-LABEL: func.func @rowwise_layernorm
// CHECK-NEXT: x1.load %arg0 {bank = 0, space = 3, thread = 0, start = [0, 0], shape = [32, 64]} : memref<128x64xf32>
// CHECK-NEXT: x1.load %arg0 {bank = 1, space = 3, thread = 1, start = [32, 0], shape = [32, 64]} : memref<128x64xf32>
// CHECK-NEXT: x1.load %arg1 {bank = 2, space = 3, thread = 0, start = [0, 0], shape = [1, 64]} : memref<1x64xf32>
// CHECK-NEXT: x1.load %arg2 {bank = 4, space = 3, thread = 0, start = [0, 0], shape = [1, 64]} : memref<1x64xf32>
// CHECK-NEXT: x1.reduce {inp0 = 0, inp1 = 1, out0 = 6, out1 = 7, shape = [32, 64], axis = 1, mode = 0}
// CHECK-NEXT: x1.barrier {mode = 0}
// CHECK-NEXT: x1.broadcast {lhs0 = 0, lhs1 = 1, rhs0 = 6, rhs1 = 7, out0 = 8, out1 = 9, shape = [32, 64], axis = 1, lhs_a = 1.000000e+00, lhs_b = 0.000000e+00, rhs_a = 1.562500e-02, rhs_b = 0.000000e+00, mode = 3}
// CHECK-NEXT: x1.barrier {mode = 0}
// CHECK-NEXT: x1.square {inp0 = 8, inp1 = 9, out0 = 6, out1 = 7, shape = [32, 64]}
// CHECK-NEXT: x1.barrier {mode = 0}
// CHECK-NEXT: x1.reduce {inp0 = 6, inp1 = 7, out0 = 10, out1 = 11, shape = [32, 64], axis = 1, mode = 0}
// CHECK-NEXT: x1.barrier {mode = 0}
// CHECK-NEXT: x1.scalar_fma {inp0 = 10, inp1 = 11, out0 = 6, out1 = 7, shape = [32, 1], a = 1.562500e-02, b = 9.99999974E-6}
// CHECK-NEXT: x1.barrier {mode = 0}
// CHECK-NEXT: x1.rsqrt {inp0 = 6, inp1 = 7, out0 = 10, out1 = 11, shape = [32, 1]}
// CHECK-NEXT: x1.barrier {mode = 0}
// CHECK-NEXT: x1.broadcast {lhs0 = 8, lhs1 = 9, rhs0 = 10, rhs1 = 11, out0 = 6, out1 = 7, shape = [32, 64], axis = 1, lhs_a = 1.000000e+00, lhs_b = 0.000000e+00, rhs_a = 1.000000e+00, rhs_b = 0.000000e+00, mode = 2}
// CHECK-NEXT: x1.barrier {mode = 0}
// CHECK-NEXT: x1.broadcast {lhs0 = 6, lhs1 = 7, rhs0 = 2, rhs1 = 2, out0 = 8, out1 = 9, shape = [32, 64], axis = 0, lhs_a = 1.000000e+00, lhs_b = 0.000000e+00, rhs_a = 1.000000e+00, rhs_b = 0.000000e+00, mode = 2}
// CHECK-NEXT: x1.barrier {mode = 0}
// CHECK-NEXT: x1.broadcast {lhs0 = 8, lhs1 = 9, rhs0 = 4, rhs1 = 4, out0 = 6, out1 = 7, shape = [32, 64], axis = 0, lhs_a = 1.000000e+00, lhs_b = 0.000000e+00, rhs_a = 1.000000e+00, rhs_b = 0.000000e+00, mode = 1}
// CHECK-NEXT: x1.barrier {mode = 0}
// CHECK-NEXT: x1.store %arg3 {bank = 6, space = 3, thread = 0, start = [0, 0], shape = [32, 64]} : memref<128x64xf32>
// CHECK-NEXT: x1.store %arg3 {bank = 7, space = 3, thread = 1, start = [32, 0], shape = [32, 64]} : memref<128x64xf32>
// CHECK-NEXT: x1.load %arg0 {bank = 8, space = 3, thread = 0, start = [64, 0], shape = [32, 64]} : memref<128x64xf32>
// CHECK-NEXT: x1.load %arg0 {bank = 9, space = 3, thread = 1, start = [96, 0], shape = [32, 64]} : memref<128x64xf32>
// CHECK-NEXT: x1.reduce {inp0 = 8, inp1 = 9, out0 = 10, out1 = 11, shape = [32, 64], axis = 1, mode = 0}
// CHECK-NEXT: x1.barrier {mode = 0}
// CHECK-NEXT: x1.broadcast {lhs0 = 8, lhs1 = 9, rhs0 = 10, rhs1 = 11, out0 = 12, out1 = 13, shape = [32, 64], axis = 1, lhs_a = 1.000000e+00, lhs_b = 0.000000e+00, rhs_a = 1.562500e-02, rhs_b = 0.000000e+00, mode = 3}
// CHECK-NEXT: x1.barrier {mode = 0}
// CHECK-NEXT: x1.square {inp0 = 12, inp1 = 13, out0 = 8, out1 = 9, shape = [32, 64]}
// CHECK-NEXT: x1.barrier {mode = 0}
// CHECK-NEXT: x1.reduce {inp0 = 8, inp1 = 9, out0 = 10, out1 = 11, shape = [32, 64], axis = 1, mode = 0}
// CHECK-NEXT: x1.barrier {mode = 0}
// CHECK-NEXT: x1.scalar_fma {inp0 = 10, inp1 = 11, out0 = 8, out1 = 9, shape = [32, 1], a = 1.562500e-02, b = 9.99999974E-6}
// CHECK-NEXT: x1.barrier {mode = 0}
// CHECK-NEXT: x1.rsqrt {inp0 = 8, inp1 = 9, out0 = 10, out1 = 11, shape = [32, 1]}
// CHECK-NEXT: x1.barrier {mode = 0}
// CHECK-NEXT: x1.broadcast {lhs0 = 12, lhs1 = 13, rhs0 = 10, rhs1 = 11, out0 = 8, out1 = 9, shape = [32, 64], axis = 1, lhs_a = 1.000000e+00, lhs_b = 0.000000e+00, rhs_a = 1.000000e+00, rhs_b = 0.000000e+00, mode = 2}
// CHECK-NEXT: x1.barrier {mode = 0}
// CHECK-NEXT: x1.broadcast {lhs0 = 8, lhs1 = 9, rhs0 = 2, rhs1 = 2, out0 = 10, out1 = 11, shape = [32, 64], axis = 0, lhs_a = 1.000000e+00, lhs_b = 0.000000e+00, rhs_a = 1.000000e+00, rhs_b = 0.000000e+00, mode = 2}
// CHECK-NEXT: x1.barrier {mode = 0}
// CHECK-NEXT: x1.broadcast {lhs0 = 10, lhs1 = 11, rhs0 = 4, rhs1 = 4, out0 = 2, out1 = 3, shape = [32, 64], axis = 0, lhs_a = 1.000000e+00, lhs_b = 0.000000e+00, rhs_a = 1.000000e+00, rhs_b = 0.000000e+00, mode = 1}
// CHECK-NEXT: x1.barrier {mode = 0}
// CHECK-NEXT: x1.store %arg3 {bank = 2, space = 3, thread = 0, start = [64, 0], shape = [32, 64]} : memref<128x64xf32>
// CHECK-NEXT: x1.store %arg3 {bank = 3, space = 3, thread = 1, start = [96, 0], shape = [32, 64]} : memref<128x64xf32>
// CHECK-NEXT: x1.barrier {mode = 1}
// CHECK-NEXT: return

// CHECK-LABEL: func.func @column_reduce
// CHECK: x1.load %arg0 {bank = 0, space = 3, thread = 0, start = [0, 0], shape = [32, 64]} : memref<64x64xf32>
// CHECK: x1.load %arg0 {bank = 1, space = 3, thread = 1, start = [32, 0], shape = [32, 64]} : memref<64x64xf32>
// CHECK-NEXT: x1.reduce {inp0 = 0, inp1 = 1, out0 = 2, out1 = 2, shape = [32, 64], axis = 0, mode = 0}
// CHECK-NEXT: return

// CHECK-LABEL: func.func @legacy_split_bank_inference
// CHECK: x1.load %arg0 {bank = 0, space = 3, thread = 0, start = [0, 0], shape = [32, 128]} : memref<128x128xf32>
// CHECK-NEXT: x1.load %arg0 {bank = 1, space = 3, thread = 1, start = [32, 0], shape = [32, 128]} : memref<128x128xf32>
// CHECK-NEXT: return
