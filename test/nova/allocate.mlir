// RUN: xt-opt --nova-allocate %s | FileCheck %s

func.func @allocate_basic(%src: memref<64x16xf32>, %dst: memref<64x16xf32>) {
  %0 = nova.load %src [0, 0] : memref<64x16xf32> -> tensor<16x16xf32>
  %1 = nova.square %0 : tensor<16x16xf32> -> tensor<16x16xf32>
  nova.store %1, %dst [0, 0] : (tensor<16x16xf32>, memref<64x16xf32>) -> ()
  %2 = nova.load %src [1, 0] : memref<64x16xf32> -> tensor<16x16xf32>
  %3 = nova.square %2 : tensor<16x16xf32> -> tensor<16x16xf32>
  %4 = arith.constant dense<1.0> : tensor<1x1xf32>
  nova.store %3, %dst [1, 0] : (tensor<16x16xf32>, memref<64x16xf32>) -> ()
  func.return
}

func.func @allocate_keep_alive_extends_liveness(%src: memref<64x16xf32>, %dst: memref<64x16xf32>) {
  %0 = nova.load %src [0, 0] : memref<64x16xf32> -> tensor<16x16xf32>
  %1 = nova.square %0 : tensor<16x16xf32> -> tensor<16x16xf32>
  nova.store %1, %dst [0, 0] : (tensor<16x16xf32>, memref<64x16xf32>) -> ()
  %2 = nova.load %src [1, 0] : memref<64x16xf32> -> tensor<16x16xf32>
  %3 = nova.square %2 : tensor<16x16xf32> -> tensor<16x16xf32>
  nova.store %3, %dst [1, 0] : (tensor<16x16xf32>, memref<64x16xf32>) -> ()
  nova.keep_alive %1, %3 : tensor<16x16xf32>, tensor<16x16xf32>
  func.return
}

func.func @allocate_multi_bank(%a: tensor<70000xf32>, %b: tensor<16x16xf32>, %dst: memref<16x16xf32>) {
  %0 = nova.square %a : tensor<70000xf32> -> tensor<70000xf32>
  %1 = nova.square %b : tensor<16x16xf32> -> tensor<16x16xf32>
  nova.store %1, %dst [0, 0] : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
  func.return
}

func.func @allocate_space_assignment(%lhs_src: memref<16x32xf32>, %rhs_src: memref<32x8xf32>, %scale_src: memref<16x8xf32>, %bias_src: memref<16x8xf32>) {
  %lhs = nova.load %lhs_src [0, 0] : memref<16x32xf32> -> tensor<16x32xf32>
  %rhs = nova.load %rhs_src [0, 0] : memref<32x8xf32> -> tensor<32x8xf32>
  %scale = nova.load %scale_src [0, 0] : memref<16x8xf32> -> tensor<16x8xf32>
  %bias = nova.load %bias_src [0, 0] : memref<16x8xf32> -> tensor<16x8xf32>
  %result = nova.matmul %lhs, %rhs, %scale, %bias : tensor<16x32xf32>, tensor<32x8xf32>, tensor<16x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
  func.return
}

func.func @allocate_split_banks(%arg0: tensor<64x128xf32, #nova.tensor_layout<range0 [0, 0] [32, 128], range1 [32, 0] [32, 128]>>) {
  %0 = nova.square %arg0 : tensor<64x128xf32, #nova.tensor_layout<range0 [0, 0] [32, 128], range1 [32, 0] [32, 128]>> -> tensor<64x128xf32, #nova.tensor_layout<range0 [0, 0] [32, 128], range1 [32, 0] [32, 128]>>
  func.return
}

// CHECK-LABEL: func.func @allocate_basic
// CHECK: %[[LOAD0:.*]] = nova.load %arg0 [0, 0] : memref<64x16xf32> -> tensor<16x16xf32, #nova.tensor_layout<bank0 = 0, space = 3>>
// CHECK: %[[SQUARE0:.*]] = nova.square %[[LOAD0]] : tensor<16x16xf32, #nova.tensor_layout<bank0 = 0, space = 3>> -> tensor<16x16xf32, #nova.tensor_layout<bank0 = 1, space = 3>>
// CHECK: nova.store %[[SQUARE0]], %arg1 [0, 0] : (tensor<16x16xf32, #nova.tensor_layout<bank0 = 1, space = 3>>, memref<64x16xf32>) -> ()
// CHECK: %[[LOAD1:.*]] = nova.load %arg0 [1, 0] : memref<64x16xf32> -> tensor<16x16xf32, #nova.tensor_layout<bank0 = 0, space = 3>>
// CHECK: %[[SQUARE1:.*]] = nova.square %[[LOAD1]] : tensor<16x16xf32, #nova.tensor_layout<bank0 = 0, space = 3>> -> tensor<16x16xf32, #nova.tensor_layout<bank0 = 1, space = 3>>
// CHECK: %[[SCALAR:.*]] = arith.constant dense<1.000000e+00> : tensor<1x1xf32>
// CHECK: nova.store %[[SQUARE1]], %arg1 [1, 0] : (tensor<16x16xf32, #nova.tensor_layout<bank0 = 1, space = 3>>, memref<64x16xf32>) -> ()
// CHECK-NOT: nova.free(
// CHECK-LABEL: func.func @allocate_keep_alive_extends_liveness
// CHECK: %[[KL0:.*]] = nova.load %arg0 [0, 0] : memref<64x16xf32> -> tensor<16x16xf32, #nova.tensor_layout<bank0 = 0, space = 3>>
// CHECK: %[[KS0:.*]] = nova.square %[[KL0]] : tensor<16x16xf32, #nova.tensor_layout<bank0 = 0, space = 3>> -> tensor<16x16xf32, #nova.tensor_layout<bank0 = 1, space = 3>>
// CHECK: nova.store %[[KS0]], %arg1 [0, 0] : (tensor<16x16xf32, #nova.tensor_layout<bank0 = 1, space = 3>>, memref<64x16xf32>) -> ()
// CHECK: %[[KL1:.*]] = nova.load %arg0 [1, 0] : memref<64x16xf32> -> tensor<16x16xf32, #nova.tensor_layout<bank0 = 0, space = 3>>
// CHECK: %[[KS1:.*]] = nova.square %[[KL1]] : tensor<16x16xf32, #nova.tensor_layout<bank0 = 0, space = 3>> -> tensor<16x16xf32, #nova.tensor_layout<bank0 = 2, space = 3>>
// CHECK: nova.store %[[KS1]], %arg1 [1, 0] : (tensor<16x16xf32, #nova.tensor_layout<bank0 = 2, space = 3>>, memref<64x16xf32>) -> ()
// CHECK-NOT: nova.keep_alive 
// CHECK-LABEL: func.func @allocate_multi_bank
// CHECK: %[[LARGE:.*]] = nova.square %arg0 : tensor<70000xf32> -> tensor<70000xf32, #nova.tensor_layout<bank0 = 0, space = 3>>
// CHECK: %[[SMALL:.*]] = nova.square %arg1 : tensor<16x16xf32> -> tensor<16x16xf32, #nova.tensor_layout<bank0 = 0, space = 3>>
// CHECK: nova.store %[[SMALL]], %arg2 [0, 0] : (tensor<16x16xf32, #nova.tensor_layout<bank0 = 0, space = 3>>, memref<16x16xf32>) -> ()
// CHECK-NOT: nova.free(
// CHECK-LABEL: func.func @allocate_space_assignment
// CHECK: %[[LHS:.*]] = nova.load %arg0 [0, 0] : memref<16x32xf32> -> tensor<16x32xf32, #nova.tensor_layout<bank0 = 0, space = 3>>
// CHECK: %[[RHS:.*]] = nova.load %arg1 [0, 0] : memref<32x8xf32> -> tensor<32x8xf32, #nova.tensor_layout<bank0 = 1, space = 3>>
// CHECK: %[[SCALE:.*]] = nova.load %arg2 [0, 0] : memref<16x8xf32> -> tensor<16x8xf32, #nova.tensor_layout<bank0 = 0, space = 4>>
// CHECK: %[[BIAS:.*]] = nova.load %arg3 [0, 0] : memref<16x8xf32> -> tensor<16x8xf32, #nova.tensor_layout<bank0 = 0, space = 5>>
// CHECK: %[[RESULT:.*]] = nova.matmul %[[LHS]], %[[RHS]], %[[SCALE]], %[[BIAS]] : tensor<16x32xf32, #nova.tensor_layout<bank0 = 0, space = 3>>, tensor<32x8xf32, #nova.tensor_layout<bank0 = 1, space = 3>>, tensor<16x8xf32, #nova.tensor_layout<bank0 = 0, space = 4>>, tensor<16x8xf32, #nova.tensor_layout<bank0 = 0, space = 5>> -> tensor<16x8xf32, #nova.tensor_layout<bank0 = 2, space = 3>>
// CHECK-NOT: nova.free(
// CHECK-LABEL: func.func @allocate_split_banks
// CHECK: %[[SPLIT:.*]] = nova.square %arg0 : tensor<64x128xf32, #nova.tensor_layout<range0 [0, 0] [32, 128], range1 [32, 0] [32, 128]>> -> tensor<64x128xf32, #nova.tensor_layout<range0 [0, 0] [32, 128], range1 [32, 0] [32, 128], bank0 = 0, bank1 = 1, space = 3>>
// CHECK-NOT: nova.free(
