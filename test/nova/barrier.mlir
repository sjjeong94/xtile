// RUN: xt-opt --nova-barrier %s | FileCheck %s
// RUN: xt-opt --nova-barrier %s | xt-opt --nova-barrier | FileCheck %s --check-prefix=IDEMPOTENT

func.func @insert_after_compute_and_before_return(%src: memref<16x16xf32>, %dst: memref<16x16xf32>) {
  %0 = nova.load(%src) {start = array<i64: 0, 0>} : memref<16x16xf32> -> tensor<16x16xf32>
  %1 = nova.square(%0) : tensor<16x16xf32> -> tensor<16x16xf32>
  nova.store(%1, %dst) {start = array<i64: 0, 0>} : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
  nova.free(%1) : tensor<16x16xf32>
  func.return
}

func.func @skip_existing_barriers(%src: memref<16x16xf32>, %dst: memref<16x16xf32>) {
  %0 = nova.load(%src) {start = array<i64: 0, 0>} : memref<16x16xf32> -> tensor<16x16xf32>
  %1 = nova.square(%0) : tensor<16x16xf32> -> tensor<16x16xf32>
  nova.barrier() {mode = 0 : i32}
  nova.store(%1, %dst) {start = array<i64: 0, 0>} : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
  nova.barrier() {mode = 1 : i32}
  func.return
}

// CHECK-LABEL: func.func @insert_after_compute_and_before_return
// CHECK: %[[LOAD:.*]] = nova.load(%arg0) {start = array<i64: 0, 0>} : memref<16x16xf32> -> tensor<16x16xf32>
// CHECK: %[[SQUARE:.*]] = nova.square(%[[LOAD]]) : tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT: nova.barrier() {mode = 0 : i32}
// CHECK-NEXT: nova.store(%[[SQUARE]], %arg1) {start = array<i64: 0, 0>} : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
// CHECK-NEXT: nova.free(%[[SQUARE]]) : tensor<16x16xf32>
// CHECK-NEXT: nova.barrier() {mode = 1 : i32}
// CHECK-NEXT: return

// CHECK-LABEL: func.func @skip_existing_barriers
// CHECK: %[[LOAD2:.*]] = nova.load(%arg0) {start = array<i64: 0, 0>} : memref<16x16xf32> -> tensor<16x16xf32>
// CHECK: %[[SQUARE2:.*]] = nova.square(%[[LOAD2]]) : tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT: nova.barrier() {mode = 0 : i32}
// CHECK-NEXT: nova.store(%[[SQUARE2]], %arg1) {start = array<i64: 0, 0>} : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
// CHECK-NEXT: nova.barrier() {mode = 1 : i32}
// CHECK-NEXT: return

// IDEMPOTENT-LABEL: func.func @insert_after_compute_and_before_return
// IDEMPOTENT: %[[LOAD3:.*]] = nova.load(%arg0) {start = array<i64: 0, 0>} : memref<16x16xf32> -> tensor<16x16xf32>
// IDEMPOTENT: %[[SQUARE3:.*]] = nova.square(%[[LOAD3]]) : tensor<16x16xf32> -> tensor<16x16xf32>
// IDEMPOTENT-NEXT: nova.barrier() {mode = 0 : i32}
// IDEMPOTENT-NEXT: nova.store(%[[SQUARE3]], %arg1) {start = array<i64: 0, 0>} : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
// IDEMPOTENT-NEXT: nova.free(%[[SQUARE3]]) : tensor<16x16xf32>
// IDEMPOTENT-NEXT: nova.barrier() {mode = 1 : i32}
// IDEMPOTENT-NEXT: return

// IDEMPOTENT-LABEL: func.func @skip_existing_barriers
// IDEMPOTENT: %[[LOAD4:.*]] = nova.load(%arg0) {start = array<i64: 0, 0>} : memref<16x16xf32> -> tensor<16x16xf32>
// IDEMPOTENT: %[[SQUARE4:.*]] = nova.square(%[[LOAD4]]) : tensor<16x16xf32> -> tensor<16x16xf32>
// IDEMPOTENT-NEXT: nova.barrier() {mode = 0 : i32}
// IDEMPOTENT-NEXT: nova.store(%[[SQUARE4]], %arg1) {start = array<i64: 0, 0>} : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
// IDEMPOTENT-NEXT: nova.barrier() {mode = 1 : i32}
// IDEMPOTENT-NEXT: return
