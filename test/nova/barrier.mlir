// RUN: xt-opt --nova-barrier %s | FileCheck %s
// RUN: xt-opt --nova-barrier %s | xt-opt --nova-barrier | FileCheck %s --check-prefix=IDEMPOTENT

func.func @insert_after_compute_and_before_return(%src: memref<16x16xf32>, %dst: memref<16x16xf32>) {
  %0 = nova.load(%src) {start = [0, 0]} : memref<16x16xf32> -> tensor<16x16xf32>
  %1 = nova.square(%0) : tensor<16x16xf32> -> tensor<16x16xf32>
  nova.store(%1, %dst) {start = [0, 0]} : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
  func.return
}

func.func @skip_existing_barriers(%src: memref<16x16xf32>, %dst: memref<16x16xf32>) {
  %0 = nova.load(%src) {start = [0, 0]} : memref<16x16xf32> -> tensor<16x16xf32>
  %1 = nova.square(%0) : tensor<16x16xf32> -> tensor<16x16xf32>
  nova.barrier() {mode = 0 : i32}
  nova.store(%1, %dst) {start = [0, 0]} : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
  nova.barrier() {mode = 1 : i32}
  func.return
}

func.func @insert_after_store_when_double_buffering_disabled(%src: memref<16x16xf32>, %dst: memref<16x16xf32>) attributes {xt.double_buffering = 0 : i32} {
  %0 = nova.load(%src) {start = [0, 0]} : memref<16x16xf32> -> tensor<16x16xf32>
  nova.store(%0, %dst) {start = [0, 0]} : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
  func.return
}

func.func @insert_after_store_when_double_buffering_unspecified(%src: memref<16x16xf32>, %dst: memref<16x16xf32>) {
  %0 = nova.load(%src) {start = [0, 0]} : memref<16x16xf32> -> tensor<16x16xf32>
  nova.store(%0, %dst) {start = [0, 0]} : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
  func.return
}

func.func @insert_keep_alive_when_double_buffering_enabled(%src: memref<16x16xf32>, %dst: memref<16x16xf32>) attributes {xt.double_buffering = 1 : i32} {
  %0 = nova.load(%src) {start = [0, 0]} : memref<16x16xf32> -> tensor<16x16xf32>
  %1 = nova.square(%0) : tensor<16x16xf32> -> tensor<16x16xf32>
  %2 = nova.exp(%1) : tensor<16x16xf32> -> tensor<16x16xf32>
  nova.store(%2, %dst) {start = [0, 0]} : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
  func.return
}

// CHECK-LABEL: func.func @insert_after_compute_and_before_return
// CHECK: %[[LOAD:.*]] = nova.load(%arg0) {start = [0, 0]} : memref<16x16xf32> -> tensor<16x16xf32>
// CHECK: %[[SQUARE:.*]] = nova.square(%[[LOAD]]) : tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT: nova.barrier() {mode = 0 : i32}
// CHECK-NEXT: nova.store(%[[SQUARE]], %arg1) {start = [0, 0]} : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
// CHECK-NEXT: nova.barrier() {mode = 1 : i32}
// CHECK-NEXT: return

// CHECK-LABEL: func.func @skip_existing_barriers
// CHECK: %[[LOAD2:.*]] = nova.load(%arg0) {start = [0, 0]} : memref<16x16xf32> -> tensor<16x16xf32>
// CHECK: %[[SQUARE2:.*]] = nova.square(%[[LOAD2]]) : tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT: nova.barrier() {mode = 0 : i32}
// CHECK-NEXT: nova.store(%[[SQUARE2]], %arg1) {start = [0, 0]} : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
// CHECK-NEXT: nova.barrier() {mode = 1 : i32}
// CHECK-NEXT: return

// CHECK-LABEL: func.func @insert_after_store_when_double_buffering_disabled
// CHECK: %[[LOAD3:.*]] = nova.load(%arg0) {start = [0, 0]} : memref<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT: nova.store(%[[LOAD3]], %arg1) {start = [0, 0]} : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
// CHECK-NEXT: nova.barrier() {mode = 1 : i32}
// CHECK-NEXT: return

// CHECK-LABEL: func.func @insert_after_store_when_double_buffering_unspecified
// CHECK: %[[LOAD4:.*]] = nova.load(%arg0) {start = [0, 0]} : memref<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT: nova.store(%[[LOAD4]], %arg1) {start = [0, 0]} : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
// CHECK-NEXT: nova.barrier() {mode = 1 : i32}
// CHECK-NEXT: return

// CHECK-LABEL: func.func @insert_keep_alive_when_double_buffering_enabled
// CHECK: %[[LOAD_DB:.*]] = nova.load(%arg0) {start = [0, 0]} : memref<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT: %[[SQUARE_DB:.*]] = nova.square(%[[LOAD_DB]]) : tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT: nova.barrier() {mode = 0 : i32}
// CHECK-NEXT: %[[EXP_DB:.*]] = nova.exp(%[[SQUARE_DB]]) : tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT: nova.keep_alive(%[[LOAD_DB]], %[[SQUARE_DB]]) : tensor<16x16xf32>, tensor<16x16xf32>
// CHECK-NEXT: nova.barrier() {mode = 0 : i32}
// CHECK-NEXT: nova.store(%[[EXP_DB]], %arg1) {start = [0, 0]} : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
// CHECK-NEXT: nova.keep_alive(%[[SQUARE_DB]], %[[EXP_DB]]) : tensor<16x16xf32>, tensor<16x16xf32>
// CHECK-NEXT: nova.barrier() {mode = 1 : i32}
// CHECK-NEXT: return

// IDEMPOTENT-LABEL: func.func @insert_after_compute_and_before_return
// IDEMPOTENT: %[[LOAD5:.*]] = nova.load(%arg0) {start = [0, 0]} : memref<16x16xf32> -> tensor<16x16xf32>
// IDEMPOTENT: %[[SQUARE5:.*]] = nova.square(%[[LOAD5]]) : tensor<16x16xf32> -> tensor<16x16xf32>
// IDEMPOTENT-NEXT: nova.barrier() {mode = 0 : i32}
// IDEMPOTENT-NEXT: nova.store(%[[SQUARE5]], %arg1) {start = [0, 0]} : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
// IDEMPOTENT-NEXT: nova.barrier() {mode = 1 : i32}
// IDEMPOTENT-NEXT: return

// IDEMPOTENT-LABEL: func.func @skip_existing_barriers
// IDEMPOTENT: %[[LOAD6:.*]] = nova.load(%arg0) {start = [0, 0]} : memref<16x16xf32> -> tensor<16x16xf32>
// IDEMPOTENT: %[[SQUARE6:.*]] = nova.square(%[[LOAD6]]) : tensor<16x16xf32> -> tensor<16x16xf32>
// IDEMPOTENT-NEXT: nova.barrier() {mode = 0 : i32}
// IDEMPOTENT-NEXT: nova.store(%[[SQUARE6]], %arg1) {start = [0, 0]} : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
// IDEMPOTENT-NEXT: nova.barrier() {mode = 1 : i32}
// IDEMPOTENT-NEXT: return

// IDEMPOTENT-LABEL: func.func @insert_after_store_when_double_buffering_disabled
// IDEMPOTENT: %[[LOAD7:.*]] = nova.load(%arg0) {start = [0, 0]} : memref<16x16xf32> -> tensor<16x16xf32>
// IDEMPOTENT-NEXT: nova.store(%[[LOAD7]], %arg1) {start = [0, 0]} : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
// IDEMPOTENT-NEXT: nova.barrier() {mode = 1 : i32}
// IDEMPOTENT-NEXT: return

// IDEMPOTENT-LABEL: func.func @insert_after_store_when_double_buffering_unspecified
// IDEMPOTENT: %[[LOAD8:.*]] = nova.load(%arg0) {start = [0, 0]} : memref<16x16xf32> -> tensor<16x16xf32>
// IDEMPOTENT-NEXT: nova.store(%[[LOAD8]], %arg1) {start = [0, 0]} : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
// IDEMPOTENT-NEXT: nova.barrier() {mode = 1 : i32}
// IDEMPOTENT-NEXT: return

// IDEMPOTENT-LABEL: func.func @insert_keep_alive_when_double_buffering_enabled
// IDEMPOTENT: %[[LOAD9:.*]] = nova.load(%arg0) {start = [0, 0]} : memref<16x16xf32> -> tensor<16x16xf32>
// IDEMPOTENT-NEXT: %[[SQUARE9:.*]] = nova.square(%[[LOAD9]]) : tensor<16x16xf32> -> tensor<16x16xf32>
// IDEMPOTENT-NEXT: nova.barrier() {mode = 0 : i32}
// IDEMPOTENT-NEXT: %[[EXP9:.*]] = nova.exp(%[[SQUARE9]]) : tensor<16x16xf32> -> tensor<16x16xf32>
// IDEMPOTENT-NEXT: nova.keep_alive(%[[LOAD9]], %[[SQUARE9]]) : tensor<16x16xf32>, tensor<16x16xf32>
// IDEMPOTENT-NEXT: nova.barrier() {mode = 0 : i32}
// IDEMPOTENT-NEXT: nova.store(%[[EXP9]], %arg1) {start = [0, 0]} : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
// IDEMPOTENT-NEXT: nova.keep_alive(%[[SQUARE9]], %[[EXP9]]) : tensor<16x16xf32>, tensor<16x16xf32>
// IDEMPOTENT-NEXT: nova.barrier() {mode = 1 : i32}
// IDEMPOTENT-NEXT: return
