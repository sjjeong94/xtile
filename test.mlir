module {
  func.func @matmul_kernel(%arg0: memref<?x128xf32>, %arg1: memref<128x?xf32>, %arg2: memref<?x?xf32>) {
    %cst = arith.constant dense<2.000000e+00> : tensor<1x1xf32>
    %cst_0 = arith.constant dense<3.000000e+00> : tensor<1x1xf32>
    %c0_i32 = arith.constant 0 : i32
    %0:3 = "xt.get_tile_block_id"() : () -> (i32, i32, i32)
    %1 = "xt.load"(%arg0, %0#0, %c0_i32) : (memref<?x128xf32>, i32, i32) -> tensor<64x128xf32>
    %2 = "xt.load"(%arg1, %c0_i32, %0#1) {shared = 1 : i64} : (memref<128x?xf32>, i32, i32) -> tensor<128x64xf32>
    %3 = "nova.matmul"(%1, %2, %cst_0, %cst) : (tensor<64x128xf32>, tensor<128x64xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<64x64xf32>
    "xt.store"(%3, %arg2, %0#0, %0#1) : (tensor<64x64xf32>, memref<?x?xf32>, i32, i32) -> ()
    return
  }
}

