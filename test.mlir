module {
  func.func @layernorm_kernel(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>) {
    %c0_i32 = arith.constant 0 : i32
    %0:3 = "xt.get_tile_block_id"() : () -> (i32, i32, i32)
    %1 = "xt.load"(%arg0, %0#0, %c0_i32) : (memref<?x?xf32>, i32, i32) -> tensor<16x16xf32>
    %2 = "nova.reduce"(%1) <{mode = 0 : i32}> : (tensor<16x16xf32>) -> tensor<16x1xf32>
    %3 = "nova.broadcast"(%1, %2) <{lhs_b = 0.000000e+00 : f32, lhs_s = 1.000000e+00 : f32, mode = 3 : i32, rhs_b = 0.000000e+00 : f32, rhs_s = 6.250000e-02 : f32}> : (tensor<16x16xf32>, tensor<16x1xf32>) -> tensor<16x16xf32>
    %4 = "nova.elementwise"(%3, %3) <{lhs_b = 0.000000e+00 : f32, lhs_s = 1.000000e+00 : f32, mode = 2 : i32, rhs_b = 0.000000e+00 : f32, rhs_s = 1.000000e+00 : f32}> : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    %5 = "nova.reduce"(%4) <{mode = 0 : i32}> : (tensor<16x16xf32>) -> tensor<16x1xf32>
    %6 = "nova.scalar"(%5) <{mode = 2 : i32, rhs = 6.250000e-02 : f32}> : (tensor<16x1xf32>) -> tensor<16x1xf32>
    %7 = "xt.rsqrt"(%6) : (tensor<16x1xf32>) -> tensor<16x1xf32>
    %8 = "nova.broadcast"(%3, %7) <{lhs_b = 0.000000e+00 : f32, lhs_s = 1.000000e+00 : f32, mode = 2 : i32, rhs_b = 0.000000e+00 : f32, rhs_s = 1.000000e+00 : f32}> : (tensor<16x16xf32>, tensor<16x1xf32>) -> tensor<16x16xf32>
    "xt.store"(%8, %arg1, %0#0, %c0_i32) : (tensor<16x16xf32>, memref<?x?xf32>, i32, i32) -> ()
    return
  }
}

