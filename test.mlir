module {
  func.func @layernorm_kernel(%arg0: memref<?xf32>, %arg1: memref<?xf32>) {
    %cst = arith.constant dense<6.250000e-02> : tensor<1x1xf32>
    %0:3 = "xt.get_tile_block_id"() : () -> (i32, i32, i32)
    %1 = "xt.load"(%arg0, %0#0) : (memref<?xf32>, i32) -> tensor<16x16xf32>
    %2 = "nova.reduce"(%1) <{mode = 0 : i32}> : (tensor<16x16xf32>) -> tensor<16x1xf32>
    %3 = "nova.broadcast"(%2, %cst) <{lhs_b = 0.000000e+00 : f32, lhs_s = 1.000000e+00 : f32, mode = 2 : i32, rhs_b = 0.000000e+00 : f32, rhs_s = 1.000000e+00 : f32}> : (tensor<16x1xf32>, tensor<1x1xf32>) -> tensor<16x1xf32>
    %4 = "nova.broadcast"(%1, %3) <{lhs_b = 0.000000e+00 : f32, lhs_s = 1.000000e+00 : f32, mode = 3 : i32, rhs_b = 0.000000e+00 : f32, rhs_s = 1.000000e+00 : f32}> : (tensor<16x16xf32>, tensor<16x1xf32>) -> tensor<16x16xf32>
    %5 = "nova.elementwise"(%4, %4) <{lhs_b = 0.000000e+00 : f32, lhs_s = 1.000000e+00 : f32, mode = 2 : i32, rhs_b = 0.000000e+00 : f32, rhs_s = 1.000000e+00 : f32}> : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    %6 = "nova.reduce"(%5) <{mode = 0 : i32}> : (tensor<16x16xf32>) -> tensor<16x1xf32>
    %7 = "nova.broadcast"(%6, %cst) <{lhs_b = 0.000000e+00 : f32, lhs_s = 1.000000e+00 : f32, mode = 2 : i32, rhs_b = 0.000000e+00 : f32, rhs_s = 1.000000e+00 : f32}> : (tensor<16x1xf32>, tensor<1x1xf32>) -> tensor<16x1xf32>
    %8 = "xt.rsqrt"(%7) : (tensor<16x1xf32>) -> tensor<16x1xf32>
    %9 = "nova.broadcast"(%4, %8) <{lhs_b = 0.000000e+00 : f32, lhs_s = 1.000000e+00 : f32, mode = 2 : i32, rhs_b = 0.000000e+00 : f32, rhs_s = 1.000000e+00 : f32}> : (tensor<16x16xf32>, tensor<16x1xf32>) -> tensor<16x16xf32>
    "xt.store"(%9, %arg1, %0#0) : (tensor<16x16xf32>, memref<?xf32>, i32) -> ()
    return
  }
}

