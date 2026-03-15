module {
  func.func @generic_shared_and_contract(%arg0: memref<128x256xi8>, %arg1: memref<256x512xi8>, %arg2: memref<128x512xf32>, %arg3: memref<1x512xf32>, %arg4: memref<1x512xf32>) attributes {xt.grid = array<i32: 2, 8, 1>} {
    %c7_i32 = arith.constant 7 : i32
    %c6_i32 = arith.constant 6 : i32
    %c5_i32 = arith.constant 5 : i32
    %c4_i32 = arith.constant 4 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = "xt.load"(%arg0, %c0_i32, %c0_i32) : (memref<128x256xi8>, i32, i32) -> tensor<64x256xi8>
    %1 = "xt.load"(%arg1, %c0_i32, %c0_i32) <{shared = 1 : i64}> : (memref<256x512xi8>, i32, i32) -> tensor<256x64xi8>
    %2 = "xt.load"(%arg3, %c0_i32, %c0_i32) <{shared = 1 : i64}> : (memref<1x512xf32>, i32, i32) -> tensor<1x64xf32>
    %3 = "xt.load"(%arg4, %c0_i32, %c0_i32) <{shared = 1 : i64}> : (memref<1x512xf32>, i32, i32) -> tensor<1x64xf32>
    %4 = "nova.matmul"(%0, %1, %2, %3) : (tensor<64x256xi8>, tensor<256x64xi8>, tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<64x64xf32>
    "xt.store"(%4, %arg2, %c0_i32, %c0_i32) : (tensor<64x64xf32>, memref<128x512xf32>, i32, i32) -> ()
    %5 = "xt.load"(%arg0, %c1_i32, %c0_i32) : (memref<128x256xi8>, i32, i32) -> tensor<64x256xi8>
    %6 = "nova.matmul"(%5, %1, %2, %3) : (tensor<64x256xi8>, tensor<256x64xi8>, tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<64x64xf32>
    "xt.store"(%6, %arg2, %c1_i32, %c0_i32) : (tensor<64x64xf32>, memref<128x512xf32>, i32, i32) -> ()
    %7 = "xt.load"(%arg0, %c0_i32, %c0_i32) : (memref<128x256xi8>, i32, i32) -> tensor<64x256xi8>
    %8 = "xt.load"(%arg1, %c0_i32, %c1_i32) <{shared = 1 : i64}> : (memref<256x512xi8>, i32, i32) -> tensor<256x64xi8>
    %9 = "xt.load"(%arg3, %c0_i32, %c1_i32) <{shared = 1 : i64}> : (memref<1x512xf32>, i32, i32) -> tensor<1x64xf32>
    %10 = "xt.load"(%arg4, %c0_i32, %c1_i32) <{shared = 1 : i64}> : (memref<1x512xf32>, i32, i32) -> tensor<1x64xf32>
    %11 = "nova.matmul"(%7, %8, %9, %10) : (tensor<64x256xi8>, tensor<256x64xi8>, tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<64x64xf32>
    "xt.store"(%11, %arg2, %c0_i32, %c1_i32) : (tensor<64x64xf32>, memref<128x512xf32>, i32, i32) -> ()
    %12 = "xt.load"(%arg0, %c1_i32, %c0_i32) : (memref<128x256xi8>, i32, i32) -> tensor<64x256xi8>
    %13 = "nova.matmul"(%12, %8, %9, %10) : (tensor<64x256xi8>, tensor<256x64xi8>, tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<64x64xf32>
    "xt.store"(%13, %arg2, %c1_i32, %c1_i32) : (tensor<64x64xf32>, memref<128x512xf32>, i32, i32) -> ()
    %14 = "xt.load"(%arg0, %c0_i32, %c0_i32) : (memref<128x256xi8>, i32, i32) -> tensor<64x256xi8>
    %15 = "xt.load"(%arg1, %c0_i32, %c2_i32) <{shared = 1 : i64}> : (memref<256x512xi8>, i32, i32) -> tensor<256x64xi8>
    %16 = "xt.load"(%arg3, %c0_i32, %c2_i32) <{shared = 1 : i64}> : (memref<1x512xf32>, i32, i32) -> tensor<1x64xf32>
    %17 = "xt.load"(%arg4, %c0_i32, %c2_i32) <{shared = 1 : i64}> : (memref<1x512xf32>, i32, i32) -> tensor<1x64xf32>
    %18 = "nova.matmul"(%14, %15, %16, %17) : (tensor<64x256xi8>, tensor<256x64xi8>, tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<64x64xf32>
    "xt.store"(%18, %arg2, %c0_i32, %c2_i32) : (tensor<64x64xf32>, memref<128x512xf32>, i32, i32) -> ()
    %19 = "xt.load"(%arg0, %c1_i32, %c0_i32) : (memref<128x256xi8>, i32, i32) -> tensor<64x256xi8>
    %20 = "nova.matmul"(%19, %15, %16, %17) : (tensor<64x256xi8>, tensor<256x64xi8>, tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<64x64xf32>
    "xt.store"(%20, %arg2, %c1_i32, %c2_i32) : (tensor<64x64xf32>, memref<128x512xf32>, i32, i32) -> ()
    %21 = "xt.load"(%arg0, %c0_i32, %c0_i32) : (memref<128x256xi8>, i32, i32) -> tensor<64x256xi8>
    %22 = "xt.load"(%arg1, %c0_i32, %c3_i32) <{shared = 1 : i64}> : (memref<256x512xi8>, i32, i32) -> tensor<256x64xi8>
    %23 = "xt.load"(%arg3, %c0_i32, %c3_i32) <{shared = 1 : i64}> : (memref<1x512xf32>, i32, i32) -> tensor<1x64xf32>
    %24 = "xt.load"(%arg4, %c0_i32, %c3_i32) <{shared = 1 : i64}> : (memref<1x512xf32>, i32, i32) -> tensor<1x64xf32>
    %25 = "nova.matmul"(%21, %22, %23, %24) : (tensor<64x256xi8>, tensor<256x64xi8>, tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<64x64xf32>
    "xt.store"(%25, %arg2, %c0_i32, %c3_i32) : (tensor<64x64xf32>, memref<128x512xf32>, i32, i32) -> ()
    %26 = "xt.load"(%arg0, %c1_i32, %c0_i32) : (memref<128x256xi8>, i32, i32) -> tensor<64x256xi8>
    %27 = "nova.matmul"(%26, %22, %23, %24) : (tensor<64x256xi8>, tensor<256x64xi8>, tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<64x64xf32>
    "xt.store"(%27, %arg2, %c1_i32, %c3_i32) : (tensor<64x64xf32>, memref<128x512xf32>, i32, i32) -> ()
    %28 = "xt.load"(%arg0, %c0_i32, %c0_i32) : (memref<128x256xi8>, i32, i32) -> tensor<64x256xi8>
    %29 = "xt.load"(%arg1, %c0_i32, %c4_i32) <{shared = 1 : i64}> : (memref<256x512xi8>, i32, i32) -> tensor<256x64xi8>
    %30 = "xt.load"(%arg3, %c0_i32, %c4_i32) <{shared = 1 : i64}> : (memref<1x512xf32>, i32, i32) -> tensor<1x64xf32>
    %31 = "xt.load"(%arg4, %c0_i32, %c4_i32) <{shared = 1 : i64}> : (memref<1x512xf32>, i32, i32) -> tensor<1x64xf32>
    %32 = "nova.matmul"(%28, %29, %30, %31) : (tensor<64x256xi8>, tensor<256x64xi8>, tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<64x64xf32>
    "xt.store"(%32, %arg2, %c0_i32, %c4_i32) : (tensor<64x64xf32>, memref<128x512xf32>, i32, i32) -> ()
    %33 = "xt.load"(%arg0, %c1_i32, %c0_i32) : (memref<128x256xi8>, i32, i32) -> tensor<64x256xi8>
    %34 = "nova.matmul"(%33, %29, %30, %31) : (tensor<64x256xi8>, tensor<256x64xi8>, tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<64x64xf32>
    "xt.store"(%34, %arg2, %c1_i32, %c4_i32) : (tensor<64x64xf32>, memref<128x512xf32>, i32, i32) -> ()
    %35 = "xt.load"(%arg0, %c0_i32, %c0_i32) : (memref<128x256xi8>, i32, i32) -> tensor<64x256xi8>
    %36 = "xt.load"(%arg1, %c0_i32, %c5_i32) <{shared = 1 : i64}> : (memref<256x512xi8>, i32, i32) -> tensor<256x64xi8>
    %37 = "xt.load"(%arg3, %c0_i32, %c5_i32) <{shared = 1 : i64}> : (memref<1x512xf32>, i32, i32) -> tensor<1x64xf32>
    %38 = "xt.load"(%arg4, %c0_i32, %c5_i32) <{shared = 1 : i64}> : (memref<1x512xf32>, i32, i32) -> tensor<1x64xf32>
    %39 = "nova.matmul"(%35, %36, %37, %38) : (tensor<64x256xi8>, tensor<256x64xi8>, tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<64x64xf32>
    "xt.store"(%39, %arg2, %c0_i32, %c5_i32) : (tensor<64x64xf32>, memref<128x512xf32>, i32, i32) -> ()
    %40 = "xt.load"(%arg0, %c1_i32, %c0_i32) : (memref<128x256xi8>, i32, i32) -> tensor<64x256xi8>
    %41 = "nova.matmul"(%40, %36, %37, %38) : (tensor<64x256xi8>, tensor<256x64xi8>, tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<64x64xf32>
    "xt.store"(%41, %arg2, %c1_i32, %c5_i32) : (tensor<64x64xf32>, memref<128x512xf32>, i32, i32) -> ()
    %42 = "xt.load"(%arg0, %c0_i32, %c0_i32) : (memref<128x256xi8>, i32, i32) -> tensor<64x256xi8>
    %43 = "xt.load"(%arg1, %c0_i32, %c6_i32) <{shared = 1 : i64}> : (memref<256x512xi8>, i32, i32) -> tensor<256x64xi8>
    %44 = "xt.load"(%arg3, %c0_i32, %c6_i32) <{shared = 1 : i64}> : (memref<1x512xf32>, i32, i32) -> tensor<1x64xf32>
    %45 = "xt.load"(%arg4, %c0_i32, %c6_i32) <{shared = 1 : i64}> : (memref<1x512xf32>, i32, i32) -> tensor<1x64xf32>
    %46 = "nova.matmul"(%42, %43, %44, %45) : (tensor<64x256xi8>, tensor<256x64xi8>, tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<64x64xf32>
    "xt.store"(%46, %arg2, %c0_i32, %c6_i32) : (tensor<64x64xf32>, memref<128x512xf32>, i32, i32) -> ()
    %47 = "xt.load"(%arg0, %c1_i32, %c0_i32) : (memref<128x256xi8>, i32, i32) -> tensor<64x256xi8>
    %48 = "nova.matmul"(%47, %43, %44, %45) : (tensor<64x256xi8>, tensor<256x64xi8>, tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<64x64xf32>
    "xt.store"(%48, %arg2, %c1_i32, %c6_i32) : (tensor<64x64xf32>, memref<128x512xf32>, i32, i32) -> ()
    %49 = "xt.load"(%arg0, %c0_i32, %c0_i32) : (memref<128x256xi8>, i32, i32) -> tensor<64x256xi8>
    %50 = "xt.load"(%arg1, %c0_i32, %c7_i32) <{shared = 1 : i64}> : (memref<256x512xi8>, i32, i32) -> tensor<256x64xi8>
    %51 = "xt.load"(%arg3, %c0_i32, %c7_i32) <{shared = 1 : i64}> : (memref<1x512xf32>, i32, i32) -> tensor<1x64xf32>
    %52 = "xt.load"(%arg4, %c0_i32, %c7_i32) <{shared = 1 : i64}> : (memref<1x512xf32>, i32, i32) -> tensor<1x64xf32>
    %53 = "nova.matmul"(%49, %50, %51, %52) : (tensor<64x256xi8>, tensor<256x64xi8>, tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<64x64xf32>
    "xt.store"(%53, %arg2, %c0_i32, %c7_i32) : (tensor<64x64xf32>, memref<128x512xf32>, i32, i32) -> ()
    %54 = "xt.load"(%arg0, %c1_i32, %c0_i32) : (memref<128x256xi8>, i32, i32) -> tensor<64x256xi8>
    %55 = "nova.matmul"(%54, %50, %51, %52) : (tensor<64x256xi8>, tensor<256x64xi8>, tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<64x64xf32>
    "xt.store"(%55, %arg2, %c1_i32, %c7_i32) : (tensor<64x64xf32>, memref<128x512xf32>, i32, i32) -> ()
    return
  }
}

