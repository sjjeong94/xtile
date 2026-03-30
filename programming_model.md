# XT / xtile DSL Programming Model

## 1. Model Summary

README 기준으로 보면, xtile은 다음 성격의 프로그래밍 모델이다.

- tile-centric
- block-oriented
- target-independent
- compiler-managed for execution details

핵심 목표는 사용자가 "각 tile-level block이 무엇을 해야 하는지"를 기술하게 하고, 실제 실행 스케줄링, 메모리 구체화, 동기화, 하드웨어 자원 매핑은 컴파일러와 하위 dialect가 맡도록 하는 것이다.

현재 저장소 구현까지 함께 보면, xtile은 이 모델을 Python 내장 DSL과 MLIR dialect 체인으로 실현한다.

`Python kernel` -> `xt dialect` -> `nova dialect` -> `x1 dialect`

즉, xtile은 단순한 Python 라이브러리가 아니라, block 단위 tile 프로그램을 기술하고 점차 하드웨어 지향 IR로 낮추는 컴파일 프로그래밍 모델이다.

## 2. First-Class Concepts

README가 전면에 두는 개념은 `Task`, `Grid`, `Block`, `Kernel`이다.

### Task

Task는 사용자가 수행하려는 전체 계산이다.

예:

- 두 `M x N` 행렬 더하기
- row-wise softmax
- tile matmul
- conv2d

Task는 사용자 관점의 전체 일이다. xtile은 이 전체 Task를 직접 순회하는 언어가 아니라, Task를 여러 Block으로 분할해 표현하는 언어다.

### Block

Block은 기본 작업 단위다.

각 Block은 Task의 일부를 담당한다. 보통 하나 이상의 tile을 읽고, tile 계산을 수행하고, 결과 tile을 기록한다.

README의 핵심 표현을 따르면, 사용자는 "한 Block이 무엇을 해야 하는가"를 기술한다.

### Grid

Grid는 모든 Block의 집합이다.

Grid는 최대 3차원이며, Python 표면 API에서는 `grid=(gx, gy, gz)`로 지정된다. 각 Block은 Grid 안의 좌표를 갖고, 그 좌표를 통해 자신이 처리할 tile을 결정한다.

### Kernel

Kernel은 "한 Block의 관점"에서 작성된 프로그램이다.

같은 Kernel이 Grid의 모든 Block에 적용되고, 각 Block은 자신의 block id를 사용해 처리할 tile을 계산한다. 이 점이 xtile의 가장 중요한 정신 모델이다.

정리하면:

- Task는 전체 계산
- Grid는 Block들의 배치
- Block은 작업 단위
- Kernel은 한 Block의 프로그램

## 3. Block-Centric Semantics

xtile에서 사용자는 전체 tensor loop nest를 직접 쓰지 않는다. 대신 Block 하나의 동작을 기술한다.

대표적인 예는 다음과 같다.

```python
@xt.kernel
def softmax(inp, out, trow, col):
    bid_x = xt.bid(0)

    x = xt.load(inp, index=(bid_x, 0), shape=(trow, col))
    x = x - xt.max(x, axis=-1)
    x = xt.exp(x)
    r = xt.sum(x, axis=-1)
    x = x / r
    xt.store(out, index=(bid_x, 0), tile=x)
```

이 코드는 "전체 softmax"를 순회하는 코드가 아니라:

- 현재 Block의 좌표를 읽고
- 그 Block이 담당하는 tile을 load하고
- tile 단위 연산을 수행하고
- 결과 tile을 store하는

하나의 Block 프로그램이다.

Host 쪽에서 `grid=(...)`를 지정하면, 그 Grid의 모든 Block에 같은 Kernel이 적용된다.

## 4. Logical Parallelism

README가 명시하듯이, Block들은 **logically parallel** 하다.

이 뜻은:

- 프로그래밍 모델 차원에서는 각 Block이 독립적인 작업 단위라고 가정한다.
- 하지만 실제 병렬 실행 여부, 부분 직렬화 여부, 하드웨어 자원에 어떻게 remap되는지는 backend/compiler가 결정한다.

현재 구현까지 보면 이 설명은 더 구체적이다.

- Python DSL은 Block 프로그램을 trace한다.
- `xt-serialize` pass는 Grid 전체를 순회하며 그 Block 프로그램을 복제한다.
- 이후 `nova` / `x1` lowering에서 barrier, allocation, command scheduling 같은 실행 세부사항이 구체화된다.

즉, 사용자는 논리적 병렬 구조를 기술하고, 실제 실행 형태는 compiler-managed다.

## 5. Memory Model

README 기준으로 xtile은 세 가지 memory class를 둔다.

- Global Memory
- Shared Memory
- Local Memory

이 구분이 xtile의 설계 중심이다.

### Global Memory

Global Memory는 전체 Grid에서 보이는 메모리다.

사용자 레벨에서는 `Array`로 표현된다. README는 Array가 개념적으로 다음 정보를 가진다고 설명한다.

- `shape`
- `dtype`
- `stride`

현재 Python 구현의 `xt.Array`는 `shape`와 `dtype`를 명시적으로 보유하고, stride는 아직 표면 타입으로 노출하지 않는다. 하지만 프로그래밍 모델 차원에서는 Array가 global storage를 나타내는 객체라는 점이 중요하다.

### Local Memory

Local Memory는 Block private memory다.

기본적으로 `xt.load(...)`로 읽은 데이터는 Block 내부의 tile 값이 되며, 이 tile은 local memory에 있는 것으로 모델링된다.

### Shared Memory

Shared Memory는 여러 Block이 같은 tile 데이터를 재사용한다는 사실을 표현하는 **logical memory class**다.

README가 특히 강조하는 점은 이것이:

- 사용자가 직접 할당/해제하는 물리 scratchpad가 아니고
- "이 tile이 Block들 사이에서 논리적으로 공유된다"는 의미를 담는다는 점이다.

실제 materialization은 컴파일러가 결정한다.

## 6. Array and Tile

xtile은 `Array`와 `Tile`을 구분한다.

### Array

Array는 Global Memory에 사는 객체다.

예:

- input tensors
- output tensors
- weights
- large global buffers

현재 구현에서는 `xt.Array(shape=..., dtype=...)`가 여기에 해당하며, lowering 후에는 보통 `memref<...>` 함수 인자가 된다.

### Tile

Tile은 Local Memory 또는 Shared Memory에 사는 계산 단위다.

Tile은 기본적으로:

- `shape`
- `dtype`

를 가진다.

현재 구현에서 kernel 내부 값은 `TensorValue`로 추적되며, 이것이 README가 말하는 Tile에 대응한다.

전형적인 xtile 프로그램의 형태는 항상 거의 같다.

1. Array에서 Tile을 load한다.
2. Tile에 대해 계산한다.
3. 결과 Tile을 Array로 store한다.

## 7. Tile Coordinates

README가 가장 강하게 규정하는 의미론 중 하나는 `xt.load`와 `xt.store`의 `index`가 **element coordinate가 아니라 tile coordinate** 라는 점이다.

예를 들어:

```python
tile = xt.load(a, index=(0, 1), shape=(32, 64))
```

이것은:

- row 0, col 1 element를 읽는 뜻이 아니라
- `(32, 64)` tile로 논리 분할된 공간에서 tile coordinate `(0, 1)`을 읽는 뜻이다.

현재 구현에서도 이 해석이 실제 lowering에 반영된다. `xt -> nova` lowering은 constant tile index를 tile shape와 곱해서 concrete start offset으로 바꾼다.

즉:

- `index=(i, j)`는 tile 번호
- 실제 주소 시작점은 `(i * tile_h, j * tile_w)`에 해당

라는 모델이 맞다.

## 8. Shared Tiles

README는 shared tile을 중요한 1급 개념으로 둔다.

예:

```python
tile = xt.load(a, index=(0, 1), shape=(32, 64), shared=1)
```

이것은 해당 tile이 local tile이 아니라 shared tile이라는 사실을 나타낸다.

### Meaning of `shared=k`

README 설명을 따르면, `shared=k`는 Block ID의 suffix가 같은 Block들을 같은 sharing group으로 묶는다.

3D Grid에서 Block ID가 `(b0, b1, b2)`일 때:

- `shared=1`: 같은 `b1`, `b2`를 가진 Block들이 공유
- `shared=2`: 같은 `b2`를 가진 Block들이 공유

현재 구현과 대조하면:

- Python 표면 API는 `shared=None|0|1|2`만 허용한다.
- `xt-serialize`는 `shared=1`과 `shared=2`에 대해 load 재사용을 구현한다.
- 결과적으로 README의 논리 모델과 현재 pass의 재사용 전략이 서로 대응한다.

이 개념의 본질은 "사용자가 reuse 구조를 선언하고, 구현 방식은 compiler가 결정한다"는 데 있다.

## 9. Compiler-Managed Execution Details

README의 공식 입장에서 사용자가 직접 관리하지 않는 것은 다음과 같다.

- physical scheduling
- hardware execution unit mapping
- shared memory allocation/free
- explicit synchronization such as barriers or waits

현재 코드베이스는 이 주장을 실제 구현으로 뒷받침한다.

- Python DSL에는 allocation/free API가 없다.
- explicit barrier API도 없다.
- 사용자는 `shared`와 `double_buffering` 같은 논리적 힌트만 준다.
- 이후 `nova_barrier`, `nova_allocate`, `nova_to_x1` 같은 pass가 하드웨어 관점 세부사항을 만든다.

즉, xtile의 모델은 "사용자에게 execution control을 주는" 모델이 아니라, "사용자에게 block/tile 의미론만 주고 execution details는 compiler가 추론하게 하는" 모델이다.

## 10. Host vs Kernel Roles

xtile은 Host와 Kernel의 책임을 분리한다.

### Host Responsibilities

Host는 다음을 정한다.

- global array shapes and dtypes
- tile sizes
- task partitioning
- grid shape
- compiler hints such as `double_buffering`

예를 들어 `kernels/add.py`, `kernels/matmul.py`, `kernels/conv2d.py`에서는 host 코드가 `xt.cdiv(...)`로 Grid를 계산하고, tile sizes를 정해 `xt.convert(...)`에 넘긴다.

### Kernel Responsibilities

Kernel은 다음만 기술한다.

- 현재 Block ID
- 어떤 tile을 읽는지
- tile 위에서 어떤 계산을 하는지
- 결과 tile을 어디에 쓰는지

Kernel은 실행 계획 전체를 다루지 않는다. Block-local semantics만 다룬다.

## 11. Current Surface DSL

현재 Python 표면 DSL에서 사용 가능한 핵심 연산은 다음과 같다.

- `xt.bid(dim)`
- `xt.load(...)`
- `xt.load_conv2d(...)`
- `xt.store(...)`
- `xt.add`, `xt.sub`, `xt.mul`
- `xt.exp`, `xt.cos`, `xt.sin`, `xt.reciprocal`, `xt.rsqrt`, `xt.sigmoid`, `xt.tanh`, `xt.silu`
- `xt.sum`, `xt.max`
- `xt.matmul`, `xt.mma`
- `xt.astype`
- `xt.reshape`, `xt.transpose`

Kernel 함수는 `@xt.kernel`로 감싸며, `xt.convert(...)` 중에만 의미가 있다. 일반 실행 시 tensor 계산을 수행하는 eager API가 아니다.

또한 kernel은 값을 반환하지 않아야 한다. 결과는 `xt.store(...)`로 기록한다.

## 12. Tracing Model

현재 구현에서 xtile은 eager execution이 아니라 tracing DSL이다.

`xt.convert(...)`가 호출되면:

1. `TraceContext`가 만들어진다.
2. Array 인자는 symbolic kernel arg로 바뀐다.
3. integer 인자는 compile-time 상수처럼 전달된다.
4. kernel 함수가 한 번 실행된다.
5. 각 `xt.*` 호출이 symbolic operation으로 기록된다.
6. 기록된 결과가 하나의 `func.func` MLIR로 직렬화된다.

이 점은 공식 모델을 구현 관점에서 해석할 때 중요하다.

- 사용자가 쓰는 Kernel은 "runtime body"라기보다 "Block 프로그램을 생성하는 trace"다.
- Python control flow는 실행 semantics의 중심이 아니라, 어떤 ops를 기록했는지가 중심이다.

## 13. Computation Semantics on Tiles

xtile의 계산은 tile 위에서 일어난다. 즉, Block은 element-by-element imperative code를 쓰는 것이 아니라, tile value를 받아 tile value를 만드는 SSA-style dataflow를 기술한다.

### Binary Tile Ops

- `xt.add`
- `xt.sub`
- `xt.mul`

현재 구현에서는:

- dtype이 같아야 하고
- broadcast-compatible shape여야 한다

Python tracing은 주로 row-wise broadcast 패턴을 허용하고, MLIR `xt` verifier는 더 일반적인 trailing broadcast 규칙을 허용한다.

### Unary Tile Ops

- `xt.exp`
- `xt.cos`
- `xt.sin`
- `xt.reciprocal`
- `xt.rsqrt`
- `xt.sigmoid`
- `xt.tanh`
- `xt.silu`

이 연산들은 shape과 dtype을 보존한다.

### Derived Division

표면 DSL에는 별도 `xt.div`가 없다.

`a / b`는 실제로:

```text
xt.reciprocal(b)
xt.mul(a, reciprocal_b)
```

로 표현된다.

### Reductions

- `xt.sum(tile, axis=...)`
- `xt.max(tile, axis=...)`

현재 Python 구현은 rank-2 reduction을 중심으로 구현되어 있으며, reduced dimension을 `1`로 유지한다. 즉 rank를 없애는 것이 아니라 유지한 채 축소한다.

예:

- `(M, N)` -> `(M, 1)` for `axis=1`
- `(M, N)` -> `(1, N)` for `axis=0`

이 역시 tile-oriented 모델과 잘 맞는다.

### Matmul and MMA

`xt.matmul(lhs, rhs)`는 rank-2 tile matmul이다.

- `lhs.shape[1] == rhs.shape[0]`
- 결과 shape는 `(lhs.shape[0], rhs.shape[1])`
- 현재 Python DSL에서는 `int8 x int8 -> float32`를 허용한다

`xt.mma(lhs, rhs, acc)`는 accumulate가 명시된 tile matmul이다.

- `lhs`, `rhs`는 현재 `int8`
- accumulator는 현재 `float32`
- accumulator shape는 결과 shape와 같아야 한다

## 14. Shape and Layout Transformations

README 첫 문장은 xtile이 tile-level data movement뿐 아니라 layout transformation도 표현한다고 말한다. 현재 구현에서 이 축에 해당하는 것은 다음이다.

### Reshape

`xt.reshape(tile, shape=...)`

- dtype 보존
- element count 동일해야 함

### Transpose

`xt.transpose(tile)`

- 현재 Python DSL에서는 rank-3 only
- dim 0 유지
- dim 1과 dim 2 swap

### Permute

`xt` IR에는 `xt.permute(...){permutation=[...]}`가 있고 `nova.permute`로 내려간다.

즉, 프로그래밍 모델 차원에서는 일반 permute까지 포함하는 layout transformation 모델이 존재한다. 다만 현재 Python 표면 DSL은 이를 직접 helper로 노출하지 않는다.

## 15. Load and Store as the Boundary Between Memory Classes

`xt.load(...)`와 `xt.store(...)`는 단순한 I/O API가 아니라 memory class 사이의 경계 연산이다.

`xt.load(array, index=..., shape=..., shared=...)`

- Global Memory의 Array에서
- Local 또는 Shared Tile을 만든다

`xt.store(array, index=..., tile=...)`

- Tile 결과를
- Global Memory의 Array에 기록한다

현재 verifier는 rank, shape, dtype 일관성을 강하게 요구한다. 따라서 xtile의 프로그래밍 모델은 동적이고 느슨한 모델이 아니라, 정적 shape 위에서 tile 이동과 계산을 정확하게 맞추는 모델이다.

## 16. Convolution as a Tile Primitive

현재 구현은 `xt.load_conv2d(...)`를 지원한다.

이 연산은 단순히 "global memory에서 4D tile을 읽는다"가 아니라:

1. conv output tile에 필요한 input slice를 계산하고
2. boundary padding을 반영하고
3. filter tile과 함께 convolution을 적용하는

fused tile primitive에 가깝다.

표면 제약:

- source rank-4 `int8`
- filter rank-4 `int8`
- result rank-4
- `group > 0`
- `pad`, `stride`, `dilation` 명시

`xt -> nova` lowering은 이것을 `nova.load + nova.conv2d`로 구체화한다. 따라서 공식 모델의 "tile-level data movement + computation"이라는 설명을 conv2d가 잘 보여준다.

## 17. Serialization and Realization

공식 모델에서 Block들은 논리적 구조다. 현재 구현에서 이를 실제 IR로 구체화하는 핵심 pass는 `xt-serialize`다.

이 pass는:

- `xt.grid`를 읽고
- single-block, void kernel을 요구하며
- Grid 전체 `(x, y, z)`를 순회하고
- block id를 concrete constant로 바꿔
- Block body를 반복적으로 복제한다

그리고 `shared` 힌트에 맞춰 load를 재사용한다.

이것은 README의 설명과 정확히 이어진다.

- Grid는 Block들의 집합이고
- Kernel은 한 Block 프로그램이며
- compiler가 그 Block 구조를 실제 스케줄/재사용 구조로 구체화한다

## 18. Lowering Pipeline

현재 저장소에서 보이는 lowering pipeline은 다음과 같다.

1. `xt.convert(...)`
   - Python kernel을 `xt` dialect MLIR로 변환
2. `xt.xt_serialize(...)`
   - Grid/Block semantics를 concrete serialized form으로 확장
3. `xt.xt_to_nova(...)`
   - hardware-oriented tile ops로 변환
4. `xt.nova_optimize(...)`
   - tile IR 최적화
5. `xt.nova_threading(...)`
   - threading/layout propagation
6. `xt.nova_barrier(...)`
   - synchronization insertion
7. `xt.nova_allocate(...)`
   - storage/materialization decisions
8. `xt.nova_to_x1(...)`
   - low-level command IR로 lowering

이 pipeline 자체가 README의 핵심 주장, 즉 "execution details are compiler-managed"를 실증한다.

## 19. What the User Controls vs What the Compiler Controls

### User Controls

사용자는 다음을 직접 표현한다.

- Task를 어떻게 Block으로 분할할지
- Grid shape
- 각 Block이 어떤 tile을 담당하는지
- tile load / compute / store 흐름
- 어떤 tile이 논리적으로 shared인지
- tile shape와 dtype

### Compiler Controls

컴파일러는 다음을 맡는다.

- serialization strategy
- physical scheduling
- hardware resource mapping
- barrier / synchronization insertion
- allocation / lifetime materialization
- lower-level command selection

이 구분이 xtile의 공식 모델을 가장 잘 요약한다.

## 20. Current Limitations in the Implementation

공식 모델은 더 일반적이지만, 현재 코드베이스 구현은 몇 가지 제약을 가진다.

- Python `xt.Array`는 README가 말하는 stride를 아직 노출하지 않는다
- dtype은 현재 매우 제한적이다
- kernel은 사실상 straight-line, single-block form에 맞춰져 있다
- `xt-serialize`는 void, single-block 함수만 지원한다
- Python 표면 DSL은 IR 전체 기능을 다 노출하지 않는다
- 일부 Python-side shape checks는 IR verifier보다 더 보수적이다

즉, README는 공식 모델을 설명하고, 현재 구현은 그 모델의 제한된 but coherent subset을 실현하고 있다고 보는 것이 가장 정확하다.

## 21. Concise Mental Model

xtile의 공식 모델을 가장 짧게 요약하면 이렇다.

> xtile은 전체 Task를 3D Grid의 Block들로 분해하고, 사용자가 한 Block의 tile-level program을 기술하면, 컴파일러가 그 논리 구조를 실제 memory movement, synchronization, scheduling, hardware commands로 구체화하는 target-independent tile programming model이다.

실무적으로는 아래 다섯 문장으로 기억하면 된다.

1. 사용자는 전체 계산이 아니라 한 Block의 tile program을 쓴다.
2. `Grid`는 Block들의 논리적 배치이고, `xt.bid(...)`는 현재 Block 좌표다.
3. `Array`는 global object이고 `Tile`은 local/shared 계산 단위다.
4. `xt.load` / `xt.store`의 `index`는 element index가 아니라 tile coordinate다.
5. 공유, 스케줄, 배리어, allocation, hardware mapping은 사용자가 아니라 compiler가 맡는다.
