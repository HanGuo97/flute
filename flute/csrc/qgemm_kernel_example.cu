#include <cuda.h>
#include <stdio.h>
#include <cute/tensor.hpp>
#include "cutlass/workspace.h"
#include "qgemm_kernel.hpp"


template <
  typename Slices,
  typename Blocks,
  typename Threads,
  typename TileM,
  typename TileK,
  typename TileP,
  typename Stages,
  typename NumBits,
  typename GroupSize,
  config::QuantMapModeEnum QuantMapMode,
  config::AccumulationModeEnum AccumulationMode,
  config::DecompositionModeEnum DecompositionMode,
  typename G2STiledCopySizeS,
  typename MmaPrmK
>
void run_example(int M, int N, int K) {

  using namespace cute;
  using X = Underscore;

  // Y = X Q.T
  // X: (M, K)
  // Q: (N, K)
  using T  = cute::half_t;
  using TQ = cute::uint16_t;
  using TC = conditional_t<AccumulationMode == config::AccumulationModeEnum::Low , T, float>;  // type for accumulation
  using TR = conditional_t<AccumulationMode != config::AccumulationModeEnum::High, T, float>;  // type for reduction
  using T2 = conditional_t<is_same_v<T, half_t>, __half2, __nv_bfloat162>;

  int P       = ceil_div(N, sizeof(TQ) * 8) * NumBits{};
  int G       = K / GroupSize{};
  int qmsize  = _1{} << (NumBits{});
  int qmsize2 = _1{} << (NumBits{} * _2{});
  int workspace_blocks;
  int workspace_accum_size;

  if constexpr (DecompositionMode == config::DecompositionModeEnum::SplitK)
  {
    workspace_blocks = ceil_div(M, TileM{}) * ceil_div(P, TileP{}) * Slices{};
  }
  else
  {
    workspace_blocks = Blocks{};
  }

  if constexpr (DecompositionMode == config::DecompositionModeEnum::StreamK)
  {
    auto tiles_M = ceil_div(M, TileM{});
    auto tiles_K = ceil_div(K, TileK{});
    auto tiles_P = ceil_div(P, TileP{});
    if constexpr (is_same_v<NumBits, _3>) {
      tiles_P    = ceil_div(ceil_div(N, sizeof(TQ) * 8), TileP{});
    }
    auto tiles   = tiles_M * tiles_P * tiles_K;
    if (tiles < workspace_blocks)
    {
      return;
    }
  }

  if constexpr (cute::is_same_v<NumBits, _4>)
  {
    workspace_accum_size = sizeof(TR) * 16 * 4;
  }
  else if constexpr (cute::is_same_v<NumBits, _3>)
  {
    workspace_accum_size = sizeof(TR) * 64 * 4;
  }
  else
  {
    workspace_accum_size = sizeof(TR) * 32 * 4;
  }

  int workspace_size_partials = config::get_workspace_size_partials(workspace_blocks, Threads{}, workspace_accum_size);
  int workspace_size_barriers = config::get_workspace_size_barriers(workspace_blocks);
  int workspace_size = workspace_size_partials + workspace_size_barriers;

  T  *h_D  , *d_D;
  T  *h_A  , *d_A;
  TQ *h_Q  , *d_Q;
  T  *h_S  , *d_S;
  T  *h_QM , *d_QM;
  T2 *h_QM2, *d_QM2;
  void *d_workspace;

  h_D   = (T *)malloc(sizeof(T ) * M * N);
  h_A   = (T *)malloc(sizeof(T ) * M * K);
  h_Q   = (TQ*)malloc(sizeof(TQ) * P * K);  // transposed
  h_S   = (T *)malloc(sizeof(T ) * N * G);  // transposed
  h_QM  = (T *)malloc(sizeof(T ) * qmsize);
  h_QM2 = (T2*)malloc(sizeof(T2) * qmsize2);

  cudaMalloc(&d_D  , sizeof(T ) * M * N);
  cudaMalloc(&d_A  , sizeof(T ) * M * K);
  cudaMalloc(&d_Q  , sizeof(TQ) * P * K);
  cudaMalloc(&d_S  , sizeof(T ) * N * G);
  cudaMalloc(&d_QM , sizeof(T ) * qmsize);
  cudaMalloc(&d_QM2, sizeof(T2) * qmsize2);
  cudaMalloc(&d_workspace, workspace_size);

  auto tD   = make_tensor(h_D  , make_shape(M, N   ), make_stride(N, 1));
  auto tA   = make_tensor(h_A  , make_shape(M, K   ), make_stride(K, 1));
  auto tQ   = make_tensor(h_Q  , make_shape(P, K   ), make_stride(K, 1));
  auto tS   = make_tensor(h_S  , make_shape(N, G   ), make_stride(G, 1));
  auto tQM  = make_tensor(h_QM , make_shape(qmsize ), make_stride(1));
  auto tQM2 = make_tensor(h_QM2, make_shape(qmsize2), make_stride(1));

  // we only need to clear the workspace of barriers, and we assume that
  // the barriers workspace is at the beginning of the workspace
  cutlass::zero_workspace(d_workspace, workspace_size_barriers);

  srand(0);
  clear(tD);
  for (int i = 0; i < size(tA); ++i) {
    tA(i) = static_cast<T>( 2 * (rand() / double(RAND_MAX)) - 1 );
  }

  for (int i = 0; i < size(tQ); ++i) {
    tQ(i) = static_cast<TQ>( rand() % USHRT_MAX );
  }

  for (int i = 0; i < size(tS); ++i) {
    tS(i) = static_cast<T>( 2 * (rand() / double(RAND_MAX)) - 1 );
  }

  for (int i = 0; i < size(tQM); ++i) {
    tQM(i) = static_cast<T>(i);
  }

  for (int i = 0; i < size(tQM); ++i) {
    for (int j = 0; j < size(tQM); ++j) {
      tQM2(i * size(tQM) + j) = make_half2(
        // https://stackoverflow.com/questions/5924248/why-is-it-allowed-to-cast-a-pointer-to-a-reference
        reinterpret_cast<__half&>(tQM(i)),
        reinterpret_cast<__half&>(tQM(j)));
    }
  }

  cudaMemcpy(d_D  , h_D  , sizeof(T ) * M * N  , cudaMemcpyHostToDevice);
  cudaMemcpy(d_A  , h_A  , sizeof(T ) * M * K  , cudaMemcpyHostToDevice);
  cudaMemcpy(d_Q  , h_Q  , sizeof(TQ) * P * K  , cudaMemcpyHostToDevice);
  cudaMemcpy(d_S  , h_S  , sizeof(T ) * N * G  , cudaMemcpyHostToDevice);
  cudaMemcpy(d_QM , h_QM , sizeof(T ) * qmsize , cudaMemcpyHostToDevice);
  cudaMemcpy(d_QM2, h_QM2, sizeof(T2) * qmsize2, cudaMemcpyHostToDevice);

  qgemm_host<
    T,
    TQ,
    T2,
    Slices,
    Blocks,
    Threads,
    TileM,
    TileK,
    TileP,
    Stages,
    NumBits,
    GroupSize,
    QuantMapMode,
    AccumulationMode,
    DecompositionMode,
    G2STiledCopySizeS,
    MmaPrmK
  > (
    M,
    N,
    K,
    P,
    d_A,
    d_Q,
    d_D,
    d_S,
    d_QM,
    d_QM2,
    d_workspace,
    0);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  printf("shape = (%d, %d, %d) err = %d, str = %s\n", M, N, K, err, cudaGetErrorString(err));

  cudaMemcpy(h_D, d_D, sizeof(T) * M * N, cudaMemcpyDeviceToHost);
  auto tD_host = make_tensor(h_D, make_shape(M, N), make_stride(N, 1));
  auto tile    = make_tile(min(8, M), min(8, N));
  auto t32x32  = local_tile(tD_host, tile, make_coord(0, 0));
  print_tensor(t32x32);

  free(h_D);
  free(h_A);
  free(h_Q);
  free(h_S);
  free(h_QM);
  free(h_QM2);
  cudaFree(d_D);
  cudaFree(d_A);
  cudaFree(d_Q);
  cudaFree(d_S);
  cudaFree(d_QM);
  cudaFree(d_QM2);
  cudaFree(d_workspace);
}


template <
  typename NumBits,
  typename GroupSize
>
void tune(int M, int N, int K) {
  using namespace cute;

  static constexpr int kSMs = 108;
  static constexpr config::QuantMapModeEnum      kVectorized = config::QuantMapModeEnum     ::Vectorized;
  static constexpr config::QuantMapModeEnum      kMarlin     = config::QuantMapModeEnum     ::Marlin;
  static constexpr config::AccumulationModeEnum  kLow        = config::AccumulationModeEnum ::Low;
  static constexpr config::AccumulationModeEnum  kHigh       = config::AccumulationModeEnum ::High;
  static constexpr config::DecompositionModeEnum kSplitK     = config::DecompositionModeEnum::SplitK;
  static constexpr config::DecompositionModeEnum kStreamK    = config::DecompositionModeEnum::StreamK;

  //          Slices,        Blocks, Threads, TileM, TileK, TileP, Stages, NumBits, GroupSize, QuantMapMode, AccumulationMode, DecompositionMode, G2STiledCopySizeS, MmaPrmK
  run_example<    _0, Int<kSMs * 1>,    _256,   _32,   _64,   _32,     _2, NumBits, GroupSize,  kVectorized,             kLow,          kStreamK,                _2,      _1>(M, N, K);
  run_example<    _0, Int<kSMs * 1>,    _256,   _32,   _64,   _32,     _3, NumBits, GroupSize,  kVectorized,             kLow,          kStreamK,                _2,      _1>(M, N, K);
  run_example<    _0, Int<kSMs * 1>,    _256,   _32,   _64,   _32,     _4, NumBits, GroupSize,  kVectorized,             kLow,          kStreamK,                _2,      _1>(M, N, K);
  run_example<    _0, Int<kSMs * 1>,    _256,   _32,   _64,   _32,     _5, NumBits, GroupSize,  kVectorized,             kLow,          kStreamK,                _2,      _1>(M, N, K);

  run_example<    _0, Int<kSMs * 2>,    _256,   _32,   _64,   _32,     _2, NumBits, GroupSize,  kVectorized,             kLow,          kStreamK,                _2,      _1>(M, N, K);
  run_example<    _0, Int<kSMs * 2>,    _256,   _32,   _64,   _32,     _3, NumBits, GroupSize,  kVectorized,             kLow,          kStreamK,                _2,      _1>(M, N, K);
  run_example<    _0, Int<kSMs * 2>,    _256,   _32,   _64,   _32,     _4, NumBits, GroupSize,  kVectorized,             kLow,          kStreamK,                _2,      _1>(M, N, K);
  run_example<    _0, Int<kSMs * 2>,    _256,   _32,   _64,   _32,     _5, NumBits, GroupSize,  kVectorized,             kLow,          kStreamK,                _2,      _1>(M, N, K);

  run_example<    _0, Int<kSMs * 4>,    _256,   _32,   _64,   _32,     _2, NumBits, GroupSize,  kVectorized,             kLow,          kStreamK,                _2,      _1>(M, N, K);
  run_example<    _0, Int<kSMs * 4>,    _256,   _32,   _64,   _32,     _3, NumBits, GroupSize,  kVectorized,             kLow,          kStreamK,                _2,      _1>(M, N, K);
  run_example<    _0, Int<kSMs * 4>,    _256,   _32,   _64,   _32,     _4, NumBits, GroupSize,  kVectorized,             kLow,          kStreamK,                _2,      _1>(M, N, K);
  run_example<    _0, Int<kSMs * 4>,    _256,   _32,   _64,   _32,     _5, NumBits, GroupSize,  kVectorized,             kLow,          kStreamK,                _2,      _1>(M, N, K);

  run_example<    _0, Int<kSMs * 1>,    _128,   _16,   _64,   _32,     _2, NumBits, GroupSize,  kVectorized,             kLow,          kStreamK,                _2,      _1>(M, N, K);
  run_example<    _0, Int<kSMs * 1>,    _128,   _16,   _64,   _32,     _3, NumBits, GroupSize,  kVectorized,             kLow,          kStreamK,                _2,      _1>(M, N, K);
  run_example<    _0, Int<kSMs * 1>,    _128,   _16,   _64,   _32,     _4, NumBits, GroupSize,  kVectorized,             kLow,          kStreamK,                _2,      _1>(M, N, K);
  run_example<    _0, Int<kSMs * 1>,    _128,   _16,   _64,   _32,     _5, NumBits, GroupSize,  kVectorized,             kLow,          kStreamK,                _2,      _1>(M, N, K);

  run_example<    _0, Int<kSMs * 2>,    _128,   _16,   _64,   _32,     _2, NumBits, GroupSize,  kVectorized,             kLow,          kStreamK,                _2,      _1>(M, N, K);
  run_example<    _0, Int<kSMs * 2>,    _128,   _16,   _64,   _32,     _3, NumBits, GroupSize,  kVectorized,             kLow,          kStreamK,                _2,      _1>(M, N, K);
  run_example<    _0, Int<kSMs * 2>,    _128,   _16,   _64,   _32,     _4, NumBits, GroupSize,  kVectorized,             kLow,          kStreamK,                _2,      _1>(M, N, K);
  run_example<    _0, Int<kSMs * 2>,    _128,   _16,   _64,   _32,     _5, NumBits, GroupSize,  kVectorized,             kLow,          kStreamK,                _2,      _1>(M, N, K);

  run_example<    _0, Int<kSMs * 4>,    _128,   _16,   _64,   _32,     _2, NumBits, GroupSize,  kVectorized,             kLow,          kStreamK,                _2,      _1>(M, N, K);
  run_example<    _0, Int<kSMs * 4>,    _128,   _16,   _64,   _32,     _3, NumBits, GroupSize,  kVectorized,             kLow,          kStreamK,                _2,      _1>(M, N, K);
  run_example<    _0, Int<kSMs * 4>,    _128,   _16,   _64,   _32,     _4, NumBits, GroupSize,  kVectorized,             kLow,          kStreamK,                _2,      _1>(M, N, K);
  run_example<    _0, Int<kSMs * 4>,    _128,   _16,   _64,   _32,     _5, NumBits, GroupSize,  kVectorized,             kLow,          kStreamK,                _2,      _1>(M, N, K);
}


int main(int argc, char *argv[]) {

  int M = 1;
  int N = 8192;
  int K = 8192;

  if (argc >= 2)
  {
    sscanf(argv[1], "%d", &M);
  }
  if (argc >= 3)
  {
    sscanf(argv[2], "%d", &N);
  }
  if (argc >= 4)
  {
    sscanf(argv[3], "%d", &K);
  }

  // tune<cute::Int<4>, cute::Int<128>>(M, N, K);
  // tune<cute::Int<3>, cute::Int<128>>(M, N, K);
  // tune<cute::Int<2>, cute::Int<128>>(M, N, K);
  return 0;
}
