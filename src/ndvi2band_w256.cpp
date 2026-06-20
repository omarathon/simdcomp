// Two-band lock-step fused NDVI decode (256-bit, 16-lane, independent b).
//
// The 2-band analog of the single-band fused-sum kernel: for each 256-elem
// sub-block, OutReg j of band A and OutReg j of band B are produced IN REGISTERS
// (immediate-shift unpack) and handed straight to an NDVI aggregate — never
// materialised to a buffer (that store+reload is what kills the L1-temp variant).
//
// Independent b: bit-widths bA, bB are compile-time per kernel, so the shifts are
// immediates. One out-of-line kernel per (bA,bB) pair (17x17=289), dispatched by
// g_tbl[bA][bB] — exactly like single-band's switch(b), so only the (bA,bB) pairs
// that actually occur go hot in I-cache.
//
// Built into its own lib with -mno-avx512f (AVX2-only target; the server SIGILLs
// on EVEX-encoded YMM that -march=native would otherwise emit).

#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <utility>

namespace {

// OutReg j of a 256-block at bit-width B (compile-time): bits [j*B, j*B+B) of
// each 16-bit lane's stream, straddling into the next 16-bit word when
// (j*B)%16 + B > 16. B==0 -> all-zero OutReg (mask 0; stray load is harmless).
template <int B, int J>
static inline __m256i ex(const __m256i* in) {
  constexpr int o = J * B, w = o >> 4, s = o & 15;
  __m256i v = _mm256_srli_epi16(_mm256_loadu_si256(in + w), s);
  if constexpr (s + B > 16)
    v = _mm256_or_si256(v, _mm256_slli_epi16(_mm256_loadu_si256(in + w + 1),
                                             16 - s));
  constexpr int m = (B >= 16) ? 0xFFFF : ((1 << B) - 1);
  return _mm256_and_si256(v, _mm256_set1_epi16((short)m));
}

// Kernel ladder (op selects the per-OutReg-pair aggregate), so the cost deltas
// isolate decode vs widen vs divide — the 2-band analog of the single-band ladder.
enum { OP_NOOP = 0, OP_ADD = 1, OP_DIV = 2, OP_RCP = 3, OP_RCPRAW = 4,
       OP_NDVI_DIV = 5, OP_NDVI_RCP = 6, OP_NDVI_RCPRAW = 7, OP_NDVI_COUNT = 8,
       OP_ADDFP = 9, OP_CVTFP = 10 };

// Fixed-point threshold coefficients for OP_NDVI_COUNT: NDVI > x  <=>
// (a-b) > x*(a+b)  <=>  a*(1-x) - b*(1+x) > 0  (valid since a+b >= 0; the
// a=b=0 case gives 0 > 0 = false, consistent with masking den==0 as "no count").
// x is a runtime threshold but FIXED across the whole decode (not per-pixel),
// so it folds into two constants reused via one broadcast register — exactly
// the case where a constant-divisor fixed-point multiply trick applies
// (unlike the per-pixel-variable divide NDVI itself needs).
// Layout: 16 lanes of [K1,-K2,K1,-K2,...] so vpmaddwd(interleave(a,b), coef)
// directly yields a*K1 - b*K2 in int32 — no float widen, no divide at all.
thread_local __m256i g_count_coef = _mm256_setzero_si256();

template <int OP>
static inline void acc_op(__m256i va, __m256i vb, __m256& accf, __m256i& accx) {
  if constexpr (OP == OP_NOOP) {
    accx = _mm256_xor_si256(accx, _mm256_xor_si256(va, vb));
    return;
  }
  if constexpr (OP == OP_ADD) {
    // Integer widen-sum of both bands — the 2-band analog of single-band sum's
    // aggregate_sums_u16 (unpacklo/hi widen + add). NOT float (that widen would
    // dominate and hide the bandwidth win).
    const __m256i z = _mm256_setzero_si256();
    accx = _mm256_add_epi32(accx, _mm256_unpacklo_epi16(va, z));
    accx = _mm256_add_epi32(accx, _mm256_unpackhi_epi16(va, z));
    accx = _mm256_add_epi32(accx, _mm256_unpacklo_epi16(vb, z));
    accx = _mm256_add_epi32(accx, _mm256_unpackhi_epi16(vb, z));
    return;
  }
  if constexpr (OP == OP_NDVI_COUNT) {
    // count(NDVI > x) per OutReg: pure integer, no widen-to-float, no divide.
    // unpacklo/hi interleave (a,b) pairs (8 pixels each; order doesn't matter,
    // we only count); vpmaddwd against [K1,-K2,...] gives a*K1-b*K2 per pixel
    // directly in int32; cmpgt yields -1/0 per lane, summed into accx (negate
    // once at the very end to recover the true count).
    __m256i lo = _mm256_unpacklo_epi16(va, vb);
    __m256i hi = _mm256_unpackhi_epi16(va, vb);
    __m256i dlo = _mm256_madd_epi16(lo, g_count_coef);
    __m256i dhi = _mm256_madd_epi16(hi, g_count_coef);
    const __m256i z = _mm256_setzero_si256();
    accx = _mm256_add_epi32(accx, _mm256_cmpgt_epi32(dlo, z));
    accx = _mm256_add_epi32(accx, _mm256_cmpgt_epi32(dhi, z));
    return;
  }
  if constexpr (OP == OP_CVTFP) {
    // Convert both bands to float, XOR bit-pattern into accx — no float arithmetic.
    // Isolates pure conversion cost; XOR prevents DCE without any add_ps overhead.
    __m128i alo = _mm256_castsi256_si128(va), ahi = _mm256_extracti128_si256(va, 1);
    __m128i blo = _mm256_castsi256_si128(vb), bhi = _mm256_extracti128_si256(vb, 1);
    accx = _mm256_xor_si256(accx, _mm256_castps_si256(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(alo))));
    accx = _mm256_xor_si256(accx, _mm256_castps_si256(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(ahi))));
    accx = _mm256_xor_si256(accx, _mm256_castps_si256(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(blo))));
    accx = _mm256_xor_si256(accx, _mm256_castps_si256(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(bhi))));
    return;
  }
  if constexpr (OP == OP_ADDFP) {
    // Tree-reduce 4 converted values before touching accf — reduces accf dep chain
    // from 4×add_ps (16 cycles) to 1 (4 cycles), exposing memory-bandwidth limit.
    __m128i alo = _mm256_castsi256_si128(va), ahi = _mm256_extracti128_si256(va, 1);
    __m128i blo = _mm256_castsi256_si128(vb), bhi = _mm256_extracti128_si256(vb, 1);
    __m256 ta = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(alo));
    __m256 tb = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(ahi));
    __m256 tc = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(blo));
    __m256 td = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(bhi));
    accf = _mm256_add_ps(accf, _mm256_add_ps(_mm256_add_ps(ta, tb), _mm256_add_ps(tc, td)));
    return;
  }
  // OP_DIV / OP_RCP / OP_RCPRAW: widen to float, compute, combine results before
  // touching accf (tree-reduce: 2→1 add_ps on accf per call).
  __m128i alo = _mm256_castsi256_si128(va), ahi = _mm256_extracti128_si256(va, 1);
  __m128i blo = _mm256_castsi256_si128(vb), bhi = _mm256_extracti128_si256(vb, 1);
  __m256 a0 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(alo));
  __m256 b0 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(blo));
  __m256 a1 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(ahi));
  __m256 b1 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(bhi));
  if constexpr (OP == OP_DIV) {
    // Combine both divides before accumulating — div0 and div1 run in parallel.
    accf = _mm256_add_ps(accf, _mm256_add_ps(_mm256_div_ps(a0, b0), _mm256_div_ps(a1, b1)));
  }
  if constexpr (OP == OP_RCP) {
    __m256 r0 = _mm256_rcp_ps(b0), r1 = _mm256_rcp_ps(b1);
    r0 = _mm256_mul_ps(r0, _mm256_fnmadd_ps(b0, r0, _mm256_set1_ps(2.f)));
    r1 = _mm256_mul_ps(r1, _mm256_fnmadd_ps(b1, r1, _mm256_set1_ps(2.f)));
    accf = _mm256_add_ps(accf, _mm256_add_ps(_mm256_mul_ps(a0, r0), _mm256_mul_ps(a1, r1)));
  }
  if constexpr (OP == OP_RCPRAW) {
    accf = _mm256_add_ps(accf, _mm256_add_ps(_mm256_mul_ps(a0, _mm256_rcp_ps(b0)),
                                              _mm256_mul_ps(a1, _mm256_rcp_ps(b1))));
  }
  // Full NDVI formula: (a-b)/(a+b), den==0 -> 0.
  // num/den computed in integer (exact: uint16 add/sub fits in int32, zero precision cost).
  if constexpr (OP == OP_NDVI_DIV || OP == OP_NDVI_RCP || OP == OP_NDVI_RCPRAW) {
    const __m256i ia0 = _mm256_cvtepu16_epi32(alo), ib0 = _mm256_cvtepu16_epi32(blo);
    const __m256i ia1 = _mm256_cvtepu16_epi32(ahi), ib1 = _mm256_cvtepu16_epi32(bhi);
    __m256 num0 = _mm256_cvtepi32_ps(_mm256_sub_epi32(ia0, ib0));
    __m256 den0 = _mm256_cvtepi32_ps(_mm256_add_epi32(ia0, ib0));
    __m256 num1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(ia1, ib1));
    __m256 den1 = _mm256_cvtepi32_ps(_mm256_add_epi32(ia1, ib1));
    const __m256 z = _mm256_setzero_ps();
    __m256 mask0 = _mm256_cmp_ps(den0, z, _CMP_GT_OQ);
    __m256 mask1 = _mm256_cmp_ps(den1, z, _CMP_GT_OQ);
    if constexpr (OP == OP_NDVI_DIV) {
      accf = _mm256_add_ps(accf, _mm256_add_ps(
          _mm256_and_ps(_mm256_div_ps(num0, den0), mask0),
          _mm256_and_ps(_mm256_div_ps(num1, den1), mask1)));
    }
    if constexpr (OP == OP_NDVI_RCP) {
      __m256 r0 = _mm256_rcp_ps(den0), r1 = _mm256_rcp_ps(den1);
      r0 = _mm256_mul_ps(r0, _mm256_fnmadd_ps(den0, r0, _mm256_set1_ps(2.f)));
      r1 = _mm256_mul_ps(r1, _mm256_fnmadd_ps(den1, r1, _mm256_set1_ps(2.f)));
      accf = _mm256_add_ps(accf, _mm256_add_ps(
          _mm256_and_ps(_mm256_mul_ps(num0, r0), mask0),
          _mm256_and_ps(_mm256_mul_ps(num1, r1), mask1)));
    }
    if constexpr (OP == OP_NDVI_RCPRAW) {
      accf = _mm256_add_ps(accf, _mm256_add_ps(
          _mm256_and_ps(_mm256_mul_ps(num0, _mm256_rcp_ps(den0)), mask0),
          _mm256_and_ps(_mm256_mul_ps(num1, _mm256_rcp_ps(den1)), mask1)));
    }
  }
}

// One sub-block, both bands, lock-step (16 OutRegs). Out-of-line per (bA,bB,OP).
template <int OP, int bA, int bB>
__attribute__((noinline)) static void sub_op(const __m256i* inA,
                                             const __m256i* inB, __m256* pf,
                                             __m256i* px) {
  constexpr bool kFloat = (OP == OP_DIV || OP == OP_RCP || OP == OP_RCPRAW ||
                            OP == OP_NDVI_DIV || OP == OP_NDVI_RCP || OP == OP_NDVI_RCPRAW ||
                            OP == OP_ADDFP);
  if constexpr (kFloat) {
    // Round-robin across 4 independent float accumulators to break the 16-deep serial
    // add_ps chain down to 4-deep, letting divide/rcp throughput show through.
    __m256 farr[4] = {*pf, _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    __m256i xd = _mm256_setzero_si256();
    [&]<int... J>(std::integer_sequence<int, J...>) {
      ((acc_op<OP>(ex<bA, J>(inA), ex<bB, J>(inB), farr[J & 3], xd)), ...);
    }(std::make_integer_sequence<int, 16>{});
    *pf = _mm256_add_ps(_mm256_add_ps(farr[0], farr[1]), _mm256_add_ps(farr[2], farr[3]));
  } else {
    __m256 f = *pf;
    __m256i x = *px;
    [&]<int... J>(std::integer_sequence<int, J...>) {
      ((acc_op<OP>(ex<bA, J>(inA), ex<bB, J>(inB), f, x)), ...);
    }(std::make_integer_sequence<int, 16>{});
    *pf = f;
    *px = x;
  }
}

using Fn = void (*)(const __m256i*, const __m256i*, __m256*, __m256i*);
Fn g_tbl[11][17][17];

template <int OP, int A>
static void reg_row() {
  [&]<int... B>(std::integer_sequence<int, B...>) {
    ((g_tbl[OP][A][B] = &sub_op<OP, A, B>), ...);
  }(std::make_integer_sequence<int, 17>{});
}

template <int OP>
static void reg_op() {
  [&]<int... A>(std::integer_sequence<int, A...>) {
    (reg_row<OP, A>(), ...);
  }(std::make_integer_sequence<int, 17>{});
}

struct Init {
  Init() {
    reg_op<OP_NOOP>();
    reg_op<OP_ADD>();
    reg_op<OP_DIV>();
    reg_op<OP_RCP>();
    reg_op<OP_RCPRAW>();
    reg_op<OP_NDVI_DIV>();
    reg_op<OP_NDVI_RCP>();
    reg_op<OP_NDVI_RCPRAW>();
    reg_op<OP_NDVI_COUNT>();
    reg_op<OP_ADDFP>();
    reg_op<OP_CVTFP>();
  }
} g_init;

static inline double hsum_ps(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  return _mm_cvtss_f32(s);
}

static inline int32_t hsum_epi32(__m256i v) {
  __m128i lo = _mm256_castsi256_si128(v), hi = _mm256_extracti128_si256(v, 1);
  __m128i s = _mm_add_epi32(lo, hi);
  s = _mm_hadd_epi32(s, s);
  s = _mm_hadd_epi32(s, s);
  return _mm_cvtsi128_si32(s);
}

static inline double xreduce(__m256i x) {
  alignas(32) int32_t v[8];
  _mm256_store_si256((__m256i*)v, x);
  int64_t r = 0;
  for (int i = 0; i < 8; ++i) r ^= v[i];
  return (double)(r & 0xFFFF);
}

// Uncompressed baseline: same per-OutReg ladder over RAW uint16 (no decode).
template <int OP>
static double raw_loop(const uint16_t* a, const uint16_t* b, size_t length) {
  constexpr bool kFloat = (OP == OP_DIV || OP == OP_RCP || OP == OP_RCPRAW ||
                            OP == OP_NDVI_DIV || OP == OP_NDVI_RCP || OP == OP_NDVI_RCPRAW ||
                            OP == OP_ADDFP);
  __m256 farr[4] = {};
  __m256i x = _mm256_setzero_si256();
  size_t i = 0;
  if constexpr (kFloat) {
    // 4 independent accumulators, round-robin across 64 elements per outer iteration.
    // Breaks the cross-iteration serial add_ps dep chain (was 4096-deep, now 1024-deep
    // per accumulator), letting divide/rcp throughput dominate instead.
    __m256i xd = _mm256_setzero_si256();
    for (; i + 64 <= length; i += 64) {
      acc_op<OP>(_mm256_loadu_si256((const __m256i*)(a+i   )), _mm256_loadu_si256((const __m256i*)(b+i   )), farr[0], xd);
      acc_op<OP>(_mm256_loadu_si256((const __m256i*)(a+i+16)), _mm256_loadu_si256((const __m256i*)(b+i+16)), farr[1], xd);
      acc_op<OP>(_mm256_loadu_si256((const __m256i*)(a+i+32)), _mm256_loadu_si256((const __m256i*)(b+i+32)), farr[2], xd);
      acc_op<OP>(_mm256_loadu_si256((const __m256i*)(a+i+48)), _mm256_loadu_si256((const __m256i*)(b+i+48)), farr[3], xd);
    }
    farr[0] = _mm256_add_ps(_mm256_add_ps(farr[0], farr[1]), _mm256_add_ps(farr[2], farr[3]));
  }
  // Tail / non-float ops: single accumulator.
  for (; i < length; i += 16)
    acc_op<OP>(_mm256_loadu_si256((const __m256i*)(a+i)),
               _mm256_loadu_si256((const __m256i*)(b+i)), farr[0], x);
  if constexpr (OP == OP_NDVI_COUNT) return (double)(-hsum_epi32(x));
  return hsum_ps(farr[0]) + xreduce(x);
}

}  // namespace

// Sets the fixed-point threshold coefficients used by OP_NDVI_COUNT for the
// CURRENT THREAD. x is the NDVI threshold (typically in (-1,1)); SCALE chosen
// so a*K1, b*K2 (a,b<=65535) stay comfortably within int32 after vpmaddwd's
// implicit add of two such products (max ~1.07e9, well under 2^31-1).
extern "C" void ndvi2_set_count_threshold(float x) {
  constexpr float SCALE = 4096.0f;
  int16_t k1 = (int16_t)lrintf((1.0f - x) * SCALE);
  int16_t k2 = (int16_t)lrintf((1.0f + x) * SCALE);
  uint32_t packed = (uint32_t)(uint16_t)k1 | ((uint32_t)(uint16_t)(int16_t)(-k2) << 16);
  g_count_coef = _mm256_set1_epi32((int32_t)packed);
}

// Uncompressed 2-band aggregate over raw uint16 grids (same op ladder).
extern "C" double ndvi2_raw(const uint16_t* a, const uint16_t* b, size_t length,
                            int op) {
  switch (op) {
    case 0: return raw_loop<0>(a, b, length);
    case 1: return raw_loop<1>(a, b, length);
    case 2: return raw_loop<2>(a, b, length);
    case 3: return raw_loop<3>(a, b, length);
    case 4: return raw_loop<4>(a, b, length);
    case 5: return raw_loop<5>(a, b, length);
    case 6: return raw_loop<6>(a, b, length);
    case 7: return raw_loop<7>(a, b, length);
    case 9:  return raw_loop<9>(a, b, length);
    case 10: return raw_loop<10>(a, b, length);
    default: return raw_loop<8>(a, b, length);
  }
}

// Decode a block-pair (encoded simdcomp_fused: [madd_safe:1][bs:num_sb][payload])
// in lock-step and aggregate. op = OP_NOOP|OP_ADD|OP_DIV. length = elems/block.
extern "C" double ndvi2_indep(const uint8_t* encA, const uint8_t* encB,
                              size_t length, int op) {
  const size_t num_sb = length / 256;
  const uint8_t* bsA = encA + 1;
  const uint8_t* inA = bsA + num_sb;
  const uint8_t* bsB = encB + 1;
  const uint8_t* inB = bsB + num_sb;
  const Fn* tbl = &g_tbl[op][0][0];
  __m256 f = _mm256_setzero_ps();
  __m256i x = _mm256_setzero_si256();
  for (size_t k = 0; k < num_sb; ++k) {
    const uint32_t ba = bsA[k], bb = bsB[k];
    tbl[ba * 17 + bb](reinterpret_cast<const __m256i*>(inA),
                      reinterpret_cast<const __m256i*>(inB), &f, &x);
    inA += (size_t)ba * sizeof(__m256i);
    inB += (size_t)bb * sizeof(__m256i);
  }
  if (op == OP_NDVI_COUNT) return (double)(-hsum_epi32(x));
  return hsum_ps(f) + xreduce(x);
}
