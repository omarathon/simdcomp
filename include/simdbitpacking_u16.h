/**
 * uint16 SIMD bit-packing with fused sum aggregation.
 * This code is released under a BSD License.
 */
#ifndef SIMDBITPACKING_U16_H_
#define SIMDBITPACKING_U16_H_

#include <emmintrin.h>  /* SSE2 */
#include <tmmintrin.h>  /* SSSE3 for _mm_hadd_* */
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Pack 128 uint16 values at `bit` bits each into `bit` __m128i words.
 */
void simdpack_u16(const uint16_t *in, __m128i *out, const uint32_t bit);

/**
 * Fused unpack: processes 128 packed uint16 values, computing their sum
 * into `sum` (4 x int32 accumulator) without writing decoded values.
 */
void simdunpack_u16(const __m128i *in, uint16_t *out, const uint32_t bit,
                    __m128i *sum);

/**
 * Compute the maximum number of bits needed to represent any value in `in`.
 */
uint32_t maxbits_length_u16(const uint16_t *in, size_t length);

/**
 * How many bytes the packed representation of `length` values at `bit` bits
 * each will occupy.
 */
int simdpack_compressedbytes_u16(int length, const uint32_t bit);

/**
 * Pack `length` uint16 values (may be non-multiple of 128).
 * Returns pointer past the last written __m128i.
 */
__m128i *simdpack_length_u16(const uint16_t *in, size_t length, __m128i *out,
                              const uint32_t bit);

/**
 * Fused unpack of `length` uint16 values: computes total sum as uint32.
 * The decoded values are NOT written to `out` (fused mode).
 * Returns pointer past the last consumed __m128i.
 */
const __m128i *simdunpack_length_u16(const __m128i *in, size_t length,
                                      uint16_t *out, const uint32_t bit,
                                      uint32_t *outsum);

#ifdef __cplusplus
}
#endif

#endif /* SIMDBITPACKING_U16_H_ */
