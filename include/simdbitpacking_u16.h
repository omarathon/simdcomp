/**
 * uint16 SIMD bit-packing with fused sum aggregation.
 * This code is released under a BSD License.
 */
#ifndef SIMDBITPACKING_U16_H_
#define SIMDBITPACKING_U16_H_

#include <immintrin.h>  /* AVX2 */
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Pack 256 uint16 values at `bit` bits each into `bit` __m256i words.
 */
void simdpack_u16(const uint16_t *in, __m256i *out, const uint32_t bit);

/**
 * Fused unpack: processes 256 packed uint16 values, computing their sum
 * into `sum` (8 x int32 accumulator) without writing decoded values.
 */
void simdunpack_u16(const __m256i *in, uint16_t *out, const uint32_t bit,
                    __m256i *sum);

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
 * Pack `length` uint16 values (may be non-multiple of 256).
 * Returns pointer past the last written __m256i.
 */
__m256i *simdpack_length_u16(const uint16_t *in, size_t length, __m256i *out,
                              const uint32_t bit);

/**
 * Fused unpack of `length` uint16 values: computes total sum as uint32.
 * The decoded values are NOT written to `out` (fused mode).
 * Returns pointer past the last consumed __m256i.
 */
const __m256i *simdunpack_length_u16(const __m256i *in, size_t length,
                                      uint16_t *out, const uint32_t bit,
                                      uint32_t *outsum);

/**
 * Fused unpack + LOCAL delta+zigzag decode.
 *
 * Assumes the encoded data is the result of bit-packing a stream of
 * zigzag-encoded deltas where prev resets to 0 every 16 elements (each OutReg
 * is an independent prefix-sum window). Per-OutReg the SIMD pipeline runs:
 *   zigzag_dec -> prefix_sum -> aggregate.
 * No inter-OutReg / inter-block carry. The tail (length % 256) is handled by
 * scalar unpack into `out` followed by a scalar un-zigzag + prefix sum with
 * prev resetting every 16 elements.
 *
 * Decoded values are NOT written to `out` (only the sum is). `out` is used
 * as scratch for the scalar tail (it must have room for `length % 256`
 * uint16s plus the usual fused-mode overflow slots).
 */
const __m256i *simdunpack_length_u16_delta_local(const __m256i *in,
                                                  size_t length, uint16_t *out,
                                                  const uint32_t bit,
                                                  uint32_t *outsum);

/**
 * Fused unpack + CARRY delta+zigzag decode.
 *
 * Assumes the encoded data is the result of bit-packing a stream of
 * zigzag-encoded deltas with a single continuous prev across the whole input
 * (prev[-1] = 0). Per-OutReg the SIMD pipeline runs:
 *   zigzag_dec -> prefix_sum -> +carry -> update carry -> aggregate.
 * The broadcast-carry __m256i is threaded across OutRegs and blocks.
 * The tail (length % 256) is handled by scalar unpack into `out` followed by
 * a scalar un-zigzag + prefix sum seeded from the carry of the last block.
 */
const __m256i *simdunpack_length_u16_delta_carry(const __m256i *in,
                                                  size_t length, uint16_t *out,
                                                  const uint32_t bit,
                                                  uint32_t *outsum);

#ifdef __cplusplus
}
#endif

#endif /* SIMDBITPACKING_U16_H_ */
