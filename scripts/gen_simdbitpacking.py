#!/usr/bin/env python3
"""
Transform simdbitpacking.c to add compile-time SIMD_SUM_FUSED mode toggle.

The source file was hand-edited from the original simdcomp to comment out all
output stores and add aggregate_sums() calls.  This script adds #ifdef guards
so the file can be compiled in either mode:

  #define SIMD_SUM_FUSED   → sum-only (no output writes, original behaviour)
  (undefined)              → normal decode (output written, no sum)

Usage:
  python3 gen_simdbitpacking.py [--mode {fused,normal,guarded}] \
      [--input simdbitpacking.c] [--output simdbitpacking.c]

  fused   — emit the current sum-only code (no #ifdef, hardcoded fused)
  normal  — emit normal decode code (no #ifdef, hardcoded normal)
  guarded — (default) emit #ifdef SIMD_SUM_FUSED guarded code
"""

import argparse
import re
import sys


# ---------------------------------------------------------------------------
# Regex patterns for the two styles of commented-out store found in the file
#
# Style A (loop body, 4-OutReg form):
#   /* _mm_storeu_si128(out++, OutRegN); */
#   aggregate_sums(OutRegN, sum_lo, sum_hi);
#
# Style B (direct form):
#   /* _mm_storeu_si128(out++, OutReg); */
#   aggregate_sums(OutReg, sum_lo, sum_hi);
#
# Both are captured by a single pattern — the OutReg name may include digits.
# ---------------------------------------------------------------------------

# Matches:   <indent>/* _mm_storeu_si128(out++, <name>); */\n
#            <any_indent>aggregate_sums(<name>, sum_lo, sum_hi);
# Note: the two lines may have different indentation in the source file.
STORE_AGG_RE = re.compile(
    r'(?P<indent>[ \t]*)(?P<store>/\* _mm_storeu_si128\(out\+\+, (?P<reg>\w+)\); \*/)\n'
    r'[ \t]*aggregate_sums\((?P=reg), sum_lo, sum_hi\);',
    re.MULTILINE
)

# Matches the commented-out `out` declaration
OUT_DECL_RE = re.compile(
    r'[ \t]*/\* __m128i \*out = \(__m128i \*\)\(_out\); \*/'
)


def replacement_guarded(m):
    ind = m.group('indent')
    reg = m.group('reg')
    return (
        f'{ind}#ifdef SIMD_SUM_FUSED\n'
        f'{ind}aggregate_sums({reg}, sum_lo, sum_hi);\n'
        f'{ind}#else\n'
        f'{ind}_mm_storeu_si128(out++, {reg});\n'
        f'{ind}#endif'
    )


def replacement_normal(m):
    ind = m.group('indent')
    reg = m.group('reg')
    return f'{ind}_mm_storeu_si128(out++, {reg});'


def replacement_fused(m):
    ind = m.group('indent')
    reg = m.group('reg')
    return f'{ind}aggregate_sums({reg}, sum_lo, sum_hi);'


def out_decl_guarded(_m):
    return (
        '#ifndef SIMD_SUM_FUSED\n'
        '  __m128i *out = (__m128i *)(_out);\n'
        '#endif'
    )


def out_decl_normal(_m):
    return '  __m128i *out = (__m128i *)(_out);'


def out_decl_fused(_m):
    return '  /* out not used in fused mode */'


def transform(src, mode):
    # simdunpack_length / simdunpack_shortlength work correctly in all modes:
    # - normal: guarded simdunpack writes output; shortlength always writes;
    #           out-pointer advancement is correct; *outsum is ignored by callers
    # - fused:  guarded simdunpack accumulates sum only; shortlength writes
    #           remainder to out (needed for scalar sum loop); harmless
    if mode == 'guarded':
        src = STORE_AGG_RE.sub(replacement_guarded, src)
        src = OUT_DECL_RE.sub(out_decl_guarded, src)
    elif mode == 'normal':
        src = STORE_AGG_RE.sub(replacement_normal, src)
        src = OUT_DECL_RE.sub(out_decl_normal, src)
    else:  # fused — already in fused form; just clean up the comment style
        src = STORE_AGG_RE.sub(replacement_fused, src)
        # Leave the out decl comment as-is (it's already commented out)
    return src


def main():
    parser = argparse.ArgumentParser(description='Transform simdbitpacking.c')
    parser.add_argument('--mode', choices=['fused', 'normal', 'guarded'],
                        default='guarded')
    parser.add_argument('--input', default=None,
                        help='Input file (default: stdin)')
    parser.add_argument('--output', default=None,
                        help='Output file (default: stdout)')
    args = parser.parse_args()

    src = open(args.input).read() if args.input else sys.stdin.read()
    result = transform(src, args.mode)
    if args.output:
        open(args.output, 'w').write(result)
    else:
        sys.stdout.write(result)


if __name__ == '__main__':
    main()
