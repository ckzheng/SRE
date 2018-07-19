#ifndef _XNN_AVX2_H
#define _XNN_AVX2_H
#include<cmath>
#include<algorithm>
#include<iostream>
#include<cstdio>
//#ifdef USE_AVX2
#include"immintrin.h"

inline void macadd_avx2(const __m256i &wtik, const short *vcolt,
                        __m256i &sum8) {
  const __m256i vkt = _mm256_loadu_si256((__m256i *)vcolt);
  sum8 = _mm256_add_epi32(sum8, _mm256_madd_epi16(wtik, vkt));
}

inline void dotprod_avx2(const short *wtrowi, const short *vcolt, int *usij,
                         const size_t dim) {
  __m256i wtik[4], colk[4];
  __m256i sum8 = _mm256_set1_epi16(0);
  __m256i sum[8];

  size_t k0 = 0;

  for (k0 = 0; k0 + 128 <= dim; k0 += 128) {
#if 1
    wtik[0] = _mm256_loadu_si256((__m256i *)&wtrowi[k0 + 0]);
    colk[0] = _mm256_loadu_si256((__m256i *)&vcolt[k0 + 0]);
    sum[0] = _mm256_madd_epi16(wtik[0], colk[0]);

    wtik[1] = _mm256_loadu_si256((__m256i *)&wtrowi[k0 + 16]);
    colk[1] = _mm256_loadu_si256((__m256i *)&vcolt[k0 + 16]);
    sum[1] = _mm256_madd_epi16(wtik[1], colk[1]);

    wtik[2] = _mm256_loadu_si256((__m256i *)&wtrowi[k0 + 32]);
    colk[2] = _mm256_loadu_si256((__m256i *)&vcolt[k0 + 32]);
    sum[2] = _mm256_madd_epi16(wtik[2], colk[2]);

    wtik[3] = _mm256_loadu_si256((__m256i *)&wtrowi[k0 + 48]);
    colk[3] = _mm256_loadu_si256((__m256i *)&vcolt[k0 + 48]);
    sum[3] = _mm256_madd_epi16(wtik[3], colk[3]);

    wtik[0] = _mm256_loadu_si256((__m256i *)&wtrowi[k0 + 64]);
    colk[0] = _mm256_loadu_si256((__m256i *)&vcolt[k0 + 64]);
    sum[4] = _mm256_madd_epi16(wtik[0], colk[0]);

    wtik[1] = _mm256_loadu_si256((__m256i *)&wtrowi[k0 + 80]);
    colk[1] = _mm256_loadu_si256((__m256i *)&vcolt[k0 + 80]);
    sum[5] = _mm256_madd_epi16(wtik[1], colk[1]);

    wtik[2] = _mm256_loadu_si256((__m256i *)&wtrowi[k0 + 96]);
    colk[2] = _mm256_loadu_si256((__m256i *)&vcolt[k0 + 96]);
    sum[6] = _mm256_madd_epi16(wtik[2], colk[2]);

    wtik[3] = _mm256_loadu_si256((__m256i *)&wtrowi[k0 + 112]);
    colk[3] = _mm256_loadu_si256((__m256i *)&vcolt[k0 + 112]);
    sum[7] = _mm256_madd_epi16(wtik[3], colk[3]);

    colk[0] = _mm256_add_epi32(sum[0], sum[1]);
    colk[1] = _mm256_add_epi32(sum[2], sum[3]);
    wtik[0] = _mm256_add_epi32(colk[0], colk[1]);

    colk[0] = _mm256_add_epi32(sum[0], sum[1]);
    colk[1] = _mm256_add_epi32(sum[2], sum[3]);
    wtik[0] = _mm256_add_epi32(colk[0], colk[1]);

    wtik[1] = _mm256_add_epi32(sum[4], sum[5]);
    wtik[2] = _mm256_add_epi32(sum[6], sum[7]);
    colk[0] = _mm256_add_epi32(wtik[1], wtik[2]);
    wtik[3] = _mm256_add_epi32(colk[0], wtik[0]);

    sum8 = _mm256_add_epi32(sum8, wtik[3]);
#endif
  }

  for (; k0 + 64 <= dim; k0 += 64) {
    wtik[0] = _mm256_loadu_si256((__m256i *)&wtrowi[k0 + 0]);
    colk[0] = _mm256_loadu_si256((__m256i *)&vcolt[k0 + 0]);
    sum[0] = _mm256_madd_epi16(wtik[0], colk[0]);

    wtik[1] = _mm256_loadu_si256((__m256i *)&wtrowi[k0 + 16]);
    colk[1] = _mm256_loadu_si256((__m256i *)&vcolt[k0 + 16]);
    sum[1] = _mm256_madd_epi16(wtik[1], colk[1]);

    wtik[2] = _mm256_loadu_si256((__m256i *)&wtrowi[k0 + 32]);
    colk[2] = _mm256_loadu_si256((__m256i *)&vcolt[k0 + 32]);
    sum[2] = _mm256_madd_epi16(wtik[2], colk[2]);

    wtik[3] = _mm256_loadu_si256((__m256i *)&wtrowi[k0 + 48]);
    colk[3] = _mm256_loadu_si256((__m256i *)&vcolt[k0 + 48]);
    sum[3] = _mm256_madd_epi16(wtik[3], colk[3]);

    colk[0] = _mm256_add_epi32(sum[0], sum[1]);
    colk[1] = _mm256_add_epi32(sum[2], sum[3]);
    wtik[0] = _mm256_add_epi32(colk[0], colk[1]);

    sum8 = _mm256_add_epi32(sum8, wtik[0]);
  }

  for (; k0 + 32 <= dim; k0 += 32) {
    wtik[0] = _mm256_loadu_si256((__m256i *)&wtrowi[k0 + 0]);
    colk[0] = _mm256_loadu_si256((__m256i *)&vcolt[k0 + 0]);
    sum[0] = _mm256_madd_epi16(wtik[0], colk[0]);

    wtik[1] = _mm256_loadu_si256((__m256i *)&wtrowi[k0 + 16]);
    colk[1] = _mm256_loadu_si256((__m256i *)&vcolt[k0 + 16]);
    sum[1] = _mm256_madd_epi16(wtik[1], colk[1]);


    colk[0] = _mm256_add_epi32(sum[0], sum[1]);

    sum8 = _mm256_add_epi32(sum8, colk[0]);
  }


#ifdef _MSC_VER
  __declspec(align(32)) int sums[8];
#else
  __attribute__((aligned(32))) int sums[8];
#endif
  _mm256_store_si256((__m256i *) &sums[0], sum8);
  *usij += sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] +
           sums[7];

  for (; k0 < dim; k0++) {
    *usij += wtrowi[k0] * vcolt[k0];
  }

}

inline void dotprod4_avx2(const short *wtrowi, const short *vcolt,
                          const size_t cols4stride, int *usij, const size_t usijstride,
                          const size_t dim) {
  const short *p0 = vcolt, *p1 = p0 + cols4stride,
               *p2 = p1 + cols4stride, *p3 = p2 + cols4stride;

  __m256i sumi4[4], wtik;
  sumi4[0] = sumi4[1] = sumi4[2] = sumi4[3] = _mm256_set1_epi16(0);

  size_t k0;
#define MADD4(offset) {wtik = _mm256_lddqu_si256((__m256i *)&wtrowi[k0+offset]);\
                            macadd_avx2(wtik, p0+offset, sumi4[0]);\
                            macadd_avx2(wtik, p1+offset, sumi4[1]);\
                            macadd_avx2(wtik, p2+offset, sumi4[2]);\
                            macadd_avx2(wtik, p3+offset, sumi4[3]);}

  for (k0 = 0; k0 + 128 <= dim;
       k0 += 128, p0 += 128, p1 += 128, p2 += 128,
       p3 += 128) {  // 8-way loop unrolled (we overflow after 8), for 8 components
    MADD4(0);
    MADD4(16);
    MADD4(32);
    MADD4(48);
    MADD4(64);
    MADD4(80);
    MADD4(96);
    MADD4(112);
  }

  for (; k0 + 64 <= dim; k0 += 64, p0 += 64, p1 += 64, p2 += 64, p3 += 64) {
    MADD4(0);
    MADD4(16);
    MADD4(32);
    MADD4(48);
  }

  for (; k0 + 32 <= dim; k0 += 32, p0 += 32, p1 += 32, p2 += 32, p3 += 32) {
    MADD4(0);
    MADD4(16);
  }

  for (; k0 + 16 <= dim; k0 += 16, p0 += 16, p1 += 16, p2 += 16, p3 += 16) {
    MADD4(0);
  }

  for (; k0 < dim; k0++, p0++, p1++, p2++, p3++) {
    *(usij + 0 * usijstride) += wtrowi[k0] * (*p0);
    *(usij + 1 * usijstride) += wtrowi[k0] * (*p1);
    *(usij + 2 * usijstride) += wtrowi[k0] * (*p2);
    *(usij + 3 * usijstride) += wtrowi[k0] * (*p3);
  }

#ifdef _MSC_VER
  __declspec(align(32)) int sums[8];
#else
  __attribute__((aligned(32))) int sums[8];
#endif

#define RS(offset) {_mm256_store_si256((__m256i *) sums, sumi4[offset]);\
                    *(usij + offset * usijstride) += sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7];}
  RS(0);
  RS(1);
  RS(2);
  RS(3);
}


#if defined(__INTEL_COMPILER)
#define SIGMOD_AVX2(offset) {x[offset] = _mm256_loadu_ps(pCol);x[offset]=_mm256_sub_ps(zero,x[offset]);\
       x[offset]=_mm256_min_ps(x[offset], explimit);x[offset]=_mm256_exp_ps(x[offset]);\
       x[offset]=_mm256_add_ps(x[offset], one);x[offset]=_mm256_div_ps(one, x[offset]);_mm256_storeu_ps(pCol,x[offset]);}
inline void sigmod_avx2_svml(float *pCol, size_t len) {
#define EXPLIMIT 88.722008f
  __m256 x[0], zero = _mm256_set1_ps(0), one = _mm256_set1_ps(1.0f);
  __m256 explimit = _mm256_set1_ps(88.722008f);

  size_t row = 0;

  for (; row + 8 <= len; row += 8, pCol += 8) {
    SIGMOD_AVX2(0);
  }

  for (; row < len; row++, pCol++) {
    float expbuffer = std::min<float>(-*pCol, EXPLIMIT);
    expbuffer = expf(expbuffer);
    *pCol = 1.0f / (1.0f + expbuffer);
  }
}
#endif

inline void scaleadd_avx2(const float scale, float *usij, const int *col,
                          const size_t dim) {
  __m256 u, c, s = _mm256_set1_ps(scale);
  size_t d = 0;
  for (; d + 8 <= dim; d += 8, usij += 8, col += 8) {
    c = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i *)(col)));
    u = _mm256_loadu_ps(usij);
    u = _mm256_fmadd_ps(s,c,u);
    _mm256_storeu_ps(usij,u);
  }

  for (; d < dim; ++d, ++col, ++usij) {
    *usij += *col * scale;
  }
}

inline void dotprod_avx2(const float *row, const float *col, float *usij,
                         const size_t dim) {
  __m256 sum = _mm256_set1_ps(0);
  __m256 r, c;
  size_t d;
  
  for (d = 0; d + 32 <= dim;
       d += 32, row += 32, col += 32) { // 8-way loop unrolling
    r = _mm256_load_ps(row);
    c = _mm256_load_ps(col);
	sum = _mm256_fmadd_ps(c, r, sum);

    r = _mm256_load_ps(row + 8);
    c = _mm256_load_ps(col + 8);
	sum = _mm256_fmadd_ps(c, r, sum);

    r = _mm256_load_ps(row + 16);
    c = _mm256_load_ps(col + 16);
	sum = _mm256_fmadd_ps(c, r, sum);

    r = _mm256_load_ps(row + 24);
    c = _mm256_load_ps(col + 24);
	sum = _mm256_fmadd_ps(c, r, sum);   
  }

  for (; d < dim; ++d, ++row, ++col) {
    r = _mm256_load_ps(row);
    c = _mm256_load_ps(col);
	sum = _mm256_fmadd_ps(c, r, sum);
  }

#ifdef _MSC_VER
  __declspec(align(32)) float sums[8];
#else
  __attribute__((aligned(32))) int sums[8];
#endif

  _mm256_store_ps(sums, sum);
  (*usij) += sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7];

}

inline void dotprod4_avx2(const float *row, const float *col,
                          const size_t cols4stride, int *usij, const size_t usijstride,
                          const size_t dim) {

  const float *p = row, *p0 = col, *p1 = p0 + cols4stride,
               *p2 = p1 + cols4stride, *p3 = p2 + cols4stride;

  __m256 sum[4], r, c;
  sum[0] = sum[1] = sum[2] = sum[3] = _mm256_set1_ps(0);

  size_t d;
  for (d = 0; d + 32 <= dim;
       d += 32, p += 32, p0 += 32, p1 += 32, p2 += 32,
       p3 += 32) {  // 8-way loop unrolled (we overflow after 8), for 8 components
    r = _mm256_load_ps(p);

    c = _mm256_load_ps(p0);
	sum[0] = _mm256_fmadd_ps(r, c, sum[0]);

	c = _mm256_load_ps(p1);
	sum[1] = _mm256_fmadd_ps(r, c, sum[1]);

	c = _mm256_load_ps(p2);
	sum[2] = _mm256_fmadd_ps(r, c, sum[2]);

	c = _mm256_load_ps(p3);
    sum[3] = _mm256_fmadd_ps(r,c,sum[3]);

	r = _mm256_load_ps(p + 8);

	c = _mm256_load_ps(p0 + 8);
    sum[0] = _mm256_fmadd_ps(r,c,sum[0]);

	c = _mm256_load_ps(p1 + 8);
    sum[1] = _mm256_fmadd_ps(r,c,sum[1]);

	c = _mm256_load_ps(p2 + 8);
    sum[2] = _mm256_fmadd_ps(r,c,sum[2]);

	c = _mm256_load_ps(p3 + 8);
    sum[3] = _mm256_fmadd_ps(r,c,sum[3]);

	r = _mm256_load_ps(p + 16);

	c = _mm256_load_ps(p0 + 16);
    sum[0] = _mm256_fmadd_ps(r,c,sum[0]);

    c = _mm256_load_ps(p1 + 16);
    sum[1] = _mm256_fmadd_ps(r,c,sum[1]);

    c = _mm256_load_ps(p2 + 16);
    sum[2] = _mm256_fmadd_ps(r,c,sum[2]);

    c = _mm256_load_ps(p3 + 16);
    sum[3] = _mm256_fmadd_ps(r,c,sum[3]);

    r = _mm256_load_ps(p + 24);

    c = _mm256_load_ps(p0 + 24);
    sum[0] = _mm256_fmadd_ps(r,c,sum[0]);

    c = _mm256_load_ps(p1 + 24);
    sum[1] = _mm256_fmadd_ps(r,c,sum[1]);

    c = _mm256_load_ps(p2 + 24);
    sum[2] = _mm256_fmadd_ps(r,c,sum[2]);

    c = _mm256_load_ps(p3 + 24);
    sum[3] = _mm256_fmadd_ps(r,c,sum[3]);   
  }

  for (; d < dim; p++, p0++, p1++, p2++, p3++) {
    c = _mm256_load_ps(p0);
    sum[0] = _mm256_fmadd_ps(r,c,sum[0]);

    c = _mm256_load_ps(p1);
    sum[1] = _mm256_fmadd_ps(r,c,sum[1]);

    c = _mm256_load_ps(p2);
    sum[2] = _mm256_fmadd_ps(r,c,sum[2]);

    c = _mm256_load_ps(p3);
    sum[3] = _mm256_fmadd_ps(r,c,sum[3]);
  }

#ifdef _MSC_VER
  __declspec(align(32)) float sums[32];
#else
  __attribute__((aligned(32))) int sums[32];
#endif

  for(int offset = 0; offset < 4; ++offset) {
    _mm256_store_ps(sums, sum[offset]);
    *(usij + offset * usijstride) += sums[0] + sums[1] + sums[2] + sums[3] +
                                     sums[4] + sums[5] + sums[6] + sums[7];
  }
}

//#endif



#endif

