#include"am/xnn_sse.h"
#include <stdint.h>
#include <cmath>
#include <algorithm>
#if defined(USE_SSE41)
void quantize_sse(int16_t *sv, const float *fv, size_t len,
                  float invscale, int16_t QMAX) {
  float QMAX_U = QMAX * 1.0f, QMAX_L = -(QMAX + 1.0f);
  __m128 y, x_multi_y, v;
  __m128 x = _mm_load_ps1(&invscale);
  __m128 MIN = _mm_load_ps1(&QMAX_L);
  __m128 MAX = _mm_load_ps1(&QMAX_U);

  size_t i;
  for (i = 0; i + 4 <= len; i += 4, sv += 4, fv += 4) {
    y = _mm_loadu_ps(fv);
    x_multi_y = _mm_mul_ps(x, y);
    v = _mm_round_ps(x_multi_y, 0x00);

    v = _mm_max_ps(v, MIN);
    v = _mm_min_ps(v, MAX);
    __m128i vi = _mm_cvttps_epi32(v);
    __m128i vi16 = _mm_packs_epi32(vi, vi);
    _mm_storel_epi64(reinterpret_cast<__m128i *>(sv), vi16);
  }
  for (; i < len; ++i, ++sv, ++fv) {
    float r = round(fv[0] * invscale);
    r = std::max(r, QMAX_L);
    r = std::min(r, QMAX_U);
    sv[0] = static_cast<int16_t>(r);
  }
}

void quantize_non_symmetric_sse(int8_t *sv, const float *fv, size_t len,
                                float minf, float invscale, int8_t QMAX,
                                int &curr_col_sum) {
  float QMAX_U = QMAX * 1.0f, QMAX_L = -(QMAX + 1.0f);
  float neg_minf = -minf;
  __m128 y, x_multi_y, v;
  __m128 x = _mm_load_ps1(&invscale);
  __m128 neg_minf_128 = _mm_load_ps1(&neg_minf);
  __m128 MIN = _mm_load_ps1(&QMAX_L);
  __m128 MAX = _mm_load_ps1(&QMAX_U);

  size_t i;
  for (i = 0; i + 8 <= len;
       i += 4, sv += 4, fv +=
         4) {   // i + 8 <= len for using _mm_storel_epi64 to store 8bit integers
    y = _mm_loadu_ps(fv);
    y = _mm_add_ps(y, neg_minf_128);
    x_multi_y = _mm_mul_ps(x, y);
    x_multi_y = _mm_add_ps(x_multi_y, MIN);
    v = _mm_round_ps(x_multi_y, 0x00);

    v = _mm_max_ps(v, MIN);
    v = _mm_min_ps(v, MAX);
    __m128i vi = _mm_cvttps_epi32(v);
    __m128i vi16 = _mm_packs_epi32(vi, vi);
    __m128i vi8 = _mm_packs_epi16(vi16, vi16);
    _mm_storel_epi64(reinterpret_cast<__m128i *>(sv), vi8);

    curr_col_sum += static_cast<int>(sv[0]);
    curr_col_sum += static_cast<int>(sv[1]);
    curr_col_sum += static_cast<int>(sv[2]);
    curr_col_sum += static_cast<int>(sv[4]);
  }
  for (; i < len; ++i, ++sv, ++fv) {
    float r = round((fv[0] + neg_minf) * invscale + QMAX_L);
    r = std::max(r, QMAX_L);
    r = std::min(r, QMAX_U);
    sv[0] = static_cast<int8_t>(r);
    curr_col_sum += (int)r;
  }
}

void quantize_non_symmetric_sse(uint8_t *sv, const float *fv, size_t len,
                                float minf, float invscale, uint8_t QMAX,
                                uint32_t &curr_col_sum) {
  float QMAX_U = QMAX * 1.0f, QMAX_L = 0.0f;
  float neg_minf = -minf;
  __m128 y, x_multi_y, v;
  __m128 x = _mm_load_ps1(&invscale);
  __m128 neg_minf_128 = _mm_load_ps1(&neg_minf);
  __m128 MIN = _mm_load_ps1(&QMAX_L);
  __m128 MAX = _mm_load_ps1(&QMAX_U);

  size_t i;
  for (i = 0; i + 8 <= len;
       i += 4, sv += 4,
       fv += 4) {   // i + 8 <= len for using _mm_storel_epi64 to store 8bit integers
    y = _mm_loadu_ps(fv);
    y = _mm_add_ps(y, neg_minf_128);
    x_multi_y = _mm_mul_ps(x, y);
    v = _mm_round_ps(x_multi_y, 0x00);

    v = _mm_max_ps(v, MIN);
    v = _mm_min_ps(v, MAX);
    __m128i vi = _mm_cvttps_epi32(v);
    __m128i vi16 = _mm_packs_epi32(vi, vi);
    __m128i vi8 = _mm_packus_epi16(vi16, vi16);
    _mm_storel_epi64((__m128i *)sv, vi8);

    curr_col_sum += static_cast<int>(sv[0]);
    curr_col_sum += static_cast<int>(sv[1]);
    curr_col_sum += static_cast<int>(sv[2]);
    curr_col_sum += static_cast<int>(sv[4]);
  }
  for (; i < len; ++i, ++sv, ++fv) {
    float r = round((fv[0] + neg_minf) * invscale + QMAX_L);
    r = std::max(r, QMAX_L);
    r = std::min(r, QMAX_U);
    sv[0] = static_cast<uint8_t>(r);
    curr_col_sum += (int)r;
  }
}

inline void macadd(const __m128i &wtik, const uint8_t *vcolt, __m128i &sum4) {
  __m128i vkt = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i *)vcolt));
  sum4 = _mm_add_epi32(sum4, _mm_madd_epi16(wtik, vkt));
}

void dotprod_sse(const uint8_t *wtrowi, const uint8_t *vcolt, int *usij,
                 const size_t dim) {
  __m128i sum4 = _mm_set1_epi8(0), wtik;
  size_t k0 = 0;

  for (k0 = 0; k0 + 64 <= dim;
       k0 += 64, vcolt +=
         64) {  // 8-way loop unrolled (we overflow after 8), for 8 components
    wtik = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i *)&wtrowi[k0]));
    macadd(wtik, vcolt, sum4);
    wtik = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i *)&wtrowi[k0 + 8]));
    macadd(wtik, vcolt + 8, sum4);
    wtik = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i *)&wtrowi[k0 + 16]));
    macadd(wtik, vcolt + 16, sum4);
    wtik = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i *)&wtrowi[k0 + 24]));
    macadd(wtik, vcolt + 24, sum4);
    wtik = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i *)&wtrowi[k0 + 32]));
    macadd(wtik, vcolt + 32, sum4);
    wtik = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i *)&wtrowi[k0 + 40]));
    macadd(wtik, vcolt + 40, sum4);
    wtik = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i *)&wtrowi[k0 + 48]));
    macadd(wtik, vcolt + 48, sum4);
    wtik = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i *)&wtrowi[k0 + 56]));
    macadd(wtik, vcolt + 56, sum4);
  }

  for (; k0 + 8 <= dim; k0 += 8, vcolt += 8) {
    wtik = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i *)&wtrowi[k0]));
    macadd(wtik, vcolt, sum4);
  }

  for (; k0 < dim; k0++, vcolt++) {
    *usij += (wtrowi[k0]) * (*vcolt);
  }

#ifdef _MSC_VER
  __declspec(align(16)) int sums[4];
#else
  __attribute__((aligned(16))) int sums[4];
#endif
  _mm_store_si128((__m128i *) &sums[0], sum4);
  *usij += sums[0] + sums[1] + sums[2] + sums[3];
}

void dotprod4_sse(const uint8_t *wtrowi, const uint8_t *vcolt,
                  const size_t cols4stride, int *usij, const size_t usijstride,
                  const size_t dim) {
  const uint8_t *p0 = vcolt, *p1 = p0 + cols4stride,
                 *p2 = p1 + cols4stride, *p3 = p2 + cols4stride;

  __m128i sumi4[4], wtik;
  sumi4[0] = sumi4[1] = sumi4[2] = sumi4[3] = _mm_set1_epi8(0);

  size_t k0;
  for (k0 = 0; k0 + 64 <= dim;
       k0 += 64, p0 += 64, p1 += 64, p2 += 64,
       p3 += 64) {  // 8-way loop unrolled (we overflow after 8), for 8 components
    wtik = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i *)&wtrowi[k0]));
    macadd(wtik, p0, sumi4[0]);
    macadd(wtik, p1, sumi4[1]);
    macadd(wtik, p2, sumi4[2]);
    macadd(wtik, p3, sumi4[3]);

    wtik = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i *)&wtrowi[k0 + 8]));
    macadd(wtik, p0 + 8, sumi4[0]);
    macadd(wtik, p1 + 8, sumi4[1]);
    macadd(wtik, p2 + 8, sumi4[2]);
    macadd(wtik, p3 + 8, sumi4[3]);

    wtik = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i *)&wtrowi[k0 + 16]));
    macadd(wtik, p0 + 16, sumi4[0]);
    macadd(wtik, p1 + 16, sumi4[1]);
    macadd(wtik, p2 + 16, sumi4[2]);
    macadd(wtik, p3 + 16, sumi4[3]);

    wtik = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i *)&wtrowi[k0 + 24]));
    macadd(wtik, p0 + 24, sumi4[0]);
    macadd(wtik, p1 + 24, sumi4[1]);
    macadd(wtik, p2 + 24, sumi4[2]);
    macadd(wtik, p3 + 24, sumi4[3]);

    wtik = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i *)&wtrowi[k0 + 32]));
    macadd(wtik, p0 + 32, sumi4[0]);
    macadd(wtik, p1 + 32, sumi4[1]);
    macadd(wtik, p2 + 32, sumi4[2]);
    macadd(wtik, p3 + 32, sumi4[3]);

    wtik = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i *)&wtrowi[k0 + 40]));
    macadd(wtik, p0 + 40, sumi4[0]);
    macadd(wtik, p1 + 40, sumi4[1]);
    macadd(wtik, p2 + 40, sumi4[2]);
    macadd(wtik, p3 + 40, sumi4[3]);

    wtik = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i *)&wtrowi[k0 + 48]));
    macadd(wtik, p0 + 48, sumi4[0]);
    macadd(wtik, p1 + 48, sumi4[1]);
    macadd(wtik, p2 + 48, sumi4[2]);
    macadd(wtik, p3 + 48, sumi4[3]);

    wtik = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i *)&wtrowi[k0 + 56]));
    macadd(wtik, p0 + 56, sumi4[0]);
    macadd(wtik, p1 + 56, sumi4[1]);
    macadd(wtik, p2 + 56, sumi4[2]);
    macadd(wtik, p3 + 56, sumi4[3]);

  }

  for (; k0 + 8 <= dim; k0 += 8, p0 += 8, p1 += 8, p2 += 8, p3 += 8) {
    wtik = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i *)&wtrowi[k0]));
    macadd(wtik, p0, sumi4[0]);
    macadd(wtik, p1, sumi4[1]);
    macadd(wtik, p2, sumi4[2]);
    macadd(wtik, p3, sumi4[3]);
  }

  for (; k0 < dim; k0++, p0++, p1++, p2++, p3++) {
    *(usij + 0 * usijstride) += (wtrowi[k0]) * (*p0);
    *(usij + 1 * usijstride) += (wtrowi[k0]) * (*p1);
    *(usij + 2 * usijstride) += (wtrowi[k0]) * (*p2);
    *(usij + 3 * usijstride) += (wtrowi[k0]) * (*p3);
  }

#ifdef _MSC_VER
  __declspec(align(16)) int sums[4];
#else
  __attribute__((aligned(16))) int sums[4];
#endif

  _mm_store_si128((__m128i *) sums, sumi4[0]);
  *(usij + 0 * usijstride) += sums[0] + sums[1] + sums[2] + sums[3];
  _mm_store_si128((__m128i *) sums, sumi4[1]);
  *(usij + 1 * usijstride) += sums[0] + sums[1] + sums[2] + sums[3];
  _mm_store_si128((__m128i *) sums, sumi4[2]);
  *(usij + 2 * usijstride) += sums[0] + sums[1] + sums[2] + sums[3];
  _mm_store_si128((__m128i *) sums, sumi4[3]);
  *(usij + 3 * usijstride) += sums[0] + sums[1] + sums[2] + sums[3];
}


#endif



inline void macadd(const __m128i &wtik, const int16_t *vcolt, __m128i &sum4) {
  //const __m128i wtik = _mm_loadu_si128((__m128i *)&wtrowik0[dk]);// k...k+7
  const __m128i vkt = _mm_loadu_si128((__m128i *)vcolt);     // k...k+7
  sum4 = _mm_add_epi32(sum4, _mm_madd_epi16(wtik, vkt));
}



void dotprod_sse(const int16_t *wtrowi, const int16_t *vcolt, int *usij,
                 const size_t dim) {
  __m128i sum4 = _mm_set1_epi16(0), wtik;
  size_t k0 = 0;

  for (k0 = 0; k0 + 64 <= dim;
       k0 += 64, vcolt +=
         64) {  // 8-way loop unrolled (we overflow after 8), for 8 components
    wtik = _mm_loadu_si128((__m128i *)&wtrowi[k0]);
    macadd(wtik, vcolt, sum4);
    wtik = _mm_loadu_si128((__m128i *)&wtrowi[k0 + 8]);
    macadd(wtik, vcolt + 8, sum4);
    wtik = _mm_loadu_si128((__m128i *)&wtrowi[k0 + 16]);
    macadd(wtik, vcolt + 16, sum4);
    wtik = _mm_loadu_si128((__m128i *)&wtrowi[k0 + 24]);
    macadd(wtik, vcolt + 24, sum4);
    wtik = _mm_loadu_si128((__m128i *)&wtrowi[k0 + 32]);
    macadd(wtik, vcolt + 32, sum4);
    wtik = _mm_loadu_si128((__m128i *)&wtrowi[k0 + 40]);
    macadd(wtik, vcolt + 40, sum4);
    wtik = _mm_loadu_si128((__m128i *)&wtrowi[k0 + 48]);
    macadd(wtik, vcolt + 48, sum4);
    wtik = _mm_loadu_si128((__m128i *)&wtrowi[k0 + 56]);
    macadd(wtik, vcolt + 56, sum4);
  }

  for (; k0 + 8 <= dim; k0 += 8, vcolt += 8) {
    wtik = _mm_loadu_si128((__m128i *)&wtrowi[k0]);
    macadd(wtik, vcolt, sum4);
  }

  for (; k0 < dim; k0++, vcolt++) {
    *usij += wtrowi[k0] * (*vcolt);
  }

#ifdef _MSC_VER
  __declspec(align(16)) int sums[4];
#else
  __attribute__((aligned(16))) int sums[4];
#endif
  _mm_store_si128((__m128i *) &sums[0], sum4);
  *usij += sums[0] + sums[1] + sums[2] + sums[3];
}
void dotprod4_sse(const int16_t *wtrowi, const int16_t *vcolt,
                  const size_t cols4stride, int *usij, const size_t usijstride,
                  const size_t dim) {
  const int16_t *p0 = vcolt, *p1 = p0 + cols4stride,
               *p2 = p1 + cols4stride, *p3 = p2 + cols4stride;

  __m128i sumi4[4], wtik;
  sumi4[0] = sumi4[1] = sumi4[2] = sumi4[3] = _mm_set1_epi16(0);

  size_t k0;
  for (k0 = 0; k0 + 64 <= dim;
       k0 += 64, p0 += 64, p1 += 64, p2 += 64,
       p3 += 64) {  // 8-way loop unrolled (we overflow after 8), for 8 components
    wtik = _mm_loadu_si128((__m128i *)&wtrowi[k0]);
    macadd(wtik, p0, sumi4[0]);
    macadd(wtik, p1, sumi4[1]);
    macadd(wtik, p2, sumi4[2]);
    macadd(wtik, p3, sumi4[3]);

    wtik = _mm_loadu_si128((__m128i *)&wtrowi[k0 + 8]);
    macadd(wtik, p0 + 8, sumi4[0]);
    macadd(wtik, p1 + 8, sumi4[1]);
    macadd(wtik, p2 + 8, sumi4[2]);
    macadd(wtik, p3 + 8, sumi4[3]);

    wtik = _mm_loadu_si128((__m128i *)&wtrowi[k0 + 16]);
    macadd(wtik, p0 + 16, sumi4[0]);
    macadd(wtik, p1 + 16, sumi4[1]);
    macadd(wtik, p2 + 16, sumi4[2]);
    macadd(wtik, p3 + 16, sumi4[3]);

    wtik = _mm_loadu_si128((__m128i *)&wtrowi[k0 + 24]);
    macadd(wtik, p0 + 24, sumi4[0]);
    macadd(wtik, p1 + 24, sumi4[1]);
    macadd(wtik, p2 + 24, sumi4[2]);
    macadd(wtik, p3 + 24, sumi4[3]);

    wtik = _mm_loadu_si128((__m128i *)&wtrowi[k0 + 32]);
    macadd(wtik, p0 + 32, sumi4[0]);
    macadd(wtik, p1 + 32, sumi4[1]);
    macadd(wtik, p2 + 32, sumi4[2]);
    macadd(wtik, p3 + 32, sumi4[3]);

    wtik = _mm_loadu_si128((__m128i *)&wtrowi[k0 + 40]);
    macadd(wtik, p0 + 40, sumi4[0]);
    macadd(wtik, p1 + 40, sumi4[1]);
    macadd(wtik, p2 + 40, sumi4[2]);
    macadd(wtik, p3 + 40, sumi4[3]);

    wtik = _mm_loadu_si128((__m128i *)&wtrowi[k0 + 48]);
    macadd(wtik, p0 + 48, sumi4[0]);
    macadd(wtik, p1 + 48, sumi4[1]);
    macadd(wtik, p2 + 48, sumi4[2]);
    macadd(wtik, p3 + 48, sumi4[3]);

    wtik = _mm_loadu_si128((__m128i *)&wtrowi[k0 + 56]);
    macadd(wtik, p0 + 56, sumi4[0]);
    macadd(wtik, p1 + 56, sumi4[1]);
    macadd(wtik, p2 + 56, sumi4[2]);
    macadd(wtik, p3 + 56, sumi4[3]);
  }

  for (; k0 + 8 <= dim; k0 += 8, p0 += 8, p1 += 8, p2 += 8, p3 += 8) {
    wtik = _mm_loadu_si128((__m128i *)&wtrowi[k0]);
    macadd(wtik, p0, sumi4[0]);
    macadd(wtik, p1, sumi4[1]);
    macadd(wtik, p2, sumi4[2]);
    macadd(wtik, p3, sumi4[3]);
  }

  for (; k0 < dim; k0++, p0++, p1++, p2++, p3++) {
    *(usij + 0 * usijstride) += wtrowi[k0] * (*p0);
    *(usij + 1 * usijstride) += wtrowi[k0] * (*p1);
    *(usij + 2 * usijstride) += wtrowi[k0] * (*p2);
    *(usij + 3 * usijstride) += wtrowi[k0] * (*p3);
  }

#ifdef _MSC_VER
  __declspec(align(16)) int sums[4];
#else
  __attribute__((aligned(16))) int sums[4];
#endif

  _mm_store_si128((__m128i *) sums, sumi4[0]);
  *(usij + 0 * usijstride) += sums[0] + sums[1] + sums[2] + sums[3];
  _mm_store_si128((__m128i *) sums, sumi4[1]);
  *(usij + 1 * usijstride) += sums[0] + sums[1] + sums[2] + sums[3];
  _mm_store_si128((__m128i *) sums, sumi4[2]);
  *(usij + 2 * usijstride) += sums[0] + sums[1] + sums[2] + sums[3];
  _mm_store_si128((__m128i *) sums, sumi4[3]);
  *(usij + 3 * usijstride) += sums[0] + sums[1] + sums[2] + sums[3];
}


void dotprod4_sse(const float *row, const float *cols4,
                  const size_t cols4stride, float *usij, const size_t usijstride,
                  const size_t dim) {
  const float *p = row;
  const float *p0 = cols4, *p1 = p0 + cols4stride, *p2 = p1 + cols4stride,
               *p3 = p2 + cols4stride;
  __m128 sum[4];
  __m128 r, c;
  size_t d;

  sum[0] = sum[1] = sum[2] = sum[3] = _mm_set1_ps(0);
  for (d = 0; d + 32 <= dim;
       d += 32, p += 32, p0 += 32, p1 += 32, p2 += 32,
       p3 += 32) { // 8-way loop unrolling
    r = _mm_loadu_ps(p);

    c = _mm_loadu_ps(p0);
    c = _mm_mul_ps(c, r);
    sum[0] = _mm_add_ps(sum[0], c);

    c = _mm_loadu_ps(p1);
    c = _mm_mul_ps(c, r);
    sum[1] = _mm_add_ps(sum[1], c);

    c = _mm_loadu_ps(p2);
    c = _mm_mul_ps(c, r);
    sum[2] = _mm_add_ps(sum[2], c);

    c = _mm_loadu_ps(p3);
    c = _mm_mul_ps(c, r);
    sum[3] = _mm_add_ps(sum[3], c);

    r = _mm_loadu_ps(p + 4);

    c = _mm_loadu_ps(p0 + 4);
    c = _mm_mul_ps(c, r);
    sum[0] = _mm_add_ps(sum[0], c);

    c = _mm_loadu_ps(p1 + 4);
    c = _mm_mul_ps(c, r);
    sum[1] = _mm_add_ps(sum[1], c);

    c = _mm_loadu_ps(p2 + 4);
    c = _mm_mul_ps(c, r);
    sum[2] = _mm_add_ps(sum[2], c);

    c = _mm_loadu_ps(p3 + 4);
    c = _mm_mul_ps(c, r);
    sum[3] = _mm_add_ps(sum[3], c);

    r = _mm_loadu_ps(p + 8);

    c = _mm_loadu_ps(p0 + 8);
    c = _mm_mul_ps(c, r);
    sum[0] = _mm_add_ps(sum[0], c);

    c = _mm_loadu_ps(p1 + 8);
    c = _mm_mul_ps(c, r);
    sum[1] = _mm_add_ps(sum[1], c);

    c = _mm_loadu_ps(p2 + 8);
    c = _mm_mul_ps(c, r);
    sum[2] = _mm_add_ps(sum[2], c);

    c = _mm_loadu_ps(p3 + 8);
    c = _mm_mul_ps(c, r);
    sum[3] = _mm_add_ps(sum[3], c);

    r = _mm_loadu_ps(p + 12);

    c = _mm_loadu_ps(p0 + 12);
    c = _mm_mul_ps(c, r);
    sum[0] = _mm_add_ps(sum[0], c);

    c = _mm_loadu_ps(p1 + 12);
    c = _mm_mul_ps(c, r);
    sum[1] = _mm_add_ps(sum[1], c);

    c = _mm_loadu_ps(p2 + 12);
    c = _mm_mul_ps(c, r);
    sum[2] = _mm_add_ps(sum[2], c);

    c = _mm_loadu_ps(p3 + 12);
    c = _mm_mul_ps(c, r);
    sum[3] = _mm_add_ps(sum[3], c);

    r = _mm_loadu_ps(p + 16);

    c = _mm_loadu_ps(p0 + 16);
    c = _mm_mul_ps(c, r);
    sum[0] = _mm_add_ps(sum[0], c);

    c = _mm_loadu_ps(p1 + 16);
    c = _mm_mul_ps(c, r);
    sum[1] = _mm_add_ps(sum[1], c);

    c = _mm_loadu_ps(p2 + 16);
    c = _mm_mul_ps(c, r);
    sum[2] = _mm_add_ps(sum[2], c);

    c = _mm_loadu_ps(p3 + 16);
    c = _mm_mul_ps(c, r);
    sum[3] = _mm_add_ps(sum[3], c);

    r = _mm_loadu_ps(p + 20);

    c = _mm_loadu_ps(p0 + 20);
    c = _mm_mul_ps(c, r);
    sum[0] = _mm_add_ps(sum[0], c);

    c = _mm_loadu_ps(p1 + 20);
    c = _mm_mul_ps(c, r);
    sum[1] = _mm_add_ps(sum[1], c);

    c = _mm_loadu_ps(p2 + 20);
    c = _mm_mul_ps(c, r);
    sum[2] = _mm_add_ps(sum[2], c);

    c = _mm_loadu_ps(p3 + 20);
    c = _mm_mul_ps(c, r);
    sum[3] = _mm_add_ps(sum[3], c);

    r = _mm_loadu_ps(p + 24);

    c = _mm_loadu_ps(p0 + 24);
    c = _mm_mul_ps(c, r);
    sum[0] = _mm_add_ps(sum[0], c);

    c = _mm_loadu_ps(p1 + 24);
    c = _mm_mul_ps(c, r);
    sum[1] = _mm_add_ps(sum[1], c);

    c = _mm_loadu_ps(p2 + 24);
    c = _mm_mul_ps(c, r);
    sum[2] = _mm_add_ps(sum[2], c);

    c = _mm_loadu_ps(p3 + 24);
    c = _mm_mul_ps(c, r);
    sum[3] = _mm_add_ps(sum[3], c);

    r = _mm_loadu_ps(p + 28);

    c = _mm_loadu_ps(p0 + 28);
    c = _mm_mul_ps(c, r);
    sum[0] = _mm_add_ps(sum[0], c);

    c = _mm_loadu_ps(p1 + 28);
    c = _mm_mul_ps(c, r);
    sum[1] = _mm_add_ps(sum[1], c);

    c = _mm_loadu_ps(p2 + 28);
    c = _mm_mul_ps(c, r);
    sum[2] = _mm_add_ps(sum[2], c);

    c = _mm_loadu_ps(p3 + 28);
    c = _mm_mul_ps(c, r);
    sum[3] = _mm_add_ps(sum[3], c);
  }
  for (; d < dim; ++d, ++p, ++p0, ++p1, ++p2, ++p3) {
    r = _mm_load_ss(p);

    c = _mm_load_ss(p0);
    c = _mm_mul_ss(c, r);
    sum[0] = _mm_add_ss(sum[0], c);

    c = _mm_load_ss(p1);
    c = _mm_mul_ss(c, r);
    sum[1] = _mm_add_ss(sum[1], c);

    c = _mm_load_ss(p2);
    c = _mm_mul_ss(c, r);
    sum[2] = _mm_add_ss(sum[2], c);

    c = _mm_load_ss(p3);
    c = _mm_mul_ss(c, r);
    sum[3] = _mm_add_ss(sum[3], c);
  }


  // *usij *= scale;
  // *usij += sum[0].m128_f32[0] + sum[0].m128_f32[1] + sum[0].m128_f32[2] + sum[0].m128_f32[3];
  r = _mm_load_ss(usij);
  r = _mm_add_ps(r, sum[0]);
  r = _mm_hadd_ps(r, r);
  r = _mm_hadd_ps(r, r);
  _mm_store_ss(usij, r);

  usij += usijstride;
  r = _mm_load_ss(usij);
  r = _mm_add_ps(r, sum[1]);
  r = _mm_hadd_ps(r, r);
  r = _mm_hadd_ps(r, r);
  _mm_store_ss(usij, r);

  usij += usijstride;
  r = _mm_load_ss(usij);
  r = _mm_add_ps(r, sum[2]);
  r = _mm_hadd_ps(r, r);
  r = _mm_hadd_ps(r, r);
  _mm_store_ss(usij, r);

  usij += usijstride;
  r = _mm_load_ss(usij);
  r = _mm_add_ps(r, sum[3]);
  r = _mm_hadd_ps(r, r);
  r = _mm_hadd_ps(r, r);
  _mm_store_ss(usij, r);
}

void dotprod_sse(const float *row, const float *col, float *usij,
                 const size_t dim) {
  __m128 sum;
  __m128 r, c;
  size_t d;

  sum = _mm_set1_ps(0);
  for (d = 0; d + 32 <= dim;
       d += 32, row += 32, col += 32) { // 8-way loop unrolling
    r = _mm_loadu_ps(row);
    c = _mm_loadu_ps(col);
    c = _mm_mul_ps(c, r);
    sum = _mm_add_ps(sum, c);

    r = _mm_loadu_ps(row + 4);
    c = _mm_loadu_ps(col + 4);
    c = _mm_mul_ps(c, r);
    sum = _mm_add_ps(sum, c);

    r = _mm_loadu_ps(row + 8);
    c = _mm_loadu_ps(col + 8);
    c = _mm_mul_ps(c, r);
    sum = _mm_add_ps(sum, c);

    r = _mm_loadu_ps(row + 12);
    c = _mm_loadu_ps(col + 12);
    c = _mm_mul_ps(c, r);
    sum = _mm_add_ps(sum, c);

    r = _mm_loadu_ps(row + 16);
    c = _mm_loadu_ps(col + 16);
    c = _mm_mul_ps(c, r);
    sum = _mm_add_ps(sum, c);

    r = _mm_loadu_ps(row + 20);
    c = _mm_loadu_ps(col + 20);
    c = _mm_mul_ps(c, r);
    sum = _mm_add_ps(sum, c);

    r = _mm_loadu_ps(row + 24);
    c = _mm_loadu_ps(col + 24);
    c = _mm_mul_ps(c, r);
    sum = _mm_add_ps(sum, c);

    r = _mm_loadu_ps(row + 28);
    c = _mm_loadu_ps(col + 28);
    c = _mm_mul_ps(c, r);
    sum = _mm_add_ps(sum, c);
  }
  for (; d<dim; ++d, ++row, ++col) {
    r = _mm_load_ss(row);
    c = _mm_load_ss(col);
    c = _mm_mul_ss(c, r);
    sum = _mm_add_ss(sum, c);
  }

  r = _mm_load_ss(usij);
  r = _mm_add_ps(r, sum);
  r = _mm_hadd_ps(r, r);
  r = _mm_hadd_ps(r, r);
  _mm_store_ss(usij, r);
}

void scaleadd_sse(const float scale, float *usij, const float *col,
                  const size_t dim) {
  size_t d;
  __m128 s = _mm_set1_ps(scale), m, n;

  for (d = 0; d + 32 <= dim;
       d += 32, col += 32, usij += 32) { // 8-way loop unrolling
    m = _mm_loadu_ps(usij);
    m = _mm_mul_ps(m, s);
    n = _mm_loadu_ps(col);
    m = _mm_add_ps(m, n);
    _mm_storeu_ps(usij, m);

    m = _mm_loadu_ps(usij + 4);
    m = _mm_mul_ps(m, s);
    n = _mm_loadu_ps(col + 4);
    m = _mm_add_ps(m, n);
    _mm_storeu_ps(usij + 4, m);

    m = _mm_loadu_ps(usij + 8);
    m = _mm_mul_ps(m, s);
    n = _mm_loadu_ps(col + 8);
    m = _mm_add_ps(m, n);
    _mm_storeu_ps(usij + 8, m);

    m = _mm_loadu_ps(usij + 12);
    m = _mm_mul_ps(m, s);
    n = _mm_loadu_ps(col + 12);
    m = _mm_add_ps(m, n);
    _mm_storeu_ps(usij + 12, m);

    m = _mm_loadu_ps(usij + 16);
    m = _mm_mul_ps(m, s);
    n = _mm_loadu_ps(col + 16);
    m = _mm_add_ps(m, n);
    _mm_storeu_ps(usij + 16, m);

    m = _mm_loadu_ps(usij + 20);
    m = _mm_mul_ps(m, s);
    n = _mm_loadu_ps(col + 20);
    m = _mm_add_ps(m, n);
    _mm_storeu_ps(usij + 20, m);

    m = _mm_loadu_ps(usij + 24);
    m = _mm_mul_ps(m, s);
    n = _mm_loadu_ps(col + 24);
    m = _mm_add_ps(m, n);
    _mm_storeu_ps(usij + 24, m);

    m = _mm_loadu_ps(usij + 28);
    m = _mm_mul_ps(m, s);
    n = _mm_loadu_ps(col + 28);
    m = _mm_add_ps(m, n);
    _mm_storeu_ps(usij + 28, m);
  }
  for (; d<dim; ++d, ++col, ++usij) {
    m = _mm_load_ss(usij);
    m = _mm_mul_ss(m, s);
    n = _mm_load_ss(col);
    m = _mm_add_ss(m, n);
    _mm_store_ss(usij, m);
  }
}

void scaleadd_sse(float *usij, const float *col, const float scale,
	const size_t dim) {
	size_t d;
	__m128 s = _mm_set1_ps(scale), m, n;

	for (d = 0; d + 32 <= dim;
		d += 32, col += 32, usij += 32) { // 8-way loop unrolling
		m = _mm_loadu_ps(usij);		
		n = _mm_loadu_ps(col);
		n = _mm_mul_ps(n, s);
		m = _mm_add_ps(m, n);
		_mm_storeu_ps(usij, m);

		m = _mm_loadu_ps(usij + 4);		
		n = _mm_loadu_ps(col + 4);
		n = _mm_mul_ps(n, s);
		m = _mm_add_ps(m, n);
		_mm_storeu_ps(usij + 4, m);

		m = _mm_loadu_ps(usij + 8);
		n = _mm_loadu_ps(col + 8);
		n = _mm_mul_ps(n, s);
		m = _mm_add_ps(m, n);
		_mm_storeu_ps(usij + 8, m);

		m = _mm_loadu_ps(usij + 12);		
		n = _mm_loadu_ps(col + 12);
		n = _mm_mul_ps(n, s);
		m = _mm_add_ps(m, n);
		_mm_storeu_ps(usij + 12, m);

		m = _mm_loadu_ps(usij + 16);
		n = _mm_loadu_ps(col + 16);
		n = _mm_mul_ps(n, s);
		m = _mm_add_ps(m, n);
		_mm_storeu_ps(usij + 16, m);

		m = _mm_loadu_ps(usij + 20);
		n = _mm_loadu_ps(col + 20);
		n = _mm_mul_ps(n, s);
		m = _mm_add_ps(m, n);
		_mm_storeu_ps(usij + 20, m);

		m = _mm_loadu_ps(usij + 24);
		n = _mm_loadu_ps(col + 24);
		n = _mm_mul_ps(n, s);
		m = _mm_add_ps(m, n);
		_mm_storeu_ps(usij + 24, m);

		m = _mm_loadu_ps(usij + 28);
		n = _mm_loadu_ps(col + 28);
		n = _mm_mul_ps(n, s);
		m = _mm_add_ps(m, n);
		_mm_storeu_ps(usij + 28, m);
	}
	for (; d < dim; ++d, ++col, ++usij) {
		m = _mm_load_ss(usij);
		n = _mm_load_ss(col);
		n = _mm_mul_ps(n, s);
		m = _mm_add_ss(m, n);
		_mm_store_ss(usij, m);
	}
}

void scaleadd_sse(const float scale, float *usij, const int *col,
                  const size_t dim) {
  size_t d;
  __m128 m, n, p = _mm_set1_ps(scale);
  __m128i i;

  for (d = 0; d + 32 <= dim;
       d += 32, col += 32, usij += 32) { // 8-way loop unrolling
    i = _mm_loadu_si128((__m128i *)col);
    n = _mm_cvtepi32_ps(i);
    n = _mm_mul_ps(n, p);
    m = _mm_loadu_ps(usij);
    m = _mm_add_ps(m, n);
    _mm_storeu_ps(usij, m);

    i = _mm_loadu_si128((__m128i *)(col + 4));
    n = _mm_cvtepi32_ps(i);
    n = _mm_mul_ps(n, p);
    m = _mm_loadu_ps(usij + 4);
    m = _mm_add_ps(m, n);
    _mm_storeu_ps(usij + 4, m);

    i = _mm_loadu_si128((__m128i *)(col + 8));
    n = _mm_cvtepi32_ps(i);
    n = _mm_mul_ps(n, p);
    m = _mm_loadu_ps(usij + 8);
    m = _mm_add_ps(m, n);
    _mm_storeu_ps(usij + 8, m);

    i = _mm_loadu_si128((__m128i *)(col + 12));
    n = _mm_cvtepi32_ps(i);
    n = _mm_mul_ps(n, p);
    m = _mm_loadu_ps(usij + 12);
    m = _mm_add_ps(m, n);
    _mm_storeu_ps(usij + 12, m);

    i = _mm_loadu_si128((__m128i *)(col + 16));
    n = _mm_cvtepi32_ps(i);
    n = _mm_mul_ps(n, p);
    m = _mm_loadu_ps(usij + 16);
    m = _mm_add_ps(m, n);
    _mm_storeu_ps(usij + 16, m);

    i = _mm_loadu_si128((__m128i *)(col + 20));
    n = _mm_cvtepi32_ps(i);
    n = _mm_mul_ps(n, p);
    m = _mm_loadu_ps(usij + 20);
    m = _mm_add_ps(m, n);
    _mm_storeu_ps(usij + 20, m);

    i = _mm_loadu_si128((__m128i *)(col + 24));
    n = _mm_cvtepi32_ps(i);
    n = _mm_mul_ps(n, p);
    m = _mm_loadu_ps(usij + 24);
    m = _mm_add_ps(m, n);
    _mm_storeu_ps(usij + 24, m);

    i = _mm_loadu_si128((__m128i *)(col + 28));
    n = _mm_cvtepi32_ps(i);
    n = _mm_mul_ps(n, p);
    m = _mm_loadu_ps(usij + 28);
    m = _mm_add_ps(m, n);
    _mm_storeu_ps(usij + 28, m);
  }

  for (; d < dim; ++d, ++col, ++usij) {
    *usij += *col * scale;
  }
}

void multi_sse(float *usij, const float *col, const size_t dim) {
  size_t d;
  __m128 m, n;
  for (d = 0; d + 32 <= dim;
       d += 32, col += 32, usij += 32) { // 8-way loop unrolling
    m = _mm_loadu_ps(usij);
    n = _mm_loadu_ps(col);
    m = _mm_mul_ps(m, n);
    _mm_storeu_ps(usij, m);

    m = _mm_loadu_ps(usij + 4);
    n = _mm_loadu_ps(col + 4);
    m = _mm_mul_ps(m, n);
    _mm_storeu_ps(usij + 4, m);

    m = _mm_loadu_ps(usij + 8);
    n = _mm_loadu_ps(col + 8);
    m = _mm_mul_ps(m, n);
    _mm_storeu_ps(usij + 8, m);

    m = _mm_loadu_ps(usij + 12);
    n = _mm_loadu_ps(col + 12);
    m = _mm_mul_ps(m, n);
    _mm_storeu_ps(usij + 12, m);

    m = _mm_loadu_ps(usij + 16);
    n = _mm_loadu_ps(col + 16);
    m = _mm_mul_ps(m, n);
    _mm_storeu_ps(usij + 16, m);

    m = _mm_loadu_ps(usij + 20);
    n = _mm_loadu_ps(col + 20);
    m = _mm_mul_ps(m, n);
    _mm_storeu_ps(usij + 20, m);

    m = _mm_loadu_ps(usij + 24);
    n = _mm_loadu_ps(col + 24);
    m = _mm_mul_ps(m, n);
    _mm_storeu_ps(usij + 24, m);

    m = _mm_loadu_ps(usij + 28);
    n = _mm_loadu_ps(col + 28);
    m = _mm_mul_ps(m, n);
    _mm_storeu_ps(usij + 28, m);
  }

  for (; d<dim; ++d, ++col, ++usij) {
    m = _mm_load_ss(usij);
    n = _mm_load_ss(col);
    m = _mm_mul_ps(m, n);
    _mm_store_ss(usij, m);
  }
}

void add_sse(float *usij, const float *col, const size_t dim) {
  size_t d;
  __m128 m, n;

  for (d = 0; d + 32 <= dim;
       d += 32, col += 32, usij += 32) { // 8-way loop unrolling
    m = _mm_loadu_ps(usij);
    n = _mm_loadu_ps(col);
    m = _mm_add_ps(m, n);
    _mm_storeu_ps(usij, m);

    m = _mm_loadu_ps(usij + 4);
    n = _mm_loadu_ps(col + 4);
    m = _mm_add_ps(m, n);
    _mm_storeu_ps(usij + 4, m);

    m = _mm_loadu_ps(usij + 8);
    n = _mm_loadu_ps(col + 8);
    m = _mm_add_ps(m, n);
    _mm_storeu_ps(usij + 8, m);

    m = _mm_loadu_ps(usij + 12);
    n = _mm_loadu_ps(col + 12);
    m = _mm_add_ps(m, n);
    _mm_storeu_ps(usij + 12, m);

    m = _mm_loadu_ps(usij + 16);
    n = _mm_loadu_ps(col + 16);
    m = _mm_add_ps(m, n);
    _mm_storeu_ps(usij + 16, m);

    m = _mm_loadu_ps(usij + 20);
    n = _mm_loadu_ps(col + 20);
    m = _mm_add_ps(m, n);
    _mm_storeu_ps(usij + 20, m);

    m = _mm_loadu_ps(usij + 24);
    n = _mm_loadu_ps(col + 24);
    m = _mm_add_ps(m, n);
    _mm_storeu_ps(usij + 24, m);

    m = _mm_loadu_ps(usij + 28);
    n = _mm_loadu_ps(col + 28);
    m = _mm_add_ps(m, n);
    _mm_storeu_ps(usij + 28, m);
  }

  for (; d<dim; ++d, ++col, ++usij) {
    m = _mm_load_ss(usij);
    n = _mm_load_ss(col);
    m = _mm_add_ss(m, n);
    _mm_store_ss(usij, m);
  }
}

void minus_div_sse(float *p, const float *pmhead, const float *pvhead,
                   size_t len) {
  __m128 x, y;
  const float *pm = pmhead;
  const float *pv = pvhead;

  size_t d;
  // always use loadu_ps instead of load_ps because p may not be properly aligned in xnnRuntimeColumnMatrixView::RowView. There is no perf. penalty using loadu_ps when p is aligned.
  for (d = 0; d + 4 <= len; d += 4, p += 4, pm += 4, pv += 4) {
    x = _mm_loadu_ps(p);
    y = _mm_loadu_ps(pm);
    x = _mm_sub_ps(x, y);
    y = _mm_loadu_ps(pv);
    x = _mm_div_ps(x, y);
    _mm_storeu_ps(p, x);
  }
  for (; d < len; ++d, ++p, ++pm, ++pv) {
    x = _mm_load_ss(p);
    y = _mm_load_ss(pm);
    x = _mm_sub_ss(x, y);
    y = _mm_load_ss(pv);
    x = _mm_div_ss(x, y);
    _mm_store_ss(p, x);
  }
}

void minus_sse(float *p, const float *pv, size_t len) {
  __m128 x, y;
  size_t d;
  for (d = 0; d + 4 <= len; d += 4, p += 4, pv += 4) {
    x = _mm_loadu_ps(p);
    y = _mm_loadu_ps(pv);
    x = _mm_sub_ps(x, y);
    _mm_storeu_ps(p, x);
  }

  for (; d < len; ++d, ++p, ++pv) {
    x = _mm_load_ss(p);
    y = _mm_load_ss(pv);
    x = _mm_sub_ss(x, y);
    _mm_store_ss(p, x);
  }
}

void plus_sse(float *u, const float *v, size_t len) {
  __m128 w, x;
  size_t d;

  for (d = 0; d + 32 <= len; d += 32, v += 32, u += 32) { // 8-way loop unrolling
    w = _mm_loadu_ps(u);
    x = _mm_loadu_ps(v);
    w = _mm_add_ps(w, x);
    _mm_storeu_ps(u, w);

    w = _mm_loadu_ps(u + 4);
    x = _mm_loadu_ps(v + 4);
    w = _mm_add_ps(w, x);
    _mm_storeu_ps(u + 4, w);

    w = _mm_loadu_ps(u + 8);
    x = _mm_loadu_ps(v + 8);
    w = _mm_add_ps(w, x);
    _mm_storeu_ps(u + 8, w);

    w = _mm_loadu_ps(u + 12);
    x = _mm_loadu_ps(v + 12);
    w = _mm_add_ps(w, x);
    _mm_storeu_ps(u + 12, w);

    w = _mm_loadu_ps(u + 16);
    x = _mm_loadu_ps(v + 16);
    w = _mm_add_ps(w, x);
    _mm_storeu_ps(u + 16, w);

    w = _mm_loadu_ps(u + 20);
    x = _mm_loadu_ps(v + 20);
    w = _mm_add_ps(w, x);
    _mm_storeu_ps(u + 20, w);

    w = _mm_loadu_ps(u + 24);
    x = _mm_loadu_ps(v + 24);
    w = _mm_add_ps(w, x);
    _mm_storeu_ps(u + 24, w);

    w = _mm_loadu_ps(u + 28);
    x = _mm_loadu_ps(v + 28);
    w = _mm_add_ps(w, x);
    _mm_storeu_ps(u + 28, w);
  }

  for (; d < len; ++d, ++v, ++u) {
    w = _mm_load_ss(u);
    x = _mm_load_ss(v);
    w = _mm_add_ss(w, x);
    _mm_store_ss(u, w);
  }
}

void div_norm_sse(float *p, float divider, size_t len) {
  __m128 x, y;
  size_t d;
  y = _mm_set1_ps(divider);
  for (d = 0; d + 4 <= len; d += 4, p += 4) {
    x = _mm_loadu_ps(
          p);// always use loadu_ps instead of load_ps because p may not be properly aligned in xnnRuntimeColumnMatrixView::RowView. There is no perf. penalty using loadu_ps when p is aligned.
    x = _mm_div_ps(x, y);
    _mm_storeu_ps(p, x);
  }

  for (; d < len; ++d, ++p) {
    x = _mm_load_ss(p);
    x = _mm_div_ss(x, y);
    _mm_store_ss(p, x);
  }
}

void plus_norm(float *u, const float *v, size_t len) {
  __m128 w, x;
  size_t d;

#if 1
  // 8-way loop unrolling
  for (d = 0; d + 32 <= len; d += 32, v += 32, u += 32) {
    w = _mm_loadu_ps(u);
    x = _mm_loadu_ps(v);
    w = _mm_add_ps(w, x);
    _mm_storeu_ps(u, w);

    w = _mm_loadu_ps(u + 4);
    x = _mm_loadu_ps(v + 4);
    w = _mm_add_ps(w, x);
    _mm_storeu_ps(u + 4, w);

    w = _mm_loadu_ps(u + 8);
    x = _mm_loadu_ps(v + 8);
    w = _mm_add_ps(w, x);
    _mm_storeu_ps(u + 8, w);

    w = _mm_loadu_ps(u + 12);
    x = _mm_loadu_ps(v + 12);
    w = _mm_add_ps(w, x);
    _mm_storeu_ps(u + 12, w);

    w = _mm_loadu_ps(u + 16);
    x = _mm_loadu_ps(v + 16);
    w = _mm_add_ps(w, x);
    _mm_storeu_ps(u + 16, w);

    w = _mm_loadu_ps(u + 20);
    x = _mm_loadu_ps(v + 20);
    w = _mm_add_ps(w, x);
    _mm_storeu_ps(u + 20, w);

    w = _mm_loadu_ps(u + 24);
    x = _mm_loadu_ps(v + 24);
    w = _mm_add_ps(w, x);
    _mm_storeu_ps(u + 24, w);

    w = _mm_loadu_ps(u + 28);
    x = _mm_loadu_ps(v + 28);
    w = _mm_add_ps(w, x);
    _mm_storeu_ps(u + 28, w);
  }
#else
  for (d = 0; d + 4 <= vdim; d += 4, pW += 4, pX += 4) {
    w = _mm_loadu_ps(pW);
    x = _mm_loadu_ps(pX);
    w = _mm_mul_ps(w, x);
    sum = _mm_add_ps(sum, w);
  }
#endif
  for (; d < len; ++d, ++v, ++u) {
    w = _mm_load_ss(u);
    x = _mm_load_ss(v);
    w = _mm_add_ss(w, x);
    _mm_store_ss(u, w);
  }
}

void scale_plus_prod_sse(float *p, const float *pX, const float *pY, float scale, size_t len) {
	__m128 s = _mm_set1_ps(scale);
	__m128 x, y;
	size_t d;
	for (d = 0; d + 32 <= len;
		d += 32, p += 32, pX += 32, pY += 32) { // 8-way loop unrolling
		x = _mm_loadu_ps(pX);
		y = _mm_loadu_ps(pY);
		x = _mm_mul_ps(x, y);
		x = _mm_mul_ps(x, s);
		y = _mm_loadu_ps(p);		
		y = _mm_add_ps(y, x);
		_mm_storeu_ps(p, y);

		x = _mm_loadu_ps(pX + 4);
		y = _mm_loadu_ps(pY + 4);
		x = _mm_mul_ps(x, y);
		x = _mm_mul_ps(x, s);
		y = _mm_loadu_ps(p + 4);		
		y = _mm_add_ps(y, x);
		_mm_storeu_ps(p + 4, y);

		x = _mm_loadu_ps(pX + 8);
		y = _mm_loadu_ps(pY + 8);
		x = _mm_mul_ps(x, y);
		x = _mm_mul_ps(x, s);
		y = _mm_loadu_ps(p + 8);		
		y = _mm_add_ps(y, x);
		_mm_storeu_ps(p + 8, y);

		x = _mm_loadu_ps(pX + 12);
		y = _mm_loadu_ps(pY + 12);
		x = _mm_mul_ps(x, y);
		x = _mm_mul_ps(x, s);
		y = _mm_loadu_ps(p + 12);		
		y = _mm_add_ps(y, x);
		_mm_storeu_ps(p + 12, y);

		x = _mm_loadu_ps(pX + 16);
		y = _mm_loadu_ps(pY + 16);
		x = _mm_mul_ps(x, y);
		x = _mm_mul_ps(x, s);
		y = _mm_loadu_ps(p + 16);		
		y = _mm_add_ps(y, x);
		_mm_storeu_ps(p + 16, y);

		x = _mm_loadu_ps(pX + 20);
		y = _mm_loadu_ps(pY + 20);
		x = _mm_mul_ps(x, y);
		x = _mm_mul_ps(x, s);
		y = _mm_loadu_ps(p + 20);		
		y = _mm_add_ps(y, x);
		_mm_storeu_ps(p + 20, y);

		x = _mm_loadu_ps(pX + 24);
		y = _mm_loadu_ps(pY + 24);
		x = _mm_mul_ps(x, y);
		x = _mm_mul_ps(x, s);
		y = _mm_loadu_ps(p + 24);		
		y = _mm_add_ps(y, x);
		_mm_storeu_ps(p + 24, y);

		x = _mm_loadu_ps(pX + 28);
		y = _mm_loadu_ps(pY + 28);
		x = _mm_mul_ps(x, y);
		x = _mm_mul_ps(x, s);
		y = _mm_loadu_ps(p + 28);
		y = _mm_add_ps(y, x);
		_mm_storeu_ps(p + 28, y);
	}
	for (; d < len; ++d, ++p, ++pX, ++pY) {
		x = _mm_load_ss(pX);
		y = _mm_load_ss(pY);
		x = _mm_mul_ss(x, y);
		x = _mm_mul_ss(x, s);
		y = _mm_load_ss(p);		
		y = _mm_add_ss(y, x);
		_mm_store_ss(p, y);
	}
}

void scale_plus_prod_sse(float *p, const float *pX, const float *pY,
                         size_t len, float scale) {
  __m128 s = _mm_set1_ps(scale);
  __m128 x, y;
  size_t d;
  for (d = 0; d + 32 <= len;
       d += 32, p += 32, pX += 32, pY += 32) { // 8-way loop unrolling
    x = _mm_loadu_ps(pX);
    y = _mm_loadu_ps(pY);
    x = _mm_mul_ps(x, y);
    y = _mm_loadu_ps(p);
    y = _mm_mul_ps(y, s);
    y = _mm_add_ps(y, x);
    _mm_storeu_ps(p, y);

    x = _mm_loadu_ps(pX + 4);
    y = _mm_loadu_ps(pY + 4);
    x = _mm_mul_ps(x, y);
    y = _mm_loadu_ps(p + 4);
    y = _mm_mul_ps(y, s);
    y = _mm_add_ps(y, x);
    _mm_storeu_ps(p + 4, y);

    x = _mm_loadu_ps(pX + 8);
    y = _mm_loadu_ps(pY + 8);
    x = _mm_mul_ps(x, y);
    y = _mm_loadu_ps(p + 8);
    y = _mm_mul_ps(y, s);
    y = _mm_add_ps(y, x);
    _mm_storeu_ps(p + 8, y);

    x = _mm_loadu_ps(pX + 12);
    y = _mm_loadu_ps(pY + 12);
    x = _mm_mul_ps(x, y);
    y = _mm_loadu_ps(p + 12);
    y = _mm_mul_ps(y, s);
    y = _mm_add_ps(y, x);
    _mm_storeu_ps(p + 12, y);

    x = _mm_loadu_ps(pX + 16);
    y = _mm_loadu_ps(pY + 16);
    x = _mm_mul_ps(x, y);
    y = _mm_loadu_ps(p + 16);
    y = _mm_mul_ps(y, s);
    y = _mm_add_ps(y, x);
    _mm_storeu_ps(p + 16, y);

    x = _mm_loadu_ps(pX + 20);
    y = _mm_loadu_ps(pY + 20);
    x = _mm_mul_ps(x, y);
    y = _mm_loadu_ps(p + 20);
    y = _mm_mul_ps(y, s);
    y = _mm_add_ps(y, x);
    _mm_storeu_ps(p + 20, y);

    x = _mm_loadu_ps(pX + 24);
    y = _mm_loadu_ps(pY + 24);
    x = _mm_mul_ps(x, y);
    y = _mm_loadu_ps(p + 24);
    y = _mm_mul_ps(y, s);
    y = _mm_add_ps(y, x);
    _mm_storeu_ps(p + 24, y);

    x = _mm_loadu_ps(pX + 28);
    y = _mm_loadu_ps(pY + 28);
    x = _mm_mul_ps(x, y);
    y = _mm_loadu_ps(p + 28);
    y = _mm_mul_ps(y, s);
    y = _mm_add_ps(y, x);
    _mm_storeu_ps(p + 28, y);
  }
  for (; d < len; ++d, ++p, ++pX, ++pY) {
    x = _mm_load_ss(pX);
    y = _mm_load_ss(pY);
    x = _mm_mul_ss(x, y);
    y = _mm_load_ss(p);
    y = _mm_mul_ss(y, s);
    y = _mm_add_ss(y, x);
    _mm_store_ss(p, y);
  }
}

void sigmod_sse(float *pCol, size_t len) {
  __m128 x, zero = _mm_set1_ps(0), one = _mm_set1_ps(1.0f);
  __m128 explimit = _mm_set1_ps(88.722008f);
#ifdef _MSC_VER
  __declspec(align(16)) float expbuffer[4];
#else
  __attribute__((aligned(16))) float expbuffer[4];
#endif
  size_t row;
  for (row = 0; row + 4 <= len; row += 4, pCol += 4) {
    x = _mm_loadu_ps(pCol);
    x = _mm_sub_ps(zero, x);
    x = _mm_min_ps(x, explimit);
    _mm_store_ps(expbuffer, x);
    expbuffer[0] = expf(expbuffer[0]);
    expbuffer[1] = expf(expbuffer[1]);
    expbuffer[2] = expf(expbuffer[2]);
    expbuffer[3] = expf(expbuffer[3]);
    x = _mm_load_ps(expbuffer);
    x = _mm_add_ps(x, one);
    x = _mm_div_ps(one, x);
    _mm_storeu_ps(pCol, x);
  }
  for (; row < len; ++row, ++pCol) {
    x = _mm_load_ss(pCol);
    x = _mm_sub_ss(zero, x);
    x = _mm_min_ss(x, explimit);
    _mm_store_ss(expbuffer, x);
    expbuffer[0] = expf(expbuffer[0]);
    x = _mm_load_ss(expbuffer);
    x = _mm_add_ss(x, one);
    x = _mm_div_ss(one, x);
    _mm_store_ss(pCol, x);
  }
}


void sigmod_sse(float *pSrcCol, float *pDstCol, size_t len) {
  __m128 x, zero = _mm_set1_ps(0), one = _mm_set1_ps(1.0f);
  __m128 explimit = _mm_set1_ps(88.722008f);
#ifdef _MSC_VER
  __declspec(align(16)) float expbuffer[4];
#else
  __attribute__((aligned(16))) float expbuffer[4];
#endif
  size_t row;
  for (row = 0; row + 4 <= len; row += 4, pSrcCol += 4, pDstCol+=4) {
    x = _mm_loadu_ps(pSrcCol);
    x = _mm_sub_ps(zero, x);
    x = _mm_min_ps(x, explimit);
    _mm_store_ps(expbuffer, x);
    expbuffer[0] = expf(expbuffer[0]);
    expbuffer[1] = expf(expbuffer[1]);
    expbuffer[2] = expf(expbuffer[2]);
    expbuffer[3] = expf(expbuffer[3]);
    x = _mm_load_ps(expbuffer);
    x = _mm_add_ps(x, one);
    x = _mm_div_ps(one, x);
    _mm_storeu_ps(pDstCol, x);
  }
  for (; row < len; ++row, ++pSrcCol, ++pDstCol) {
    x = _mm_load_ss(pSrcCol);
    x = _mm_sub_ss(zero, x);
    x = _mm_min_ss(x, explimit);
    _mm_store_ss(expbuffer, x);
    expbuffer[0] = expf(expbuffer[0]);
    x = _mm_load_ss(expbuffer);
    x = _mm_add_ss(x, one);
    x = _mm_div_ss(one, x);
    _mm_store_ss(pDstCol, x);
  }
}

void relu_sse(float *pCol, size_t len) {
  __m128 x, zero = _mm_set1_ps(0);
  size_t row;
  for (row = 0; row + 4 <= len; row += 4, pCol += 4) {
    x = _mm_loadu_ps(pCol);
    x = _mm_max_ps(x, zero);
    _mm_storeu_ps(pCol, x);
  }

  for (; row < len; ++row, ++pCol) {
    x = _mm_load_ss(pCol);
    x = _mm_max_ss(x, zero);
    _mm_store_ss(pCol, x);
  }
}

void tanh_sse(float *pCol, size_t len) {
  __m128 x, zero = _mm_set1_ps(0), one = _mm_set1_ps(1.0f);
  __m128 explimit = _mm_set1_ps(88.722008f);
#ifdef _MSC_VER
  __declspec(align(16)) float expbuffer[4];
#else
  __attribute__((aligned(16))) float expbuffer[4];
#endif

  size_t row;
  for (row = 0; row + 4 <= len; row += 4, pCol += 4) {
    x = _mm_loadu_ps(pCol);     // x
    x = _mm_add_ps(x, x);     // 2x
    x = _mm_sub_ps(zero, x);    // -2x
    x = _mm_min_ps(x, explimit);   // exp(-2x) considering explimit
    _mm_store_ps(expbuffer, x);
    expbuffer[0] = expf(expbuffer[0]);
    expbuffer[1] = expf(expbuffer[1]);
    expbuffer[2] = expf(expbuffer[2]);
    expbuffer[3] = expf(expbuffer[3]);
    x = _mm_load_ps(expbuffer);
    x = _mm_add_ps(x, one);     // exp(-2x)+1
    x = _mm_div_ps(one, x);     // 1 / [exp(-2x)+1]
    x = _mm_add_ps(x, x);     // 2 / [exp(-2x)+1]
    x = _mm_sub_ps(x, one);     // 2 / [exp(-2x)+1] - 1
    _mm_storeu_ps(pCol, x);
  }
  for (; row < len; ++row, ++pCol) {
    x = _mm_load_ss(pCol);
    x = _mm_add_ss(x, x);
    x = _mm_sub_ss(zero, x);
    x = _mm_min_ss(x, explimit);
    _mm_store_ss(expbuffer, x);
    expbuffer[0] = expf(expbuffer[0]);
    x = _mm_load_ss(expbuffer);
    x = _mm_add_ss(x, one);
    x = _mm_div_ss(one, x);
    x = _mm_add_ss(x, x);
    x = _mm_sub_ss(x, one);
    _mm_store_ss(pCol, x);
  }
}

void tanh_sse(float *pSrcCol, float *pDstCol, size_t len) {
  __m128 x, zero = _mm_set1_ps(0), one = _mm_set1_ps(1.0f);
  __m128 explimit = _mm_set1_ps(88.722008f);
#ifdef _MSC_VER
  __declspec(align(16)) float expbuffer[4];
#else
  __attribute__((aligned(16))) float expbuffer[4];
#endif

  size_t row;
  for (row = 0; row + 4 <= len; row += 4, pSrcCol += 4, pDstCol+=4) {
    x = _mm_loadu_ps(pSrcCol);     // x
    x = _mm_add_ps(x, x);     // 2x
    x = _mm_sub_ps(zero, x);    // -2x
    x = _mm_min_ps(x, explimit);   // exp(-2x) considering explimit
    _mm_store_ps(expbuffer, x);
    expbuffer[0] = expf(expbuffer[0]);
    expbuffer[1] = expf(expbuffer[1]);
    expbuffer[2] = expf(expbuffer[2]);
    expbuffer[3] = expf(expbuffer[3]);
    x = _mm_load_ps(expbuffer);
    x = _mm_add_ps(x, one);     // exp(-2x)+1
    x = _mm_div_ps(one, x);     // 1 / [exp(-2x)+1]
    x = _mm_add_ps(x, x);     // 2 / [exp(-2x)+1]
    x = _mm_sub_ps(x, one);     // 2 / [exp(-2x)+1] - 1
    _mm_storeu_ps(pDstCol, x);
  }
  for (; row < len; ++row, ++pSrcCol, ++pDstCol) {
    x = _mm_load_ss(pSrcCol);
    x = _mm_add_ss(x, x);
    x = _mm_sub_ss(zero, x);
    x = _mm_min_ss(x, explimit);
    _mm_store_ss(expbuffer, x);
    expbuffer[0] = expf(expbuffer[0]);
    x = _mm_load_ss(expbuffer);
    x = _mm_add_ss(x, one);
    x = _mm_div_ss(one, x);
    x = _mm_add_ss(x, x);
    x = _mm_sub_ss(x, one);
    _mm_store_ss(pDstCol, x);
  }
}

void group_pnorm_sse(float *result, const float *p, size_t len,
                     size_t groupsize) {
  for (size_t g = 0; g < len; ++g) {
    __m128 sum = _mm_set1_ps(0), x;
    size_t d;

    for (d = 0; d + 4 <= groupsize; d += 4, p += 4) {
      x = _mm_loadu_ps(p);
      x = _mm_mul_ps(x, x);
      sum = _mm_add_ps(sum, x);
    }
    for (; d < groupsize; ++d, ++p) {
      x = _mm_load_ss(p);
      x = _mm_mul_ss(x, x);
      sum = _mm_add_ss(sum, x);
    }

    // Col(col)[g] = std::sqrt(sum.m128_f32[0] + sum.m128_f32[1] + sum.m128_f32[2] + sum.m128_f32[3]);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    _mm_store_ss(result + g, sum);
    result[g] = std::sqrt(result[g]);
  }
}

void l2norm_sse(float *result, const float *p, size_t len, float floor,
                float alpha) {
  __m128 sum = _mm_set1_ps(0), x;
  size_t d;

  for (d = 0; d + 4 <= len; d += 4, p += 4) {
    x = _mm_loadu_ps(p);
    x = _mm_mul_ps(x, x);
    sum = _mm_add_ps(sum, x);
  }
  for (; d < len; ++d, ++p) {
    x = _mm_load_ss(p);
    x = _mm_mul_ss(x, x);
    sum = _mm_add_ss(sum, x);
  }

  // Col(0)[row] = std::sqrt( std::max(floor, (sum.m128_f32[0] + sum.m128_f32[1] + sum.m128_f32[2] + sum.m128_f32[3]) * alpha));
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  _mm_store_ss(result, sum);
  *result *= alpha;
  *result = std::max(floor, *result);
  *result = std::sqrt(*result);
}

float max_abs_sse(const float *data, size_t len) {
  __m128 zero = _mm_set1_ps(0), maxf = zero, d;

  size_t i = 0;
  for (i = 0; i+4 <= len; i += 4, data += 4) {
    d = _mm_loadu_ps(data);
    maxf = _mm_max_ps(maxf, d);
    d = _mm_sub_ps(zero, d);
    maxf = _mm_max_ps(maxf, d);
  }
  for (; i < len; i++, data++) {
    d = _mm_load_ss(data);
    maxf = _mm_max_ss(maxf, d);
    d = _mm_sub_ss(zero, d);
    maxf = _mm_max_ss(maxf, d);
  }

#ifdef _MSC_VER
  __declspec(align(16)) float buff[4];
#else
  __attribute__((aligned(16))) float buff[4];
#endif

  _mm_store_ps(buff, maxf);
  buff[0] = std::max(buff[0], buff[1]);
  buff[0] = std::max(buff[0], buff[2]);
  buff[0] = std::max(buff[0], buff[3]);

  return buff[0];
}

void max_min_sse(const float *data, size_t len, float &max_f, float &min_f) {

  __m128 maxf = _mm_set1_ps(max_f), minf = _mm_set1_ps(min_f), d;
  size_t i = 0;
  for (i = 0; i + 4 <= len; i += 4, data += 4) {
    d = _mm_loadu_ps(data);
    maxf = _mm_max_ps(maxf, d);
    minf = _mm_min_ps(minf, d);
  }
  for (; i < len; i++, data++) {
    d = _mm_load_ss(data);
    maxf = _mm_max_ss(maxf, d);
    minf = _mm_min_ss(minf, d);
  }

#ifdef _MSC_VER
  __declspec(align(16)) float buff[4];
#else
  __attribute__((aligned(16))) float buff[4];
#endif

  _mm_store_ps(buff, maxf);
  buff[0] = std::max(buff[0], buff[1]);
  buff[0] = std::max(buff[0], buff[2]);
  buff[0] = std::max(buff[0], buff[3]);
  max_f = buff[0];

  _mm_store_ps(buff, minf);
  buff[0] = std::min(buff[0], buff[1]);
  buff[0] = std::min(buff[0], buff[2]);
  buff[0] = std::min(buff[0], buff[3]);
  min_f = buff[0];
}

#if defined(__INTEL_COMPILER)
#define SIGMODSSE(offset) {x[offset] = _mm_loadu_ps(pCol);x[offset]=_mm_sub_ps(zero,x[offset]);\
       x[offset]=_mm_min_ps(x[offset], explimit);x[offset]=_mm_exp_ps(x[offset]);\
       x[offset]=_mm_add_ps(x[offset], one);x[offset]=_mm_div_ps(one, x[offset]);_mm_storeu_ps(pCol,x[offset]);}
void sigmod_sse_svml(float *pCol, size_t len) {
#define EXPLIMIT 88.722008f
  __m128 x[0], zero = _mm_set1_ps(0), one = _mm_set1_ps(1.0f);
  __m128 explimit = _mm_set1_ps(88.722008f);

  size_t row = 0;

  for (; row + 4 <= len; row += 4, pCol += 4) {
    SIGMODSSE(0);
  }

  for (; row < len; row++, pCol++) {
    float expbuffer = std::min<float>(-*pCol, EXPLIMIT);
    expbuffer = expf(expbuffer);
    *pCol = 1.0f / (1.0f + expbuffer);
  }
}
#endif
