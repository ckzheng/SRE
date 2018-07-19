#include "xnn_neon.h"

#ifdef USE_NEON
#include <arm_neon.h>
#include <limits>
#include <cmath>
using namespace std;

#ifdef _MSC_VER
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE __attribute__((always_inline)) inline
#endif

inline float getsum(float32x4_t sum) {
  float32x2_t high = vget_high_f32(sum);
  float32x2_t low = vget_low_f32(sum);
  float32x2_t result = vpadd_f32(high, low);
  float ret = vget_lane_f32(result, 0) + vget_lane_f32(result, 1);
  return ret;
}

inline int getsum(int32x4_t sum) {
  int32x2_t high = vget_high_s32(sum);
  int32x2_t low = vget_low_s32(sum);
  int32x2_t result = vpadd_s32(high, low);
  result = vpadd_s32(result, result);
  int ret = vget_lane_s32(result, 0);
  return ret;
}

#if defined __arm__ || defined __aarch64__
inline uint16_t getsum(uint16x8_t sum) {
  uint16x4_t high = vget_high_u16(sum);
  uint16x4_t low = vget_low_u16(sum);
  uint16x4_t result = vpadd_u16(high, low);
  result = vpadd_u16(result, result);
  uint16_t ret = vget_lane_u16(result, 0);
  return ret;
}
#endif

template<int N>
inline void getsum(float32x4x4_t sum, float(&val)[N]) {
  if (N >= 4) {
    val[0] = getsum(sum.val[0]);
    val[1] = getsum(sum.val[1]);
    val[2] = getsum(sum.val[2]);
    val[3] = getsum(sum.val[3]);
  }
}

template<int N>
inline void getsum(int32x4x4_t sum, int(&val)[N]) {
  if (N >= 4) {
    val[0] = getsum(sum.val[0]);
    val[1] = getsum(sum.val[1]);
    val[2] = getsum(sum.val[2]);
    val[3] = getsum(sum.val[3]);
  }
}

template<int N>
inline void getsum(uint16x8x4_t sum, uint16_t(&val)[N]) {
  if (N >= 4) {
    val[0] = getsum(sum.val[0]);
    val[1] = getsum(sum.val[1]);
    val[2] = getsum(sum.val[2]);
    val[3] = getsum(sum.val[3]);
  }
}

inline float32x4_t mul_add(float32x4_t sum, const float* row, const float* col,
    int offset) {
  float32x4_t r = vld1q_f32(row + offset);
  float32x4_t c = vld1q_f32(col + offset);
  sum = vmlaq_f32(sum, r, c);
  return sum;
}

inline int32x4_t mul_add(int32x4_t sum, const short* row, const short* col,
    int offset) {
  int16x4_t r = vld1_s16(row + offset);
  int16x4_t c = vld1_s16(col + offset);
  sum = vmlal_s16(sum, r, c);
  return sum;
}

inline uint16x8_t mul_add(uint16x8_t sum, const uint8_t* row,
    const uint8_t* col, int offset) {
  uint8x8_t r = vld1_u8(row + offset);
  uint8x8_t c = vld1_u8(col + offset);
  sum = vmlal_u8(sum, r, c);
  return sum;
}

FORCE_INLINE void mal_add4(const float* row, const float* col1,
    const float* col2, const float* col3, const float* col4, float32x4x4_t& sum,
    int offset) {
  float32x4_t r = vld1q_f32(row + offset);
  float32x4_t c1 = vld1q_f32(col1 + offset);
  sum.val[0] = vmlaq_f32(sum.val[0], r, c1);
  float32x4_t c2 = vld1q_f32(col2 + offset);
  sum.val[1] = vmlaq_f32(sum.val[1], r, c2);
  float32x4_t c3 = vld1q_f32(col3 + offset);
  sum.val[2] = vmlaq_f32(sum.val[2], r, c3);
  float32x4_t c4 = vld1q_f32(col4 + offset);
  sum.val[3] = vmlaq_f32(sum.val[3], r, c4);
}

FORCE_INLINE void mal_add4(const short* row, const short* col1,
    const short* col2, const short* col3, const short* col4, int32x4x4_t& sum,
    int offset) {
  int16x4_t r = vld1_s16(row + offset);
  int16x4_t c1 = vld1_s16(col1 + offset);
  sum.val[0] = vmlal_s16(sum.val[0], r, c1);
  int16x4_t c2 = vld1_s16(col2 + offset);
  sum.val[1] = vmlal_s16(sum.val[1], r, c2);
  int16x4_t c3 = vld1_s16(col3 + offset);
  sum.val[2] = vmlal_s16(sum.val[2], r, c3);
  int16x4_t c4 = vld1_s16(col4 + offset);
  sum.val[3] = vmlal_s16(sum.val[3], r, c4);
}

FORCE_INLINE void mal_add4(const uint8_t* row, const uint8_t* col1,
    const uint8_t* col2, const uint8_t* col3, const uint8_t* col4,
    uint16x8x4_t& sum, int offset) {
  uint8x8_t r = vld1_u8(row + offset);
  uint8x8_t c1 = vld1_u8(col1 + offset);
  sum.val[0] = vmlal_u8(sum.val[0], r, c1);
  uint8x8_t c2 = vld1_u8(col2 + offset);
  sum.val[1] = vmlal_u8(sum.val[1], r, c2);
  uint8x8_t c3 = vld1_u8(col3 + offset);
  sum.val[2] = vmlal_u8(sum.val[2], r, c3);
  uint8x8_t c4 = vld1_u8(col4 + offset);
  sum.val[3] = vmlal_u8(sum.val[3], r, c4);
}

void dotprod_neon(const float* row, const float* col, float* usij,
    const size_t dim) {
  // arm has more register than x86, 64bit*32 or 128bit*16
  float32x4_t sum = vdupq_n_f32(0.0f);
  size_t d = 0;
  for (; d + 32 <= dim; d += 32, row += 32, col += 32) {
    sum = mul_add(sum, row, col, 0);
    sum = mul_add(sum, row, col, 4);
    sum = mul_add(sum, row, col, 8);
    sum = mul_add(sum, row, col, 12);
    sum = mul_add(sum, row, col, 16);
    sum = mul_add(sum, row, col, 20);
    sum = mul_add(sum, row, col, 24);
    sum = mul_add(sum, row, col, 28);
  }

  for (; d + 4 <= dim; d += 4, row += 4, col += 4) {
    sum = mul_add(sum, row, col, 0);
  }

  float result = getsum(sum);
  for (; d<dim; d++, row++, col++){
    result += (*row)*(*col);
  }

  *usij += result;
}

void dotprod_neon(const short* row, const short* col, int* usij,
    const size_t dim) {
  // arm has more register than x86, 64bit*32 or 128bit*16
  int32x4_t sum = vdupq_n_s32(0);
  size_t d = 0;
  for (; d + 32 <= dim; d += 32, row += 32, col += 32) {
    sum = mul_add(sum, row, col, 0);
    sum = mul_add(sum, row, col, 4);
    sum = mul_add(sum, row, col, 8);
    sum = mul_add(sum, row, col, 12);
    sum = mul_add(sum, row, col, 16);
    sum = mul_add(sum, row, col, 20);
    sum = mul_add(sum, row, col, 24);
    sum = mul_add(sum, row, col, 28);
  }

  for (; d + 4 <= dim; d += 4, row += 4, col += 4) {
    sum = mul_add(sum, row, col, 0);
  }

  int result = getsum(sum);
  for (; d<dim; d++, row++, col++) {
    result += (*row)*(*col);
  }

  *usij += result;
}

void dotprod_neon(const uint8_t* row, const uint8_t* col, int* usij,
    const size_t dim) {
  uint16x8_t sum = vdupq_n_u16(0);
  size_t d = 0;
  for (; d + 32 <= dim; d += 32, row += 32, col += 32) {
    sum = mul_add(sum, row, col, 0);
    sum = mul_add(sum, row, col, 4);
    sum = mul_add(sum, row, col, 8);
    sum = mul_add(sum, row, col, 12);
    sum = mul_add(sum, row, col, 16);
    sum = mul_add(sum, row, col, 20);
    sum = mul_add(sum, row, col, 24);
    sum = mul_add(sum, row, col, 28);
  }

  for (; d + 4 <= dim; d += 4, row += 4, col += 4) {
    sum = mul_add(sum, row, col, 0);
  }

  int result = getsum(sum);
  for (; d<dim; d++, row++, col++) {
    result += (*row)*(*col);
  }

  *usij += result;
}

void dotprod4_neon(const float* row, const float* col, size_t col_stride,
    float* usij, size_t usij_stride, size_t dim) {
  float32x4x4_t sum;
  sum.val[0] = sum.val[1] = sum.val[2] = sum.val[3] = vdupq_n_f32(0);
  const float* col1 = col;
  const float* col2 = col1 + col_stride;
  const float* col3 = col2 + col_stride;
  const float* col4 = col3 + col_stride;

  size_t d = 0;
  for (; d + 32 <= dim; d += 32, row += 32, col1 += 32, col2 += 32, col3 += 32,
      col4 += 32) {
    mal_add4(row, col1, col2, col3, col4, sum, 0);
    mal_add4(row, col1, col2, col3, col4, sum, 4);
    mal_add4(row, col1, col2, col3, col4, sum, 8);
    mal_add4(row, col1, col2, col3, col4, sum, 12);
    mal_add4(row, col1, col2, col3, col4, sum, 16);
    mal_add4(row, col1, col2, col3, col4, sum, 20);
    mal_add4(row, col1, col2, col3, col4, sum, 24);
    mal_add4(row, col1, col2, col3, col4, sum, 28);
  }

  for (; d + 4 <= dim; d += 4, row += 4, col1 += 4, col2 += 4, col3 += 4,
      col4 += 4) {
    mal_add4(row, col1, col2, col3, col4, sum, 0);
  }

  float result[4];
  getsum(sum, result);
  for (; d<dim; d++, row++, col1++, col2++, col3++, col4++) {
    float rowValue = *row;
    result[0] += rowValue * (*col1);
    result[1] += rowValue * (*col2);
    result[2] += rowValue * (*col3);
    result[3] += rowValue * (*col4);
  }

  *usij += result[0];
  usij += usij_stride;
  *usij += result[1];
  usij += usij_stride;
  *usij += result[2];
  usij += usij_stride;
  *usij += result[3];
}

void dotprod4_neon(const short* row, const short* col, size_t col_stride,
    int* usij, size_t usij_stride, size_t dim) {
  int32x4x4_t sum;
  sum.val[0] = sum.val[1] = sum.val[2] = sum.val[3] = vdupq_n_s32(0);
  const short* col1 = col;
  const short* col2 = col1 + col_stride;
  const short* col3 = col2 + col_stride;
  const short* col4 = col3 + col_stride;

  size_t d = 0;
  for (; d + 32 <= dim; d += 32, row += 32, col1 += 32, col2 += 32, col3 += 32,
      col4 += 32) {
    mal_add4(row, col1, col2, col3, col4, sum, 0);
    mal_add4(row, col1, col2, col3, col4, sum, 4);
    mal_add4(row, col1, col2, col3, col4, sum, 8);
    mal_add4(row, col1, col2, col3, col4, sum, 12);
    mal_add4(row, col1, col2, col3, col4, sum, 16);
    mal_add4(row, col1, col2, col3, col4, sum, 20);
    mal_add4(row, col1, col2, col3, col4, sum, 24);
    mal_add4(row, col1, col2, col3, col4, sum, 28);
  }

  for (; d + 4 <= dim; d += 4, row += 4, col1 += 4, col2 += 4, col3 += 4,
      col4 += 4) {
    mal_add4(row, col1, col2, col3, col4, sum, 0);
  }

  int result[4];
  getsum(sum, result);
  for (; d<dim; d++, row++, col1++, col2++, col3++, col4++) {
    short rowValue = *row;
    result[0] += rowValue * (*col1);
    result[1] += rowValue * (*col2);
    result[2] += rowValue * (*col3);
    result[3] += rowValue * (*col4);
  }

  *usij += result[0];
  usij += usij_stride;
  *usij += result[1];
  usij += usij_stride;
  *usij += result[2];
  usij += usij_stride;
  *usij += result[3];
}

void dotprod4_neon(const uint8_t* row, const uint8_t* col, size_t col_stride,
    int* usij, size_t usij_stride, size_t dim) {
  uint16x8x4_t sum;
  sum.val[0] = sum.val[1] = sum.val[2] = sum.val[3] = vdupq_n_u16(0);
  const uint8_t* col1 = col;
  const uint8_t* col2 = col1 + col_stride;
  const uint8_t* col3 = col2 + col_stride;
  const uint8_t* col4 = col3 + col_stride;

  size_t d = 0;
  for (; d + 32 <= dim; d += 32, row += 32, col1 += 32, col2 += 32, col3 += 32,
      col4 += 32) {
    mal_add4(row, col1, col2, col3, col4, sum, 0);
    mal_add4(row, col1, col2, col3, col4, sum, 4);
    mal_add4(row, col1, col2, col3, col4, sum, 8);
    mal_add4(row, col1, col2, col3, col4, sum, 12);
    mal_add4(row, col1, col2, col3, col4, sum, 16);
    mal_add4(row, col1, col2, col3, col4, sum, 20);
    mal_add4(row, col1, col2, col3, col4, sum, 24);
    mal_add4(row, col1, col2, col3, col4, sum, 28);
  }

  for (; d + 4 <= dim; d += 4, row += 4, col1 += 4, col2 += 4, col3 += 4,
      col4 += 4) {
    mal_add4(row, col1, col2, col3, col4, sum, 0);
  }

  uint16_t result[4];
  getsum(sum, result);
  for (; d<dim; d++, row++, col1++, col2++, col3++, col4++) {
    uint8_t rowValue = *row;
    result[0] += rowValue * (*col1);
    result[1] += rowValue * (*col2);
    result[2] += rowValue * (*col3);
    result[3] += rowValue * (*col4);
  }

  *usij += result[0];
  usij += usij_stride;
  *usij += result[1];
  usij += usij_stride;
  *usij += result[2];
  usij += usij_stride;
  *usij += result[3];
}

void add_neon(float* summand, const float* addend, size_t size) {
  size_t d = 0;
  float* p = summand;
  const float* q = addend;
  for (; d + 4 <= size; d += 4, p += 4, q += 4) {
    float32x4_t x = vld1q_f32(p);
    float32x4_t y = vld1q_f32(q);
    x = vaddq_f32(x, y);
    vst1q_f32(p, x);
  }

  for (; d < size; d++, p++, q++) {
    *p += *q;
  }
}

void scaleadd_neon(const float scale, float* summand, const int* addend,
    size_t size) {
  size_t d = 0;
  float* p = summand;
  const int* q = addend;
  float32x4_t scalev = vdupq_n_f32(scale);
  for (; d + 4 <= size; d += 4, p += 4, q += 4) {
    float32x4_t pf = vld1q_f32(p);
    int32x4_t qi = vld1q_s32(q);
    float32x4_t qf = vcvtq_f32_s32(qi);
    pf = vmlaq_f32(pf, qf, scalev);
    vst1q_f32(p, pf);
  }

  for (; d < size; d++, p++, q++) {
    *p += *q * scale;
  }
}

float max_abs_neon(const float* v, size_t size) {
  size_t d = 0;
  const float* p = v;
  float32x4_t maxv = vdupq_n_f32(0);
  for (; d + 4 <= size; d += 4, p += 4) {
    float32x4_t tempv = vld1q_f32(p);
    tempv = vabsq_f32(tempv);
    maxv = vmaxq_f32(tempv, maxv);
  }

  float temp_max[4];
  vst1q_f32(temp_max, maxv);
  float ret = temp_max[0];
  for (int i = 1; i < 4; i++) {
    if (ret < temp_max[i]) {
      ret = temp_max[i];
    }
  }

  for (; d < size; d++, p++) {
    float temp = *p >= 0 ? *p : -*p;
    ret = ret > temp ? ret : temp;
  }

  return ret;
}

short quantize(float v, float coeff) {
  v *= coeff;
  v = round(v);
  short short_max = std::numeric_limits<short>::max();
  if (v < -(short_max + 1.0f)) {
    return -(short_max + 1);
  } else if (v > short_max) {
    return short_max;
  } else {
    return (short)v;
  }
}

void quantize_neon(const float* f, short* s, size_t size, float coeff) {
  float32x4_t coeffv = vdupq_n_f32(coeff);
  int mins = std::numeric_limits<short>::min();
  int maxs = std::numeric_limits<short>::max();
  int32x4_t miniv = vdupq_n_s32(mins);
  int32x4_t maxiv = vdupq_n_s32(maxs);
  float32x4_t maxfv = vdupq_n_f32(maxs);
  float32x4_t minfv = vdupq_n_f32(mins);
  float32x4_t halfv = vdupq_n_f32(0.5);
  float32x4_t nhalfv = vdupq_n_f32(-0.5);
  float32x4_t zerov = vdupq_n_f32(0);
  size_t d = 0;
  const float* pf = f;
  short* ps = s;
  for (; d + 4 <= size; d += 4, pf += 4, ps += 4) {
    float32x4_t tempf = vld1q_f32(pf);
    tempf = vmulq_f32(coeffv, tempf);
    // get postive part
    float32x4_t p_tempf = vmaxq_f32(zerov, tempf);
    p_tempf = vaddq_f32(halfv, p_tempf);
    p_tempf = vminq_f32(maxfv, p_tempf);
    int32x4_t p_tempi = vcvtq_s32_f32(p_tempf);

    // get negtive part
    float32x4_t n_tempf = vminq_f32(zerov, tempf);
    n_tempf = vaddq_f32(nhalfv, n_tempf);
    n_tempf = vmaxq_f32(minfv, n_tempf);
    int32x4_t n_tempi = vcvtq_s32_f32(n_tempf);

    int32x4_t tempi = vaddq_s32(p_tempi, n_tempi);
    int16x4_t temps = vqmovn_s32(tempi);
    vst1_s16(ps, temps);
  }

  for (; d < size; d++, pf++, ps++) {
    *ps = quantize(*pf, coeff);
  }
}

void relu_neon(float* v, size_t size) {
  float32x4_t zero = vdupq_n_f32(0);
  size_t d = 0;
  float* p = v;
  for (; d + 4 <= size; d += 4, p += 4) {
    float32x4_t x = vld1q_f32(p);
    x = vmaxq_f32(x, zero);
    vst1q_f32(p, x);
  }

  for (; d < size; d++) {
    if (*p < 0) *p = 0;
  }
}
#endif

