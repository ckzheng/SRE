#ifndef _XNN_NEON_H
#define _XNN_NEON_H
#include <stdint.h>
#include <stddef.h>

void dotprod_neon(const float* row, const float* col, float* usij,
    const size_t dim);
void dotprod_neon(const short* row, const short* col, int* usij,
    const size_t dim);
void dotprod_neon(const uint8_t* row, const uint8_t* col, int* usij,
    const size_t dim);

void dotprod4_neon(const float* row, const float* col, size_t col_stride,
    float* usij, size_t usij_stride, size_t dim);
void dotprod4_neon(const short* row, const short* col, size_t col_stride,
    int* usij, size_t usij_stride, size_t dim);
void dotprod4_neon(const uint8_t* row, const uint8_t* col, size_t col_stride,
    int* usij, size_t usij_stride, size_t dim);

void add_neon(float* summand, const float* addend, size_t size);
void scaleadd_neon(const float scale, float* summand, const int* addend,
    size_t size);

float max_abs_neon(const float* v, size_t size);
void quantize_neon(const float* f, short* s, size_t size, float coeff);
void relu_neon(float* v, size_t size);

#endif

