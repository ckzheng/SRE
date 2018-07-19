#ifndef _XNN_SSE_H
#define _XNN_SSE_H
#include<pmmintrin.h>
#include<stdint.h>
#if defined USE_SSE41
 #include<smmintrin.h>
#endif
void dotprod8_sse(const short *wtrowi, const short *vcolt, const size_t cols4stride, int *usij, const size_t usijstride, const size_t dim);
void dotprod4_sse(const short *wtrowi, const short *vcolt, const size_t cols4stride, int *usij, const size_t usijstride, const size_t dim);
void dotprod_sse(const short *wtrowi, const short *vcolt, int *usij, const size_t dim);
void dotprod4_sse(const uint8_t *wtrowi, const uint8_t *vcolt, const size_t cols4stride, int *usij, const size_t usijstride, const size_t dim);
void dotprod_sse(const uint8_t *wtrowi, const uint8_t *vcolt, int *usij, const size_t dim);
void dotprod4_sse(const float *row, const float *cols4, const size_t cols4stride, float *usij, const size_t usijstride, const size_t dim);
void dotprod_sse(const float *row, const float *col, float *usij, const size_t dim);
void scaleadd_sse(const float scale, float *usij, const float *col, const size_t dim);
void scaleadd_sse(float *usij, const float *col, const float scale, const size_t dim);
void scaleadd_sse(const float scale, float *usij, const int *col, const size_t dim);
void add_sse(float *usij, const float *col, const size_t dim);
void minus_div_sse(float* p, const float* pmhead, const float* pvhead, size_t len);
void minus_sse(float* p, const float* pv, size_t len);
void div_norm_sse(float* p, float divider, size_t len);
void plus_sse(float* u, const float* v, size_t len);
void scale_plus_prod_sse(float* p, const float* pX, const float* pY, size_t len, float scale);
void scale_plus_prod_sse(float *p, const float *pX, const float *pY, float scale, size_t len);
void sigmod_sse(float* pCol, size_t len);
void sigmod_sse_svml(float* pCol, size_t len);
void sigmod_sse(float* pSrcCol, float* pDstCol, size_t len);
void relu_sse(float* pCol, size_t len);
void tanh_sse(float* p, size_t len);
void tanh_sse(float* psrc,float *pdst, size_t len);
void group_pnorm_sse(float* result, const float* p, size_t len, size_t groupsize);
void l2norm_sse(float* result, const float* p, size_t len, float floor, float alpha);
float max_abs_sse(const float* data, size_t len);
void max_min_sse(const float* data, size_t len, float& max_f, float& min_f);
void multi_sse(float *usij, const float *col, const size_t dim);
void quantize_sse(short* sv, const float* fv, size_t len,float invscale, short QMAX);
void quantize_non_symmetric_sse(int8_t* sv, const float* fv, size_t len, float minf_, float invscale, int8_t QMAX, int& curr_col_sum);
void quantize_non_symmetric_sse(uint8_t* sv, const float* fv, size_t len, float minf_, float invscale, uint8_t QMAX, uint32_t& curr_col_sum);
#endif

