// xnn-evaluator/xnn-runtime.h

// Copyright 2015 Alibaba-inc  [zhijie.yzj]

#ifndef XNN_RUNTIME_H_
#define XNN_RUNTIME_H_

#include <vector>
#include <algorithm>
#include <limits>
#include <malloc.h>
#include <cmath>
#include <string.h>
#include "base/log_message.h"
#include "base/serialize_helper.h"
#include "base/idec_types.h"
#include "base/idec_common.h"

#ifdef USE_SSE
#include"am/xnn_sse.h"
#endif

#if defined(USE_AVX2)
#include "am/xnn_avx2.h"
#endif

#ifdef USE_NEON
#include "am/xnn_neon.h"
#endif

#ifdef USE_MKL
#include <mkl.h>
#endif

namespace idec {
#ifdef USE_AVX2
#define BYTE_ALIGN 32
#else
#define BYTE_ALIGN 16
#endif

typedef float Real;

const float kLogZeroFloat = -std::numeric_limits<float>::infinity();

class xnnRuntimeMatrixItf {
 protected:
  virtual ~xnnRuntimeMatrixItf() {};

  virtual void Serialize(SerializeHelper &helper) = 0;
  virtual void Deserialize(SerializeHelper &helper) = 0;
};

template<typename T> class xnnRuntimeMatrixBase : public xnnRuntimeMatrixItf {
 protected:
  size_t num_rows_;
  size_t num_cols_;
  T *data_;
  size_t allocated_bytes_;

 protected:
  void alloc() {
    if (num_rows_ * num_cols_ != 0) {
      if (num_rows_ * num_cols_ * sizeof(T) > allocated_bytes_) {
        // 20% more space is allocated than actually needed
        //float overallocateratio = 1.2f;
        size_t allocated_bytes = size_t(num_rows_ * num_cols_ * sizeof(
                                          T) /* overallocateratio */);
        if ((data_ = (T *)realloc(data_, allocated_bytes)) == NULL)
          throw std::bad_alloc();

        memset((unsigned char *)data_ + allocated_bytes_, 0,
               allocated_bytes -
               allocated_bytes_); // [zhijie.yzj] need to do this because NaNs in data_ may cause bad things, e.g., NaN * 0 != 0
        allocated_bytes_ = allocated_bytes;
      }
    } else {
      num_rows_ = num_cols_ = 0;
    }
  }

 public:
  // constructors and deconstructor
  xnnRuntimeMatrixBase() : num_rows_(0), num_cols_(0), data_(NULL),
    allocated_bytes_(0) {};
  xnnRuntimeMatrixBase(size_t num_rows, size_t num_cols) : num_rows_(num_rows),
    num_cols_(num_cols), data_(NULL), allocated_bytes_(0) {};

  virtual ~xnnRuntimeMatrixBase() {
    if (data_ != NULL) {
#ifdef _MSC_VER
      _aligned_free(data_);
      data_=NULL;
#else
      free(data_);
      data_ = NULL;
#endif
    }
  }

  // basic
  size_t NumRows() const { return num_rows_; }
  size_t NumCols() const { return num_cols_; }
  size_t ElementSize() const { return sizeof(T); }

  void Resize(size_t num_rows, size_t num_cols) {
    if (num_rows_ == num_rows && num_cols_ == num_cols) return;
    num_rows_ = num_rows;
    num_cols_ = num_cols;
    alloc();
  }

  void Swap(xnnRuntimeMatrixBase &M) {
    std::swap(num_rows_, M.num_rows_);
    std::swap(num_cols_, M.num_cols_);
    std::swap(data_, M.data_);
    std::swap(allocated_bytes_, M.allocated_bytes_);
  }

  bool Empty() const { return(num_rows_ * num_cols_ == 0); }
  void Clear() { num_rows_ = num_cols_ = 0; };

  virtual void Serialize(SerializeHelper &helper) {
    uint32 num_row32 = static_cast<uint32>(num_rows_);
    uint32 num_col32 = static_cast<uint32>(num_cols_);

    helper.Serialize(num_row32);
    helper.Serialize(num_col32);
    helper.Serialize(data_, num_rows_ * num_cols_ * sizeof(T));
  }

  virtual void Deserialize(SerializeHelper &helper) {
    uint32 num_row32 = 0;
    uint32 num_col32 = 0;
    helper.Deserialize(num_row32);
    helper.Deserialize(num_col32);
    num_rows_ = static_cast<size_t>(num_row32);
    num_cols_ = static_cast<size_t>(num_col32);

    alloc();
    helper.Deserialize(data_, num_rows_ * num_cols_ * sizeof(T));
  }
};

template<typename T> class xnnRuntimeColumnMatrix : public
  xnnRuntimeMatrixBase<T> {
 protected:
  using xnnRuntimeMatrixBase<T>::num_rows_;
  using xnnRuntimeMatrixBase<T>::num_cols_;
  using xnnRuntimeMatrixBase<T>::allocated_bytes_;
  using xnnRuntimeMatrixBase<T>::data_;
  size_t col_stride_;    // always stored in column-major, which is different from Kaldi

 protected:
  void alloc() {
    if (num_rows_ * num_cols_ != 0) {
      //size_t skip = ((BYTE_ALIGN / sizeof(T)) - num_rows_ % (BYTE_ALIGN / sizeof(T))) % (BYTE_ALIGN / sizeof(T));
      //col_stride_ = num_rows_ + skip;
      size_t block = BYTE_ALIGN / sizeof(T);
      col_stride_ = (num_rows_ + block - 1) / block * block;

      if (num_cols_ * col_stride_ * sizeof(T) > allocated_bytes_) {
        // 20% more space is allocated than actually needed
        //float overallocateratio = 1.2f;
        size_t allocated_bytes = size_t(num_cols_ * col_stride_ * sizeof(
                                          T) /* overallocateratio */);

#ifdef _MSC_VER
        if ((data_ = (data_ == NULL) ? (T *)_aligned_malloc(allocated_bytes,
                     BYTE_ALIGN) : (T *)_aligned_realloc(data_, allocated_bytes,
                         BYTE_ALIGN)) == NULL)
          throw std::bad_alloc();
#else
        void *data = NULL;
        void *free_data= NULL;
        data = IDEC_MEMALIGN(BYTE_ALIGN, allocated_bytes, &free_data);
        if (data == NULL)
          throw std::bad_alloc();
        if (data_ != NULL) {
          memcpy(data, data_, allocated_bytes_);
          free(data_);
        }
        data_ = reinterpret_cast<T *>(data);
#endif

        memset((unsigned char *)data_ + allocated_bytes_, 0,
               allocated_bytes -
               allocated_bytes_); // [zhijie.yzj] need to do this because NaNs in data_ may cause bad things, e.g., NaN * 0 != 0
        allocated_bytes_ = allocated_bytes;
      }
    } else {
      num_rows_ = num_cols_ = col_stride_ = 0;
      /*if(data_ != NULL)
      {
      _aligned_free(data_);
      data_ = NULL;
      }*/
    }
  }

 public:
  xnnRuntimeColumnMatrix() : xnnRuntimeMatrixBase<T>(), col_stride_(0) {};
  xnnRuntimeColumnMatrix(const xnnRuntimeColumnMatrix &mat) :
    xnnRuntimeMatrixBase<T>() { *this = mat; };
  xnnRuntimeColumnMatrix(size_t num_rows,
                         size_t num_cols) : xnnRuntimeMatrixBase<T>(num_rows, num_cols),
    col_stride_(0) {
    alloc();
  }

  /*xnnRuntimeColumnMatrix(const kaldi::Matrix<T> &kaldiMat) : xnnRuntimeMatrixBase<T>(kaldiMat.NumRows(), kaldiMat.NumCols())
  {
  alloc();
  for(size_t col = 0; col < num_cols_; ++col)
  {
  float *pCol = Col(col);
  for(size_t row = 0; row < num_rows_; ++row)
  {
  pCol[row] = kaldiMat.Row(row).Data()[col];
  }
  }
  }

  xnnRuntimeColumnMatrix(const kaldi::Vector<T> &kaldiVec) : xnnRuntimeMatrixBase<T>(kaldiVec.Dim(), 1)
  {
  alloc();
  float *pCol = Col(0);
  for(size_t row = 0; row < num_rows_; ++row)
  {
  pCol[row] = kaldiVec.Data()[row];
  }
  }*/

  // dim
  size_t ColStride() const { return col_stride_; }

  // accesssing
  inline T *Col(size_t col) const {
    return(data_ + col * col_stride_);
  }

  // value setting
  void SetZero() {
    //memset(data_, 0, sizeof(T)*num_cols_*col_stride_);
    for (size_t col = 0; col < num_cols_; ++col) {
      memset(Col(col), 0, sizeof(T)*num_rows_);
    }
  }

  const xnnRuntimeColumnMatrix &operator= (const xnnRuntimeColumnMatrix &mat) {
    num_rows_ = mat.NumRows();
    num_cols_ = mat.NumCols();
    alloc();

    for (size_t col = 0; col < num_cols_; ++col) {
#ifdef _MSC_VER
      memcpy_s(Col(col), sizeof(T) * num_rows_, mat.Col(col), sizeof(T) * num_rows_);
#else
      memcpy(Col(col), mat.Col(col), sizeof(T) * num_rows_);
#endif
    }

    return(*this);
  }

  void Resize(size_t num_rows, size_t num_cols) {
    if (num_rows_ == num_rows && num_cols_ == num_cols) return;
    num_rows_ = num_rows;
    num_cols_ = num_cols;
    alloc();
  }

  void Swap(xnnRuntimeColumnMatrix &M) {
    xnnRuntimeMatrixBase<T>::Swap(M);
    std::swap(col_stride_, M.col_stride_);
  }

  void Log() {
    for (size_t col = 0; col < num_cols_; ++col) {
      T *p = Col(col);
      for (size_t row = 0; row < num_rows_; ++row) {
        p[row] = log(p[row]);
      }
    }
  }

  // compare
  void ApplyFloor(T val) {
    for (size_t i = 0; i < num_cols_; i++) {
      for (size_t j = 0; j < num_rows_; j++) {
        Col(i)[j] = std::max(Col(i)[j], val);
      }
    }
  }

  void ApplyCeiling(T val) {
    for (size_t i = 0; i < num_cols_; i++) {
      for (size_t j = 0; j < num_rows_; j++) {
        Col(i)[j] = std::min(Col(i)[j], val);
      }
    }
  }

  virtual void Serialize(SerializeHelper &helper) {
    uint32 num_row32 = static_cast<uint32>(num_rows_);
    uint32 num_col32 = static_cast<uint32>(num_cols_);

    helper.Serialize(num_row32);
    helper.Serialize(num_col32);
    for (size_t col = 0; col < num_cols_; ++col) {
      helper.Serialize(Col(col), num_rows_ * sizeof(T));
    }
  }

  virtual void Deserialize(SerializeHelper &helper) {
    uint32 num_row32 = 0;
    uint32 num_col32 = 0;
    helper.Deserialize(num_row32);
    helper.Deserialize(num_col32);
    num_rows_ = static_cast<size_t>(num_row32);
    num_cols_ = static_cast<size_t>(num_col32);

    alloc();
    for (size_t col = 0; col < num_cols_; ++col) {
      helper.Deserialize(Col(col), num_rows_ * sizeof(T));
    }
  }

  void CopyRows(const xnnRuntimeColumnMatrix &Mt,
                const std::vector<size_t> &Vt) {
    // dim check
    if (Vt.size() != num_rows_) {
      IDEC_ERROR << "dimension mismatch " << Vt.size() << " vs. " << num_rows_;
    }


    for (size_t col = 0; col < num_cols_; ++col) {
      T *pCol = Col(col);

      size_t row;
      for (row = 0; row < num_rows_; row++) {
        pCol[row] = Mt.Col(col)[Vt[row]];
      }

    }
  }

  void CopyFloatSubMatrix(const xnnRuntimeColumnMatrix &Mt, size_t src_start_col,
                          size_t src_start_row, size_t des_start_col, size_t des_start_row,
                          size_t num_cols, size_t num_rows) {
    for (size_t col = 0; col < num_cols; ++col) {
      T *pCol_des = Col(des_start_col + col);
      T *pCol_src = Mt.Col(src_start_col + col);
      //std::cout << sizeof(pCol_des[0]) << "  " << sizeof(pCol_src[0]) << std::endl;
      memcpy(pCol_des + des_start_row, pCol_src + src_start_row,
             num_rows * sizeof(T));
    }
  }
};

class xnnShortRuntimeMatrix : public xnnRuntimeColumnMatrix < short > {
 public:
  // constructors
  xnnShortRuntimeMatrix() : xnnRuntimeColumnMatrix<short>() {};
  xnnShortRuntimeMatrix(size_t num_rows,
                        size_t num_cols) : xnnRuntimeColumnMatrix<short>(num_rows, num_cols) {};
};

class xnnCharRuntimeMatrix : public xnnRuntimeColumnMatrix < int8_t > {
 public:
  xnnCharRuntimeMatrix() : xnnRuntimeColumnMatrix<int8_t>() {};
  xnnCharRuntimeMatrix(size_t num_rows,
                       size_t num_cols) : xnnRuntimeColumnMatrix<int8_t>(num_rows, num_cols) {};
};

class xnnUnsignedCharRuntimeMatrix : public xnnRuntimeColumnMatrix
  < uint8_t > {
 public:
  xnnUnsignedCharRuntimeMatrix() : xnnRuntimeColumnMatrix<uint8_t>() {};
  xnnUnsignedCharRuntimeMatrix(size_t num_rows,
                               size_t num_cols) : xnnRuntimeColumnMatrix<uint8_t>(num_rows, num_cols) {};
};

class xnnFloat16RuntimeMatrix;

class xnnFloat8RuntimeMatrix;

class xnnUnsignedFloat8RuntimeMatrix;

class xnnFloatRuntimeMatrix : public xnnRuntimeColumnMatrix<float> {
 protected:
  using xnnRuntimeColumnMatrix<float>::col_stride_;

 protected:

  inline void dotprod4(const float *row, const float *cols4,
                       const size_t cols4stride, float *usij, const size_t usijstride,
                       const size_t dim) {
#ifdef USE_SSE
    dotprod4_sse(row, cols4, cols4stride, usij, usijstride, dim);
#elif defined USE_NEON
    dotprod4_neon(row, cols4, cols4stride, usij, usijstride, dim);
#else
    const float *p = row;
    const float *p0 = cols4, *p1 = p0 + cols4stride, *p2 = p1 + cols4stride,
                 *p3 = p2 + cols4stride;
    float sum[4];
    size_t d;

    sum[0] = sum[1] = sum[2] = sum[3] = 0.0;
    for (d = 0; d < dim; d++) {
      sum[0] += p0[d] * p[d];
      sum[1] += p1[d] * p[d];
      sum[2] += p2[d] * p[d];
      sum[3] += p3[d] * p[d];
    }

    for (d = 0; d < 4; d++)
      *(usij + d*usijstride) += sum[d];
#endif
  }

  inline void dotprod4(const short *wtrowi, const short *vcolt,
                       const size_t cols4stride, int *usij, const size_t usijstride,
                       const size_t dim) {
#ifdef USE_SSE
#ifdef USE_AVX2
    dotprod4_avx2(wtrowi, vcolt, cols4stride, usij, usijstride, dim);
#else
    dotprod4_sse(wtrowi, vcolt, cols4stride, usij, usijstride, dim);
#endif
#elif defined USE_NEON
    dotprod4_neon(wtrowi, vcolt, cols4stride, usij, usijstride, dim);
#else
    const short *p = wtrowi;
    const short *p0 = vcolt, *p1 = p0 + cols4stride, *p2 = p1 + cols4stride,
                 *p3 = p2 + cols4stride;
    int sum[4];
    size_t d;

    sum[0] = sum[1] = sum[2] = sum[3] = 0;
    for (d = 0; d < dim; d++) {
      sum[0] += p0[d] * p[d];
      sum[1] += p1[d] * p[d];
      sum[2] += p2[d] * p[d];
      sum[3] += p3[d] * p[d];
    }

    for (d = 0; d < 4; d++)
      *(usij + d*usijstride) += sum[d];
#endif


  }

  inline void dotprod4(const uint8_t *wtrowi, const uint8_t *vcolt,
                       const size_t cols4stride, int *usij, const size_t usijstride,
                       const size_t dim) {

#if defined (USE_SSE) && defined(USE_SSE41)
//#ifdef USE_AVX2
//            dotprod4_avx2(wtrowi, vcolt, cols4stride, usij, usijstride, dim);
//#else
    dotprod4_sse(wtrowi, vcolt, cols4stride, usij, usijstride, dim);
//#endif
#elif defined USE_NEON
    dotprod4_neon(wtrowi, vcolt, cols4stride, usij, usijstride, dim);
#else

    const uint8_t *p = wtrowi;
    const uint8_t *p0 = vcolt, *p1 = p0 + cols4stride, *p2 = p1 + cols4stride,
                   *p3 = p2 + cols4stride;
    int sum[4];
    size_t d;
    //int QMIN = -(std::numeric_limits<int8_t>::max() + 1);
    sum[0] = sum[1] = sum[2] = sum[3] = 0;
    for (d = 0; d < dim; d++) {
      sum[0] += (p0[d]) * (p[d]);
      sum[1] += (p1[d]) * (p[d]);
      sum[2] += (p2[d]) * (p[d]);
      sum[3] += (p3[d]) * (p[d]);
    }

    for (d = 0; d < 4; d++)
      *(usij + d*usijstride) += sum[d];
#endif
  }

  inline void dotprod(const float *row, const float *col, float *usij,
                      const size_t dim) {
#ifdef USE_SSE
    dotprod_sse(row, col, usij, dim);
#elif defined USE_NEON
    dotprod_neon(row, col, usij, dim);
#else
    float sum = 0.0;
    size_t d;

    for (d = 0; d < dim; d++)
      sum += row[d] * col[d];

    *usij += sum;
#endif
  }

  inline void dotprod(const short *wtrowi, const short *vcolt, int *usij,
                      const size_t dim) {
#ifdef USE_SSE       // SSE implementation
#ifdef USE_AVX2
    dotprod_avx2(wtrowi, vcolt, usij, dim);
#else
    dotprod_sse(wtrowi, vcolt, usij, dim);
#endif
#elif defined USE_NEON
    dotprod_neon(wtrowi, vcolt, usij, dim);
#else       // naive baseline
    int sum = 0;
    for (size_t k = 0; k < dim; k++)
      sum += wtrowi[k] * vcolt[k];
    *usij += sum;
#endif
  }

  inline void dotprod(const uint8_t *wtrowi, const uint8_t *vcolt, int *usij,
                      const size_t dim) {
#if defined (USE_SSE) && defined(USE_SSE41)       // SSE implementation
//#ifdef USE_AVX2
//            dotprod_avx2(wtrowi, vcolt, usij, dim);
//#else
    dotprod_sse(wtrowi, vcolt, usij, dim);
//#endif
#elif defined USE_NEON
    dotprod_neon(wtrowi, vcolt, usij, dim);
#else       // naive baseline
    int sum = 0;
    //int QMIN = -(std::numeric_limits<int8_t>::max() + 1);
    for (size_t k = 0; k < dim; k++)
      sum += (wtrowi[k]) * (vcolt[k]);
    *usij += sum;
#endif
  }

  inline void dotprod8(const short *wtrowi, const short *vcolt,
                       const size_t cols4stride, int *usij, const size_t usijstride,
                       const size_t dim) {
#ifdef USE_SSE
    dotprod8_sse(wtrowi, vcolt, cols4stride, usij, usijstride, dim);
#else
    IDEC_ERROR << "Not implemented." << std::endl;
#endif
  }

  inline void scaleadd(const float scale, float *usij, const float *col,
                       const size_t dim) {
#ifdef USE_SSE
    scaleadd_sse(scale, usij, col, dim);
#else

    size_t d;

    for (d = 0; d < dim; d++)
      usij[d] = usij[d] * scale + col[d];

#endif
  }

  inline void scaleadd(float *usij, const float *col, const float scale,
                       const size_t dim) {
#ifdef USE_SSE
    scaleadd_sse(usij, col, scale, dim);
#else

    size_t d;

    for (d = 0; d < dim; d++)
      usij[d] += col[d] * scale;

#endif
  }

  inline void add(float *usij, const float *col, const size_t dim) {
#ifdef USE_SSE
    add_sse(usij, col, dim);
#elif defined USE_NEON
    add_neon(usij, col, dim);
#else

    size_t d;

    for (d = 0; d < dim; d++)
      usij[d] += col[d];

#endif
  }

  inline void add(float *usij, const int *col, const size_t dim,
                  const float scale) {
#ifdef USE_SSE
    scaleadd_sse(scale, usij, col, dim);
#elif defined USE_NEON
    scaleadd_neon(scale, usij, col, dim);
#else

    size_t d;

    for (d = 0; d < dim; d++)
      usij[d] += col[d] * scale;

#endif

  }


 public:
  // constructors
  xnnFloatRuntimeMatrix() : xnnRuntimeColumnMatrix<float>() {};
  xnnFloatRuntimeMatrix(size_t num_rows,
                        size_t num_cols) : xnnRuntimeColumnMatrix<float>(num_rows, num_cols) {};
  //xnnFloatRuntimeMatrix(const kaldi::Matrix<float> &kaldiMat) : xnnRuntimeColumnMatrix<float>(kaldiMat) {};
  //xnnFloatRuntimeMatrix(const kaldi::Vector<float> &kaldiVec) : xnnRuntimeColumnMatrix<float>(kaldiVec) {};

  // linear argebra
  inline void Addv(const xnnFloatRuntimeMatrix &v) {
#ifdef _DEBUG
    // dim check
    if (v.num_rows_ != num_rows_) {
      IDEC_ERROR << "dimension mismatch [" << v.num_rows_ << " vs. " << num_rows_;
    }
    if (v.num_cols_ != 1) {
      IDEC_ERROR << "v is not a column vector";
    }
#endif

#ifdef USE_SSE
    const float *pvhead = v.Col(0);

    for (size_t col = 0; col < num_cols_; ++col) {
      float *p = Col(col);
      add_sse(p, pvhead, num_rows_);
    }
#else
    const float *pvhead = v.Col(0);

    for (size_t col = 0; col < num_cols_; ++col) {
      float *p = Col(col);
      const float *pv = pvhead;
      size_t d;
      for (d = 0; d < num_rows_; d++) {
        p[d] = p[d] + pv[d];
      }
    }
#endif
  }

  inline void Multiv(const xnnFloatRuntimeMatrix &v) {
#ifdef _DEBUG
    // dim check
    if (v.num_rows_ != num_rows_) {
      IDEC_ERROR << "dimension mismatch [" << v.num_rows_ << " vs. " << num_rows_;
    }
    if (v.num_cols_ != 1) {
      IDEC_ERROR << "v is not a column vector";
    }
#endif

#ifdef USE_SSE
    const float *pvhead = v.Col(0);

    for (size_t col = 0; col < num_cols_; ++col) {
      float *p = Col(col);
      multi_sse(p, pvhead, num_rows_);
    }
#else
    const float *pvhead = v.Col(0);

    for (size_t col = 0; col < num_cols_; ++col) {
      float *p = Col(col);
      const float *pv = pvhead;
      size_t d;
      for (d = 0; d < num_rows_; d++)
        p[d] = p[d] * pv[d];

    }
#endif
  }

  inline void MinusvDivv(const xnnFloatRuntimeMatrix &m,
                         const xnnFloatRuntimeMatrix &v) {
#ifdef _DEBUG
    // dim check
    if (m.num_rows_ != num_rows_ || v.num_rows_ != num_rows_) {
      IDEC_ERROR << "dimension mismatch [" << m.num_rows_ << " vs. " << num_rows_ <<
                 "], [" << v.num_rows_ << " vs. " << num_rows_ << "]";
    }
    if (m.num_cols_ != 1 || v.num_cols_ != 1) {
      IDEC_ERROR << "mean and norm are not column vectors";
    }
#endif

#ifdef USE_SSE
    const float *pmhead = m.Col(0);
    const float *pvhead = v.Col(0);

    for (size_t col = 0; col < num_cols_; ++col) {
      float *p = Col(col);
      minus_div_sse(p, pmhead, pvhead, m.num_rows_);
    }
#else
    const float *pmhead = m.Col(0);
    const float *pvhead = v.Col(0);

    for (size_t col = 0; col < num_cols_; ++col) {
      float *p = Col(col);
      const float *pm = pmhead;
      const float *pv = pvhead;
      size_t d;

      for (d = 0; d < m.num_rows_; d++)
        p[d] = (p[d] - pm[d]) / pv[d];

    }
#endif
  }

  inline void Minusv(const xnnFloatRuntimeMatrix &v) {
#ifdef _DEBUG
    // dim check
    if (v.num_rows_ != num_rows_) {
      IDEC_ERROR << "dimension mismatch [" << v.num_rows_ << " vs. " << num_rows_;
    }
    if (v.num_cols_ != 1) {
      IDEC_ERROR << "v is not a column vector";
    }
#endif

#ifdef USE_SSE
    const float *pvhead = v.Col(0);

    for (size_t col = 0; col < num_cols_; ++col) {
      float *p = Col(col);
      minus_sse(p, pvhead, num_rows_);
    }
#else
    const float *pvhead = v.Col(0);

    for (size_t col = 0; col < num_cols_; ++col) {
      float *p = Col(col);
      const float *pv = pvhead;
      size_t d;
      for (d = 0; d < num_rows_; d++)
        p[d] = p[d] - pv[d];

    }
#endif
  }

  inline void DivNorm(const xnnFloatRuntimeMatrix &v) {
#ifdef _DEBUG
    // dim check
    if (v.num_rows_ != num_cols_) {
      IDEC_ERROR << "dimension mismatch [" << v.num_rows_ << " vs. " << num_cols_ <<
                 "]";
    }
    if (v.num_cols_ != 1) {
      IDEC_ERROR << "v is not a column vector";
    }
#endif

#ifdef USE_SSE
    for (size_t col = 0; col < num_cols_; ++col) {
      float *p = Col(col);
      div_norm_sse(p, v.Col(0)[col], num_rows_);
    }
#else
    float y = 0.0f;
    for (size_t col = 0; col < num_cols_; ++col) {
      float *p = Col(col);
      y = v.Col(0)[col];
      size_t d;

      for (d = 0; d < num_rows_; d++)
        p[d] = p[d] / y;
    }
#endif
  }

#if 0
  // optimized for small X (of only several columns)
  // abandon because the block version below is in general faster than this non-block version
  inline void ScalePlusMatTMat(const float scale, const xnnFloatRuntimeMatrix &W,
                               const xnnFloatRuntimeMatrix &X) {
    // this = scale * this + W^\top X
#ifdef _DEBUG
    // dim check
    if (W.num_cols_ != num_rows_ || X.num_cols_ != num_cols_
        || W.num_rows_ != X.num_rows_) {
      XNN_ERR << "dimension mismatch [" << W.num_cols_ << " x " << W.num_rows_ <<
              " * " << X.num_rows_ << " x " << X.num_cols_ << " = " << num_rows_ << " x " <<
              num_cols_ << "]";
    }
#endif

    for (size_t row = 0; row < num_rows_; ++row) {
      const float *pWhead = W.Col(row);
      for (size_t col = 0; col < num_cols_; ++col) {
        const float *pW = pWhead;
        const float *pX = X.Col(col);
#if defined USE_AVX
#elif defined USE_SSE
        __m128 sum, w, x;
        size_t d;
        sum = _mm_setzero_ps();
#if 1
        for (d = 0; d + 32 <= W.num_rows_;
             d += 32, pW += 32, pX += 32) { // 8-way loop unrolling
          w = _mm_loadu_ps(pW);
          x = _mm_loadu_ps(pX);
          w = _mm_mul_ps(w, x);
          sum = _mm_add_ps(sum, w);

          w = _mm_loadu_ps(pW + 4);
          x = _mm_loadu_ps(pX + 4);
          w = _mm_mul_ps(w, x);
          sum = _mm_add_ps(sum, w);

          w = _mm_loadu_ps(pW + 8);
          x = _mm_loadu_ps(pX + 8);
          w = _mm_mul_ps(w, x);
          sum = _mm_add_ps(sum, w);

          w = _mm_loadu_ps(pW + 12);
          x = _mm_loadu_ps(pX + 12);
          w = _mm_mul_ps(w, x);
          sum = _mm_add_ps(sum, w);

          w = _mm_loadu_ps(pW + 16);
          x = _mm_loadu_ps(pX + 16);
          w = _mm_mul_ps(w, x);
          sum = _mm_add_ps(sum, w);

          w = _mm_loadu_ps(pW + 20);
          x = _mm_loadu_ps(pX + 20);
          w = _mm_mul_ps(w, x);
          sum = _mm_add_ps(sum, w);

          w = _mm_loadu_ps(pW + 24);
          x = _mm_loadu_ps(pX + 24);
          w = _mm_mul_ps(w, x);
          sum = _mm_add_ps(sum, w);

          w = _mm_loadu_ps(pW + 28);
          x = _mm_loadu_ps(pX + 28);
          w = _mm_mul_ps(w, x);
          sum = _mm_add_ps(sum, w);
        }
#else
        for (d = 0; d + 4 <= W.num_rows_; d += 4, pW += 4, pX += 4) {
          w = _mm_loadu_ps(pW);
          x = _mm_loadu_ps(pX);
          w = _mm_mul_ps(w, x);
          sum = _mm_add_ps(sum, w);
        }
#endif
        for (; d < W.num_rows_; ++d, ++pW, ++pX) {
          w = _mm_load_ss(pW);
          x = _mm_load_ss(pX);
          w = _mm_mul_ss(w, x);
          sum = _mm_add_ss(sum, w);
        }

        float &ret = *(Col(col) + row);
        ret *= scale;
        ret += sum.m128_f32[0] + sum.m128_f32[1] + sum.m128_f32[2] + sum.m128_f32[3];
#endif
      }
    }
  }

#else

  // block-by-block version of ScalePlusMatTMat, optimized for large X (of a lot of columns)
  // refer to matprod_mtm in ssematrix.h for implementation details
  inline void ScalePlusMatTMat(const float scale,
                               const xnnFloatRuntimeMatrix &Mt, const xnnFloatRuntimeMatrix &V) {
    // this = scale * this + W^\top X
#ifdef _DEBUG
    // dim check
    if (Mt.num_cols_ != num_rows_ || V.num_cols_ != num_cols_
        || Mt.num_rows_ != V.num_rows_) {
      IDEC_ERROR << "dimension mismatch [" << Mt.num_cols_ << " x " << Mt.num_rows_
                 << " * " << V.num_rows_ << " x " << V.num_cols_ << " = " << num_rows_ << " x "
                 << num_cols_ << "]";
    }
#endif

    const size_t cacheablerowsV = 512;
    const size_t cacheablecolsV = 16;
    // 512 * 16 -> 32 KB

    const size_t colstripewV = cacheablecolsV;
    const size_t rowstripehM = 128;
    const size_t dotprodstep = cacheablerowsV;
    // 128 * 512 -> 64 KB

    const size_t colstrideV = V.ColStride();

    // loop over col stripes of V
    for (size_t j0 = 0; j0 < V.NumCols(); j0 += colstripewV) {
      const size_t j1 = std::min(j0 + colstripewV, V.NumCols());
      // stripe of V is columns [j0,j1)

      // loop over row stripes of M
      for (size_t i0 = 0; i0 < Mt.NumCols(); i0 += rowstripehM) {
        const size_t i1 = std::min(i0 + rowstripehM, Mt.NumCols());

        // loop over sub-ranges of the dot product (full dot product will exceed the L1 cache)
#ifdef _MSC_VER
        __declspec(align(16)) float patchbuffer[rowstripehM *
                                                colstripewV];    // note: don't forget column rounding
#else
        __attribute__((aligned(16))) float patchbuffer[rowstripehM *
            colstripewV];    // note: don't forget column rounding
#endif
        // 128 * 16 -> 8 KB
        memset(patchbuffer, 0, rowstripehM * colstripewV * sizeof(float));

        for (size_t k0 = 0; k0 < V.NumRows(); k0 += dotprodstep) {
          const size_t k1 = std::min(k0 + dotprodstep, V.NumRows());
          //const bool first = k0 == 0;

          for (size_t i = i0; i < i1; ++i) {
            const size_t j14 = j1 & ~3;
            for (size_t j = j0; j < j14; j += 4) {  // grouped by 4
              const float *row = Mt.Col(i) + k0;    // of length k1-k0
              const float *cols4 = V.Col(j) +
                                   k0;    // of length k1-k0, stride = V.ColStride()
              float *patchij = patchbuffer + (j - j0)*rowstripehM + (i - i0);

              dotprod4(row, cols4, colstrideV, patchij, rowstripehM, k1 - k0);
            }
            for (size_t j = j14; j < j1; ++j) {
              dotprod(Mt.Col(i) + k0, V.Col(j) + k0,
                      patchbuffer + (j - j0)*rowstripehM + (i - i0), k1 - k0);
            }
          }
        }

        // asign patch
        for (size_t j = j0; j < j1; ++j) {
          //memcpy_s(Col(j)+i0, sizeof(float)*(i1-i0), patchbuffer+(j-j0)*rowstripehM, sizeof(float)*(i1-i0));
          scaleadd(scale, Col(j) + i0, patchbuffer + (j - j0)*rowstripehM, i1 - i0);
        }
      }
    }
  }

  // add by zhuozhu.zz
  inline float Trace() {
#ifdef _DEBUG
    // dim check
    if (num_cols_ != num_rows_) {
      IDEC_ERROR << "dimension mismatch ["<< num_rows_ << " != " << num_cols_ <<
                 " when calculate trace ]";
    }
#endif
    float trace = 0.0;
    float *pCol = NULL;
    for (size_t c = 0; c < num_cols_; ++c) {
      pCol = Col(c);
      trace += pCol[c];
    }
    return trace;
  }

  // add by zhuozhu.zz
  inline float DotProduct(const xnnFloatRuntimeMatrix &V) {
#ifdef _DEBUG
    // dim check
    if (V.num_cols_ != 1 || V.num_rows_ != num_rows_
        || num_cols_ != 1) {
      IDEC_ERROR << "dimension mismatch [" << V.num_rows_ << " x " << V.num_cols_ <<
                 " .* " << num_rows_ << " x "
                 << num_cols_ << "]";
    }
#endif
    float sum = 0.0;
#ifdef USE_SSE
    dotprod_sse(V.Col(0), Col(0), &sum, num_rows_);
#else

    const float *p = Col(0);
    const float *pX = V.Col(0);
    size_t d;
    for (d = 0; d < num_rows_; d++) {
      sum += p[d] * pX[d];
    }
#endif
    return sum;
  }

  // by zhuozhu.zz
  inline void Covariance(const xnnFloatRuntimeMatrix &V, float scale) {
#ifdef _DEBUG
    // dim check
    if (V.num_cols_ != 1) {
      IDEC_ERROR << "can't calculate covariance of a matrix, which is " <<
                 V.num_rows_ << " x " << V.num_cols_ ;
    }
#endif

    if (num_cols_ != V.num_rows_ || num_rows_ != V.num_rows_) {
      Resize(V.num_rows_, V.num_rows_);
    }

    float cst = 0.0, *pX = V.Col(0), *p = NULL;
    for (size_t col = 0; col < num_cols_; ++col) {
      p = Col(col);
      cst = pX[col] * scale;

//#ifdef USE_SSE
//		  scaleadd_sse(p, pX, cst, num_rows_);
//#else
      for (size_t d = 0; d < num_rows_; d++)
        p[d] = cst * pX[d];
//#endif
    }
  }


  // block-by-block version of ScalePlusMatTMat, optimized for large X (of a lot of columns)
  // refer to matprod_mtm in ssematrix.h for implementation details
  inline void ScalePlusMatTMat(const xnnFloatRuntimeMatrix &Mt,
                               const xnnFloatRuntimeMatrix &V, const float scale) {
    // this = scale * this + W^\top X
#ifdef _DEBUG
    // dim check
    if (Mt.num_cols_ != num_rows_ || V.num_cols_ != num_cols_
        || Mt.num_rows_ != V.num_rows_) {
      IDEC_ERROR << "dimension mismatch [" << Mt.num_cols_ << " x " << Mt.num_rows_
                 << " * " << V.num_rows_ << " x " << V.num_cols_ << " = " << num_rows_ << " x "
                 << num_cols_ << "]";
    }
#endif

    const size_t cacheablerowsV = 512;
    const size_t cacheablecolsV = 16;
    // 512 * 16 -> 32 KB

    const size_t colstripewV = cacheablecolsV;
    const size_t rowstripehM = 128;
    const size_t dotprodstep = cacheablerowsV;
    // 128 * 512 -> 64 KB

    const size_t colstrideV = V.ColStride();

    // loop over col stripes of V
    for (size_t j0 = 0; j0 < V.NumCols(); j0 += colstripewV) {
      const size_t j1 = std::min(j0 + colstripewV, V.NumCols());
      // stripe of V is columns [j0,j1)

      // loop over row stripes of M
      for (size_t i0 = 0; i0 < Mt.NumCols(); i0 += rowstripehM) {
        const size_t i1 = std::min(i0 + rowstripehM, Mt.NumCols());

        // loop over sub-ranges of the dot product (full dot product will exceed the L1 cache)
#ifdef _MSC_VER
        __declspec(align(16)) float patchbuffer[rowstripehM *
                                                colstripewV];    // note: don't forget column rounding
#else
        __attribute__((aligned(16))) float patchbuffer[rowstripehM *
            colstripewV];    // note: don't forget column rounding
#endif
        // 128 * 16 -> 8 KB
        memset(patchbuffer, 0, rowstripehM * colstripewV * sizeof(float));

        for (size_t k0 = 0; k0 < V.NumRows(); k0 += dotprodstep) {
          const size_t k1 = std::min(k0 + dotprodstep, V.NumRows());
          //const bool first = k0 == 0;

          for (size_t i = i0; i < i1; ++i) {
            const size_t j14 = j1 & ~3;
            for (size_t j = j0; j < j14; j += 4) {  // grouped by 4
              const float *row = Mt.Col(i) + k0;    // of length k1-k0
              const float *cols4 = V.Col(j) +
                                   k0;    // of length k1-k0, stride = V.ColStride()
              float *patchij = patchbuffer + (j - j0)*rowstripehM + (i - i0);

              dotprod4(row, cols4, colstrideV, patchij, rowstripehM, k1 - k0);
            }
            for (size_t j = j14; j < j1; ++j) {
              dotprod(Mt.Col(i) + k0, V.Col(j) + k0,
                      patchbuffer + (j - j0)*rowstripehM + (i - i0), k1 - k0);
            }
          }
        }

        // asign patch
        for (size_t j = j0; j < j1; ++j) {
          //memcpy_s(Col(j)+i0, sizeof(float)*(i1-i0), patchbuffer+(j-j0)*rowstripehM, sizeof(float)*(i1-i0));
          scaleadd(Col(j) + i0, patchbuffer + (j - j0)*rowstripehM, scale, i1 - i0);
        }
      }
    }
  }

  // block-by-block version of ScalePlusMatTMat, optimized for large X (of a lot of columns)
  // refer to matprod_mtm in ssematrix.h for implementation details
  // this = this + Mt^T * V
  inline void PlusMatTMat(const xnnFloatRuntimeMatrix &Mt,
                          const xnnFloatRuntimeMatrix &V) {

#ifdef _DEBUG
    // dim check
    if (Mt.num_cols_ != num_rows_ || V.num_cols_ != num_cols_
        || Mt.num_rows_ != V.num_rows_) {
      IDEC_ERROR << "dimension mismatch [" << Mt.num_cols_ << " x " << Mt.num_rows_
                 << " * " << V.num_rows_ << " x " << V.num_cols_ << " = " << num_rows_ << " x "
                 << num_cols_ << "]";
    }
#endif

#ifdef USE_MKL
    // https://software.intel.com/en-us/node/520775#AE8380B9-CAC8-4C57-9AF3-2EAAC6ACFC1B
    return cblas_sgemm(CblasColMajor,
                       CblasTrans,             // do transpose on Mt
                       CblasNoTrans,           // keep V as it is
                       (int)Mt.NumCols(),            // number of rows of the matrix Mt^T, and number of rows of this
                       (int)V.NumCols(),             // number of columns of the matrix V, and number of columns of this
                       (int)V.NumRows(),             // number of columns of the matrix Mt^t, and row of the matrix V
                       1.0,
                       Mt.Col(0),
                       (int)Mt.col_stride_,
                       V.Col(0),
                       (int)V.col_stride_,
                       1.0,
                       this->data_,
                       (int)this->col_stride_);
#endif

    const size_t cacheablerowsV = 512;
    const size_t cacheablecolsV = 16;
    // 512 * 16 -> 32 KB

    const size_t colstripewV = cacheablecolsV;
    const size_t rowstripehM = 128;
    const size_t dotprodstep = cacheablerowsV;
    // 128 * 512 -> 64 KB

    const size_t colstrideV = V.ColStride();

    // loop over col stripes of V
    for (size_t j0 = 0; j0 < V.NumCols(); j0 += colstripewV) {
      const size_t j1 = std::min(j0 + colstripewV, V.NumCols());
      // stripe of V is columns [j0,j1)

      // loop over row stripes of M
      for (size_t i0 = 0; i0 < Mt.NumCols(); i0 += rowstripehM) {
        const size_t i1 = std::min(i0 + rowstripehM, Mt.NumCols());

        // loop over sub-ranges of the dot product (full dot product will exceed the L1 cache)
#ifdef _MSC_VER
        __declspec(align(16)) float patchbuffer[rowstripehM *
                                                colstripewV];    // note: don't forget column rounding
#else
        __attribute__((aligned(16))) float patchbuffer[rowstripehM *
            colstripewV];    // note: don't forget column rounding
#endif
        // 128 * 16 -> 8 KB
        memset(patchbuffer, 0, rowstripehM * colstripewV * sizeof(float));

        for (size_t k0 = 0; k0 < V.NumRows(); k0 += dotprodstep) {
          const size_t k1 = std::min(k0 + dotprodstep, V.NumRows());
          //const bool first = k0 == 0;

          for (size_t i = i0; i < i1; ++i) {
            const size_t j14 = j1 & ~3;
            for (size_t j = j0; j < j14; j += 4) {  // grouped by 4
              const float *row = Mt.Col(i) + k0;    // of length k1-k0
              const float *cols4 = V.Col(j) +
                                   k0;    // of length k1-k0, stride = V.ColStride()
              float *patchij = patchbuffer + (j - j0)*rowstripehM + (i - i0);

              dotprod4(row, cols4, colstrideV, patchij, rowstripehM, k1 - k0);
            }
            for (size_t j = j14; j < j1; ++j) {
              dotprod(Mt.Col(i) + k0, V.Col(j) + k0,
                      patchbuffer + (j - j0)*rowstripehM + (i - i0), k1 - k0);
            }
          }
        }

        // assign patch
        for (size_t j = j0; j < j1; ++j) {
          //memcpy_s(Col(j)+i0, sizeof(float)*(i1-i0), patchbuffer+(j-j0)*rowstripehM, sizeof(float)*(i1-i0));
          //scaleadd(scale, Col(j) + i0, patchbuffer + (j - j0)*rowstripehM, i1 - i0);
          add(Col(j) + i0, patchbuffer + (j - j0)*rowstripehM, i1 - i0);
        }
      }
    }
  }



#endif

  void PlusMatTMat(const xnnFloat16RuntimeMatrix &Mt,
                   const xnnFloat16RuntimeMatrix &V);
  void PlusMatTMat(const xnnFloat8RuntimeMatrix &Mt,
                   const xnnFloat8RuntimeMatrix &V);

  void PlusSmallMatTSmallMat(const xnnFloat16RuntimeMatrix &Mt,
                             const xnnFloat16RuntimeMatrix &V);
  void PlusSmallMatTSmallMat(const xnnFloat8RuntimeMatrix &Mt,
                             const xnnFloat8RuntimeMatrix &V);

  inline void Plusv(const xnnFloatRuntimeMatrix &X) {
    // u += this^\top * v
#ifdef _DEBUG
    // dim check
    if (X.NumRows() != num_rows_ || X.NumCols() != 1) {
      IDEC_ERROR << "dimension mismatch " << X.NumRows() << " vs. " << num_rows_ <<
                 ", " << X.NumCols() << " vs. 1";
    }
#endif

#if defined USE_AVX
#elif defined USE_SSE

    const float *vhead = X.Col(0);
    for (size_t col = 0; col < num_cols_; ++col) {
      float *u = Col(col);
      plus_sse(u, vhead, num_rows_);
    }
#else
    size_t d;

    const float *vhead = X.Col(0);
    for (size_t col = 0; col < num_cols_; ++col) {
      const float *v = vhead;
      float *u = Col(col);

      for (d = 0; d < num_rows_; d++) // 8-way loop unrolling
        u[d] += v[d];

    }
#endif
  }

  inline void Setv(const xnnFloatRuntimeMatrix &X) {
    // u = v * [1 1 1 ... 1]
#ifdef _DEBUG
    // dim check
    if (X.NumRows() != num_rows_
        || X.NumCols() != 1) { // || col_stride_ != X.ColStride())
      IDEC_ERROR << "dimension mismatch " << X.NumRows() << " vs. " << num_rows_ <<
                 ", " << X.NumCols() << " vs. 1, " << col_stride_ << " vs. " << X.ColStride();
    }
#endif
    for (size_t col = 0; col < num_cols_; ++col) {
#ifdef _MSC_VER
      memcpy_s(Col(col), num_rows_*sizeof(float), X.Col(0), num_rows_*sizeof(float));
#else
      memcpy(Col(col), X.Col(0), num_rows_*sizeof(float));

#endif
    }
  }

  inline void SetAll(const float v) {
    // set all elements to v
    for (size_t col = 0; col < num_cols_; ++col) {
      float *pCol = Col(col);
      for (size_t row = 0; row < num_rows_; ++row) {
        pCol[row] = v;
      }
    }
  }

  inline void ScalePlusvElemProdv(const float scale,
                                  const xnnFloatRuntimeMatrix &X) {
    // each column of this = this_column * scale + X_column (element-wise product) Y_column
#ifdef _DEBUG
    // dim check
    if (X.NumRows() != num_rows_) {
      IDEC_ERROR << "row dimension mismatch " << X.NumRows()  <<
                 " " << num_rows_;
    }
    if (X.NumCols() != num_cols_) {
      IDEC_ERROR << "col dimension mismatch " << X.NumCols() <<
                 " " << num_cols_;
    }
#endif

#ifdef USE_SSE
    float *p[4], *pX[4];
    size_t col;
    for (col = 0; col+4 < num_cols_; col+=4) {
      p[0] = Col(col);
      p[1] = Col(col+1);
      p[2] = Col(col+2);
      p[3] = Col(col+3);

      pX[0] = X.Col(col);
      pX[1] = X.Col(col+1);
      pX[2] = X.Col(col+2);
      pX[3] = X.Col(col+3);

      scaleadd_sse(p[0], pX[0], scale, num_rows_);
      scaleadd_sse(p[1], pX[1], scale, num_rows_);
      scaleadd_sse(p[2], pX[2], scale, num_rows_);
      scaleadd_sse(p[3], pX[3], scale, num_rows_);
    }

    for (; col < num_cols_; ++col) {
      p[0] = Col(col);
      pX[0] = X.Col(col);
      scaleadd_sse(p[0], pX[0], scale, num_rows_);
    }

#else

    float *p = NULL, *pX = NULL;
    for (size_t col = 0; col < num_cols_; ++col) {
      p = Col(col);
      pX = X.Col(col);

      size_t d;
      for (d = 0; d < num_rows_; d++)
        p[d] += scale * pX[d];
    }

#endif
  }

  inline void ScalePlusvElemProdv(const xnnFloatRuntimeMatrix &X,
                                  const xnnFloatRuntimeMatrix &Y, const float scale) {
    // each column of this = this_column * scale + X_column (element-wise product) Y_column
#ifdef _DEBUG
    // dim check
    if (X.NumRows() != Y.NumRows() || X.NumRows() != num_rows_) {
      IDEC_ERROR << "row dimension mismatch " << X.NumRows() << " " << Y.NumRows() <<
                 " " << num_rows_;
    }
    if (X.NumCols() != Y.NumCols() || X.NumCols() != num_cols_) {
      IDEC_ERROR << "col dimension mismatch " << X.NumCols() << " " << Y.NumCols() <<
                 " " << num_cols_;
    }
#endif

#ifdef USE_SSE
    for (size_t col = 0; col < num_cols_; ++col) {
      float *p = Col(col);
      const float *pX = X.Col(col);
      const float *pY = Y.Col(col);
      scale_plus_prod_sse(p, pX, pY, scale, num_rows_);
    }
#else
    for (size_t col = 0; col < num_cols_; ++col) {
      float *p = Col(col);
      const float *pX = X.Col(col);
      const float *pY = Y.Col(col);

      size_t d;
      for (d = 0; d < num_rows_; d++)
        p[d] += pX[d] * pY[d] * scale;

    }

#endif
  }

  inline void ScalePlusvElemProdv(const float scale,
                                  const xnnFloatRuntimeMatrix &X, const xnnFloatRuntimeMatrix &Y) {
    // each column of this = this_column * scale + X_column (element-wise product) Y_column
#ifdef _DEBUG
    // dim check
    if (X.NumRows() != Y.NumRows() || X.NumRows() != num_rows_) {
      IDEC_ERROR << "row dimension mismatch " << X.NumRows() << " " << Y.NumRows() <<
                 " " << num_rows_;
    }
    if (X.NumCols() != Y.NumCols() || X.NumCols() != num_cols_) {
      IDEC_ERROR << "col dimension mismatch " << X.NumCols() << " " << Y.NumCols() <<
                 " " << num_cols_;
    }
#endif

#ifdef USE_SSE
    for (size_t col = 0; col < num_cols_; ++col) {
      float *p = Col(col);
      const float *pX = X.Col(col);
      const float *pY = Y.Col(col);
      scale_plus_prod_sse(p, pX, pY, num_rows_, scale);
    }
#else
    for (size_t col = 0; col < num_cols_; ++col) {
      float *p = Col(col);
      const float *pX = X.Col(col);
      const float *pY = Y.Col(col);

      size_t d;
      for (d = 0; d < num_rows_; d++)
        p[d] = pX[d] * pY[d] + scale*p[d];

    }

#endif
  }

  // non-linear functions
  inline void HardSigmoid() {
    for (size_t col = 0; col < num_cols_; ++col) {
      float *pCol = Col(col);
      for (size_t row = 0; row < num_rows_; ++row) {
        pCol[row] = 0.2f * pCol[row] + 0.5f;
        pCol[row] = std::min<float>(1, std::max<float>(0, pCol[row]));
      }
    }
  }


  // non-linear functions
  inline void Sigmoid() {
#if defined(__INTEL_COMPILER) && defined(USE_AVX2)
    for (size_t col = 0; col < num_cols_; ++col) {
      float *pCol = Col(col);
      sigmod_avx2_svml(pCol, num_rows_);
    }
#elif defined(__INTEL_COMPILER) && defined(USE_SSE)
    for (size_t col = 0; col < num_cols_; ++col) {
      float *pCol = Col(col);
      sigmod_sse_svml(pCol, num_rows_);
    }
#elif USE_SSE
    for (size_t col = 0; col < num_cols_; ++col) {
      float *pCol = Col(col);
      sigmod_sse(pCol, num_rows_);
    }
#else
    float zero = 0.0f, one = 1.0f;
    float explimit = 88.722008f;


#ifdef _MSC_VER
    __declspec(align(16)) float expbuffer;
#else
    __attribute__((aligned(16))) float expbuffer;
#endif

    for (size_t col = 0; col < num_cols_; ++col) {
      float *pCol = Col(col);

      size_t row;
      for (row = 0; row < num_rows_; row++) {
        expbuffer = std::min<float>(-pCol[row], explimit);
        expbuffer = expf(expbuffer);
        pCol[row] = one / (one + expbuffer);
      }

    }
#endif
  }

  inline void ReLU() {
#ifdef USE_SSE
    for (size_t col = 0; col < num_cols_; ++col) {
      float *pCol = Col(col);
      relu_sse(pCol, num_rows_);
    }
#elif defined USE_NEON
    for (size_t col = 0; col < num_cols_; ++col) {
      float *pCol = Col(col);
      relu_neon(pCol, num_rows_);
    }
#else
    for (size_t col = 0; col < num_cols_; ++col) {
      float *pCol = Col(col);

      size_t row;
      for (row = 0; row < num_rows_; row++)
        pCol[row] = std::max<float>(pCol[row], 0);

    }
#endif
  }

  inline void Softmax() {
    for (size_t col = 0; col < num_cols_; ++col) {
      float *pCol = Col(col);

      float sum = kLogZeroFloat;
      size_t row;
      for (row = 0; row < num_rows_; ++row) {
        sum = LogAdd(sum, pCol[row]);
      }
      for (row = 0; row < num_rows_; ++row) {
        pCol[row] -= sum;
        pCol[row] = expf(pCol[row]);
      }
    }
  }


  inline void LogSoftmax() {
    for (size_t col = 0; col < num_cols_; ++col) {
      float *pCol = Col(col);

      float sum = kLogZeroFloat;
      size_t row;
      for (row = 0; row < num_rows_; ++row) {
        sum = LogAdd(sum, pCol[row]);
      }
      for (row = 0; row < num_rows_; ++row) {
        pCol[row] -= sum;
      }
    }
  }

  inline void Tanh() {
#ifdef USE_SSE
    for (size_t col = 0; col < num_cols_; ++col) {
      float *pCol = Col(col);
      tanh_sse(pCol, num_rows_);
    }
#else

    float zero = 0.0f, one = 1.0f;
    float explimit = 88.722008f;


#ifdef _MSC_VER
    __declspec(align(16)) float expbuffer;
#else
    __attribute__((aligned(16))) float expbuffer;
#endif

    for (size_t col = 0; col < num_cols_; ++col) {
      float *pCol = Col(col);

      size_t row;
      for (row = 0; row < num_rows_; row++) {
        expbuffer = std::min<float>(-2 * pCol[row], explimit);
        expbuffer = expf(expbuffer);
        pCol[row] = 2 * one / (one + expbuffer) - one;
      }

    }
#endif
  }

  inline void GroupPnorm2(const xnnFloatRuntimeMatrix &Mt,
                          const size_t groupsize) {
#ifdef _DEBUG
    // dim check
    if (Mt.NumRows() % groupsize != 0 || Mt.NumRows() / groupsize != num_rows_
        || Mt.NumCols() != num_cols_) {
      IDEC_ERROR << "dimension mismatch " << Mt.NumRows() << " vs. " << groupsize <<
                 ", " << Mt.NumRows() / groupsize << " vs. " << num_rows_ << ", " <<
                 Mt.NumCols() << " vs. " << num_cols_;
    }
#endif

#if defined USE_AVX
#elif defined USE_SSE
    for (size_t col = 0; col < num_cols_; ++col) {
      float *p = Mt.Col(col);
      float *result = Col(col);
      group_pnorm_sse(result, p, num_rows_, groupsize);
    }

#else

    for (size_t col = 0; col < num_cols_; ++col) {
      float *p = Mt.Col(col);
      size_t d = 0;
      for (size_t g = 0; g < num_rows_; ++g) {
        float sum = 0;
        size_t i;

        for (i = 0; i < groupsize; i++, d++)
          sum += p[d] * p[d];

        Col(col)[g] = std::sqrt(sum);
      }

    }
#endif
  }

  inline void L2norm(const xnnFloatRuntimeMatrix &Mt, const float floor = 0) {
#ifdef _DEBUG
    // dim check
    if (Mt.NumCols() != num_rows_) {
      IDEC_ERROR << "dimension mismatch " << Mt.NumCols() << " vs. " << num_rows_;
    }
    if (num_cols_ != 1) {
      IDEC_ERROR << "this is not a column vector";
    }
#endif
    float alpha = 1.0f / Mt.NumRows();

#if defined USE_AVX
#elif defined USE_SSE
    for (size_t row = 0; row < num_rows_; ++row) {
      float *p = Mt.Col(row);
      float *result = Col(0);
      l2norm_sse(result, p, Mt.NumRows(), floor, alpha);
    }

#else

    for (size_t row = 0; row < num_rows_; ++row) {
      float *p = Mt.Col(row);
      float sum = 0.0f;
      size_t d;

      for (d = 0; d < Mt.NumRows(); d++)
        sum += p[d] * p[d];

      Col(0)[row] = sum;
      Col(0)[row] *= alpha;
      Col(0)[row] = std::max<float>(floor, Col(0)[row]);
      Col(0)[row] = std::sqrt(Col(0)[row]);

    }
#endif
  }

  inline void Max(const xnnFloatRuntimeMatrix &Mt, const size_t groupsize) {
#ifdef _DEBUG
    // dim check
    if (Mt.NumRows() % groupsize != 0 || Mt.NumRows() / groupsize != num_rows_
        || Mt.NumCols() != num_cols_) {
      IDEC_ERROR << "dimension mismatch " << Mt.NumRows() << " vs. " << groupsize <<
                 ", " << Mt.NumRows() / groupsize << " vs. " << num_rows_ << ", " <<
                 Mt.NumCols() << " vs. " << num_cols_;
    }
#endif

#if defined USE_AVX
#elif defined USE_SSE
    for (size_t col = 0; col < num_cols_; ++col) {
      float *p = Mt.Col(col);
      float *result = Col(col);
      group_pnorm_sse(result, p, num_rows_, groupsize);
    }

#else

    for (size_t col = 0; col < num_cols_; ++col) {
      float *p = Mt.Col(col);
      size_t d = 0;
      for (size_t g = 0; g < num_rows_; ++g) {
        float sum = 0;
        size_t i;

        for (i = 0; i < groupsize; i++, d++)
          sum += p[d] * p[d];

        Col(col)[g] = std::sqrt(sum);
      }

    }
#endif
  }

  // [yanping.wyp] max by column
  inline void Max(const xnnFloatRuntimeMatrix &Mt) {
#ifdef _DEBUG
    if (Mt.NumCols() != num_rows_) {
      IDEC_ERROR << "dimension mismatch " << Mt.NumCols() << " vs " << num_rows_;
    }
    if (num_cols_ != 1) {
      IDEC_ERROR << "this is not a column vector";
    }
#endif

    for (size_t col = 0; col < Mt.NumCols(); ++col) {
      float *pCol = Mt.Col(col);
      float max_value = kLogZeroFloat;
      for (size_t row = 0; row < Mt.NumRows(); ++row) {
        if (max_value < pCol[row]) max_value = pCol[row];
      }
      Col(0)[col] = max_value;
    }
  }

  // [yanping.wyp] padding, new add [0...] column on the left & right
  inline void Padding(const xnnFloatRuntimeMatrix &Mt, const int left_col,
                      const int right_col) {
    if (left_col < 0 || right_col < 0) {
      IDEC_ERROR << "Padding, left_col & right_col must be positive";
    }
    if (Mt.NumCols() + left_col + right_col != num_cols_) {
      IDEC_ERROR << "dimension mismatch " << Mt.NumCols() + left_col + right_col <<
                 " vs " << num_cols_;
    }
    if (Mt.NumRows() != num_rows_) {
      IDEC_ERROR << "dimension mismatch " << Mt.NumRows() << " vs " << num_rows_;
    }

    for (size_t col = 0; col < static_cast<size_t>(left_col); ++col) {
      memset(Col(col), 0, sizeof(float)*num_rows_);
    }
    for (size_t col = 0; col < Mt.NumCols(); ++col) {
      memcpy(Col(col+left_col), Mt.Col(col), sizeof(float)*num_rows_);
    }
    for (size_t col = 0; col < static_cast<size_t>(right_col); ++col) {
      memset(Col(col+left_col+Mt.NumCols()), 0, sizeof(float)*num_rows_);
    }
  }

  // [yanping.wyp] 1d-convolution: Mt size(len, A), kernel size(len, B), return size(A-B+1, 1)
  // y[i] = Mt(len, i:i+B-1) * kernel(len, B), i = 0,1...A-B+1
  inline void Convolution1d(const xnnFloatRuntimeMatrix &Mt,
                            const xnnFloatRuntimeMatrix &kernel) {
    if (Mt.NumRows() != kernel.NumRows()) {
      IDEC_ERROR << "dimension mismatch " << Mt.NumRows() << " vs " <<
                 kernel.NumRows();
    }
    if (Mt.NumCols() - kernel.NumCols() + 1 != num_rows_ || num_cols_ != 1) {
      IDEC_ERROR << "dimension wrong " << num_rows_ << " : " << num_cols_;
    }

    for (size_t i = 0; i < Mt.NumCols() - kernel.NumCols() + 1; ++i) {
      float sum = 0;
      for (size_t j = 0; j < kernel.NumCols(); ++j) {
        dotprod(kernel.Col(j), Mt.Col(i+j), &sum, Mt.NumRows());
      }
      Col(0)[i] = sum;
    }
  }

  inline void OutputMatrix() const {
    using namespace std;
    cout << "Row " << num_rows_ << endl;
    cout << "Col " << num_cols_ << endl;
    for (size_t i = 0; i < num_rows_; ++i) {
      for (size_t j = 0; j < num_cols_; ++j) {
        cout << Col(j)[i] << ",";
      }
      cout << endl;
    }
  }

//        inline void CopyRows(const xnnFloatRuntimeMatrix &Mt, const std::vector<size_t> &Vt )
//        {
//#ifdef _DEBUG
//            // dim check
//            if (Vt.size() != Mt.NumRows() )
//            {
//                IDEC_ERROR << "dimension mismatch " << Vt.size() << " vs. " << Mt.NumRows();
//            }
//#endif
//
//
//            for (size_t col = 0; col < num_cols_; ++col)
//            {
//                T *pCol = Col(col);
//
//                size_t row;
//                for (row = 0; row < num_rows_; row++)
//                {
//                    pCol[row] = Mt.Col(col)[Vt[row]];
//                }
//
//            }
//        }

};

template<typename T> class Quantizer {
 public:
  float max_value_;    // maximum value (note: we really want a supremum; max value will be wrongly quantized)
  // [zhijie.yzj] rename "max" to "max_value" because "max" can be confusing with "std::max"
  float scale_;

 protected:
  //static const short short_max = 32767;
  T QMAX;


 public:
  Quantizer() : QMAX(std::numeric_limits<T>::max()) { max_value_ = 0; scale_ = 1.0f; }

  // constructor when range is known
  //Quantizer(float max_value) { max_value_ = max_value; }     // note: max(min) fails due to some macro definition

  // constructor that determines dynamically from a matrix
  // 'extrabits' reduce the quantization bits by this, such that we can safely add 2^extrabits values
  Quantizer(const xnnFloatRuntimeMatrix &mat, size_t extrabits) {
    setQuantizer(mat, extrabits);
  }


  void setQuantizer(const xnnFloatRuntimeMatrix &mat, size_t extrabits) {
    max_value_ = 0;
    //const float* data = &m.Col(0)[0];
    //size_t cols = mat.NumCols();
#ifdef USE_SSE
    for (size_t col = 0; col < mat.NumCols(); col++) {
      max_value_ = std::max(max_value_, max_abs_sse(mat.Col(col), mat.NumRows()));
    }
#elif defined USE_NEON
    for (size_t col = 0; col < mat.NumCols(); col++) {
      float tempmax = max_abs_neon(mat.Col(col), mat.NumRows());
      max_value_ = std::max(max_value_, tempmax);
    }
#else
    for (size_t col = 0; col < mat.NumCols(); col++) {
      for (size_t row = 0; row < mat.NumRows(); row++) {
        max_value_ = std::max(max_value_, mat.Col(col)[row]);
        max_value_ = std::max(max_value_, -mat.Col(col)[row]);
      }
    }
#endif
    max_value_ *= (1 <<
                   extrabits);  // reserve value range for adding up to 2^extrabits of these values together
    scale_ = max_value_ / (QMAX + 0.5f);
  }

  // quantize a value to 'qint' range
  void quantize(T *sv, const float *fv, size_t len) const {
    float invscale = 1.0f / scale_;
#if defined USE_NEON
    quantize_neon(fv, sv, len, invscale);
#elif defined (USE_SSE) && defined(USE_SSE41)
    quantize_sse(sv, fv, len, invscale, QMAX);
#else
    for (size_t i = 0; i < len; i++) {
      float v = round(fv[i] * invscale);
      if (v < -(QMAX + 1.0f)) {
        sv[i] = -(QMAX + 1);
      } else if (v > QMAX) {
        sv[i] = QMAX;
      } else {
        sv[i] = static_cast<T>(v);
      }
    }
#endif
  }



//
//        void unquantize(float* target, const float* origin, size_t len) const
//        {
//            for (size_t i = 0; i < len; i++)
//            {
//                target[i] = origin[i] / (short_max + 0.5f) * max;
//            }
//        }

  void Serialize(SerializeHelper &helper) {
    helper.Serialize(max_value_);
    helper.Serialize(scale_);
  }

  void Deserialize(SerializeHelper &helper) {
    helper.Deserialize(max_value_);
    helper.Deserialize(scale_);
  }
};

template<typename T> class Quantizer8bitNonSymmetric {
 public:
  float maxf_, minf_;
  float scale_;
  T QMAX;

 protected:

 public:

  Quantizer8bitNonSymmetric() : QMAX(std::numeric_limits<T>::max()) { maxf_ = 0; minf_ = 0;  scale_ = 1.0f; }

  Quantizer8bitNonSymmetric(const xnnFloatRuntimeMatrix &mat, size_t extrabits) {
    setQuantizerNonSymmetric(mat, extrabits);
  }


  void setQuantizerNonSymmetric(const xnnFloatRuntimeMatrix &mat,
                                size_t extrabits) {
    maxf_ = -FLT_MAX;
    minf_ = FLT_MAX;
    float max_tmp = -FLT_MAX, min_tmp = FLT_MAX;
#ifdef USE_SSE
    for (size_t col = 0; col < mat.NumCols(); col++) {

      max_min_sse(mat.Col(col), mat.NumRows(), max_tmp, min_tmp);
      maxf_ = std::max(maxf_, max_tmp);
      minf_ = std::min(minf_, min_tmp);
    }
#elif defined USE_NEON   //currently not usable, need to be modified
    for (size_t col = 0; col < mat.NumCols(); col++) {
      float tempmax = max_abs_neon(mat.Col(col), mat.NumRows());
      // max_value_ = std::max(max_value_, tempmax);
    }
#else

    for (size_t col = 0; col < mat.NumCols(); col++) {
      for (size_t row = 0; row < mat.NumRows(); row++) {
        maxf_ = std::max(maxf_, mat.Col(col)[row]);
        minf_ = std::min(minf_, mat.Col(col)[row]);
      }
    }
    //max_value_ = max_f - min_f;
#endif
    //max_value_ *= (1 << extrabits);  // reserve value range for adding up to 2^extrabits of these values together
    scale_ = ((maxf_ - minf_) * (1 << extrabits)) / (QMAX * 2 + 1.0f);
  }

  void quantizeNonSymmetric(T *sv, const float *fv, size_t len,
                            int &curr_col_sum) {
    float invscale = 1.0f / scale_;
#if defined USE_NEON
    quantize_neon(fv, sv, len, invscale);    //need to be modefied
#elif defined (USE_SSE) && defined(USE_SSE41)
    quantize_non_symmetric_sse(sv, fv, len, minf_, invscale, QMAX, curr_col_sum);
#else
    for (size_t i = 0; i < len; i++) {
      float v = round((fv[i] - minf_) * invscale - QMAX - 1.0f);
      if (v < -(QMAX + 1.0f)) {
        sv[i] = -(QMAX + 1.0f);
        curr_col_sum += (int)(- QMAX - 1.0f);
      } else if (v > QMAX) {
        sv[i] = QMAX;
        curr_col_sum += (int)QMAX;
      } else {
        curr_col_sum += (int)v;
        sv[i] = static_cast<T>(v);
      }
    }
    //curr_col_sum = static_cast<int>(curr_col_sum);
#endif
  }


  void Serialize(SerializeHelper &helper) {
    helper.Serialize(maxf_ - minf_);
    helper.Serialize(scale_);
  }

  void Deserialize(SerializeHelper &helper) {
    float tmp = maxf_ - minf_;
    helper.Deserialize(tmp);
    helper.Deserialize(scale_);
  }

};

template<typename T> class Quantizer8bitUnsignedNonSymmetric {
 public:
  float maxf_, minf_;
  float scale_;
  T QMAX;

 protected:

 public:

  Quantizer8bitUnsignedNonSymmetric() : QMAX(std::numeric_limits<T>::max()) { maxf_ = 0.0f; minf_ = 0.0f;  scale_ = 1.0f; }

  Quantizer8bitUnsignedNonSymmetric(const xnnFloatRuntimeMatrix &mat,
                                    size_t extrabits) {
    setQuantizerNonSymmetric(mat, extrabits);
  }


  void setQuantizerNonSymmetric(const xnnFloatRuntimeMatrix &mat,
                                size_t extrabits, float quantile_cut = 0.01f) {
    maxf_ = -FLT_MAX;
    minf_ = FLT_MAX;
    size_t total_size = (mat.NumRows()) * (mat.NumCols());
    float *mat_in_a_row;
    mat_in_a_row = new float[total_size];
    for (size_t i = 0; i < mat.NumCols(); ++i) {
      memcpy(mat_in_a_row + i * mat.NumRows(), &(mat.Col(i)[0]),
             mat.NumRows() * sizeof(float));
    }
#ifdef USE_SSE
    if (fabs(quantile_cut - 0.0f) < 0.00001)
      max_min_sse(mat_in_a_row, total_size, maxf_, minf_);
    else {
      std::nth_element(mat_in_a_row, mat_in_a_row + (int)(quantile_cut * total_size),
                       mat_in_a_row + total_size);
      minf_ = *(mat_in_a_row + (int)(quantile_cut * total_size));
      std::nth_element(mat_in_a_row,
                       mat_in_a_row + (int)((1.0f - quantile_cut) * total_size),
                       mat_in_a_row + total_size);
      maxf_ = *(mat_in_a_row + (int)((1.0f - quantile_cut) * total_size));
    }
    /*for (size_t col = 0; col < mat.NumCols(); col++)
    {

    max_min_sse(mat.Col(col), mat.NumRows(), max_tmp, min_tmp);
    maxf_ = std::max(maxf_, max_tmp);
    minf_ = std::min(minf_, min_tmp);
    }*/
#elif defined USE_NEON   //currently not usable, need to be modified
    for (size_t col = 0; col < mat.NumCols(); col++) {
      float tempmax = max_abs_neon(mat.Col(col), mat.NumRows());
      // max_value_ = std::max(max_value_, tempmax);
    }
#else
    if (fabs(quantile_cut - 0.0f) < 0.00001) {
      for (size_t col = 0; col < mat.NumCols(); col++) {
        for (size_t row = 0; row < mat.NumRows(); row++) {
          maxf_ = std::max(maxf_, mat.Col(col)[row]);
          minf_ = std::min(minf_, mat.Col(col)[row]);
        }
      }
    } else {
      std::nth_element(mat_in_a_row, mat_in_a_row + (int)(quantile_cut * total_size),
                       mat_in_a_row + total_size);
      minf_ = *(mat_in_a_row + (int)(quantile_cut * total_size));
      std::nth_element(mat_in_a_row,
                       mat_in_a_row + (int)((1.0f - quantile_cut) * total_size),
                       mat_in_a_row + total_size);
      maxf_ = *(mat_in_a_row + (int)((1.0f - quantile_cut) * total_size));
    }

    //max_value_ = max_f - min_f;
#endif
    //max_value_ *= (1 << extrabits);  // reserve value range for adding up to 2^extrabits of these values together
    scale_ = ((maxf_ - minf_) * (1 << extrabits)) / QMAX;
  }

  void quantizeNonSymmetric(T *sv, const float *fv, size_t len,
                            uint32_t &curr_col_sum) {
    float invscale = 1.0f / scale_;
// #if defined USE_NEON
//            quantize_neon(fv, sv, len, invscale);
// #elif defined (USE_SSE) && !defined(NOT_HAVE_SSE41)
#if defined (USE_SSE) && defined(USE_SSE41)
    quantize_non_symmetric_sse(sv, fv, len, minf_, invscale, QMAX, curr_col_sum);
#else
    for (size_t i = 0; i < len; i++) {
      float v = round((fv[i] - minf_) * invscale);
      if (v < 0) {
        sv[i] = (T)0.0f;
        //curr_col_sum += (int)(-QMAX - 1.0f);
      } else if (v > QMAX) {
        sv[i] = QMAX;
        curr_col_sum += (int)QMAX;
      } else {
        curr_col_sum += (uint32)v;
        sv[i] = static_cast<T>(v);
      }
    }
    //curr_col_sum = static_cast<int>(curr_col_sum);
#endif
  }


  void Serialize(SerializeHelper &helper) {
    helper.Serialize(maxf_ - minf_);
    helper.Serialize(scale_);
  }

  void Deserialize(SerializeHelper &helper) {
    float tmp = maxf_ - minf_;
    helper.Deserialize(tmp);
    helper.Deserialize(scale_);
  }

};


class xnnFloat16RuntimeMatrix : public xnnShortRuntimeMatrix {
 protected:
  using xnnRuntimeColumnMatrix<short>::col_stride_;

 public:
  Quantizer<short> q;

 public:

  xnnFloat16RuntimeMatrix(const xnnFloatRuntimeMatrix &mat,
                          int extrabits) : q(mat, extrabits), xnnShortRuntimeMatrix(mat.NumRows(),
                                mat.NumCols()) {
    //for (size_t i = 0; i < mf.NumCols(); i++)
    //{
    //    q.quantize(Col(i), mf.Col(i), mf.NumCols());
    //}
  }

  xnnFloat16RuntimeMatrix() : xnnShortRuntimeMatrix() {}

  void quantize(const xnnFloatRuntimeMatrix &mat, int extrabits = 0) {
    q.setQuantizer(mat, extrabits);
    Resize(mat.NumRows(), mat.NumCols());
    for (size_t col = 0; col < num_cols_; ++col) {
      q.quantize(Col(col), mat.Col(col), num_rows_);
    }
  }


  void Serialize(SerializeHelper &helper) {
    xnnRuntimeColumnMatrix<short>::Serialize(helper);
    q.Serialize(helper);
  }

  void Deserialize(SerializeHelper &helper) {
    xnnRuntimeColumnMatrix<short>::Deserialize(helper);
    q.Deserialize(helper);
  }
};

class xnnFloat8RuntimeMatrix : public xnnUnsignedCharRuntimeMatrix {
 protected:
  using xnnRuntimeColumnMatrix<uint8_t>::col_stride_;


 public:
  Quantizer8bitUnsignedNonSymmetric<uint8_t> q;
  std::vector<uint32> col_sum;     //col sum minus QMIN

 public:

  xnnFloat8RuntimeMatrix(const xnnFloatRuntimeMatrix &mat,
                         int extrabits) : q(mat, extrabits), xnnUnsignedCharRuntimeMatrix(mat.NumRows(),
                               mat.NumCols()) {}
  xnnFloat8RuntimeMatrix() : xnnUnsignedCharRuntimeMatrix() {}

  void quantize(const xnnFloatRuntimeMatrix &mat, int extrabits = 0,
                float quantile_cut = 0.0f) {
    q.setQuantizerNonSymmetric(mat, extrabits, quantile_cut);
    col_sum.resize(mat.NumCols());
    Resize(mat.NumRows(), mat.NumCols());
    for (size_t col = 0; col < num_cols_; ++col) {
      uint32_t curr_col_sum_ = 0;
      q.quantizeNonSymmetric(Col(col), mat.Col(col), num_rows_, curr_col_sum_);
      col_sum[col] = curr_col_sum_;
    }
  }

  /*
  void Serialize(SerializeHelper &helper)
  {
      xnnRuntimeColumnMatrix<int8_t>::Serialize(helper);
      q.Serialize(helper);
  }

  void Deserialize(SerializeHelper &helper)
  {
      xnnRuntimeColumnMatrix<int8_t>::Deserialize(helper);
      q.Deserialize(helper);
  }
  */
};


template<class ParentColumnMatrix> class xnnRuntimeColumnMatrixView : public
  ParentColumnMatrix {
 protected:
  using ParentColumnMatrix::num_rows_;
  using ParentColumnMatrix::num_cols_;
  using ParentColumnMatrix::col_stride_;
  using ParentColumnMatrix::data_;
  using ParentColumnMatrix::allocated_bytes_;
  const ParentColumnMatrix &mat_;

 public:

  xnnRuntimeColumnMatrixView(const ParentColumnMatrix &mat) : mat_(mat) {
#ifdef _MSC_VER
    memcpy_s((ParentColumnMatrix *)this, sizeof(ParentColumnMatrix), &mat,
             sizeof(ParentColumnMatrix));
#else
    memcpy((ParentColumnMatrix *)this, &mat, sizeof(ParentColumnMatrix));
#endif
    allocated_bytes_ = 0;
  }

  xnnRuntimeColumnMatrixView(const xnnRuntimeColumnMatrixView &mat) : mat_(mat) {
    //IDEC_ERROR << "error";
#ifdef _MSC_VER
    memcpy_s(this, sizeof(xnnRuntimeColumnMatrixView), &mat,
             sizeof(xnnRuntimeColumnMatrixView));
#else
    memcpy(this, &mat, sizeof(xnnRuntimeColumnMatrixView));
#endif
    allocated_bytes_ = 0;
  }

  const ParentColumnMatrix *getMat() { return &mat_; }

  void ColView(size_t start_col, size_t num_cols) {
    start_col + num_cols <= mat_.NumCols()
    || IDEC_ERROR << "requested column out of range";

    data_ = mat_.Col(start_col);
    num_cols_ = num_cols;
  }

  void RowView(size_t start_row, size_t num_rows) {
    start_row + num_rows <= mat_.NumRows()
    || IDEC_ERROR << "requested row out of range";

    data_ = mat_.Col(0) + start_row;
    num_rows_ = num_rows;
  }

  void ColRowView(size_t start_col, size_t num_cols, size_t start_row,
                  size_t num_rows) {
    start_col + num_cols <= mat_.NumCols()
    || IDEC_ERROR << "requested column out of range";
    start_row + num_rows <= mat_.NumRows()
    || IDEC_ERROR << "requested row out of range";

    data_ = mat_.Col(start_col) + start_row;
    num_cols_ = num_cols;
    num_rows_ = num_rows;
  }

  ~xnnRuntimeColumnMatrixView() {
    data_ = NULL; // so it will not be deleted
  }
};

class xnnFloatRuntimeMatrixCircularBuffer {
 protected:
  xnnFloatRuntimeMatrix buff_;
  size_t head_;
  size_t tail_;
  size_t count_;

 public:

  xnnFloatRuntimeMatrixCircularBuffer() : head_((size_t)-1), tail_((size_t)-1),
    count_(0) {
  }

  xnnFloatRuntimeMatrixCircularBuffer(size_t num_rows,
                                      size_t num_cols) : head_((size_t)-1), tail_((size_t)-1), count_(0) {
    Reserve(num_rows, num_cols);
  }

  size_t NumRows() { return buff_.NumRows(); }
  size_t NumCols() { return count_; }
  size_t NumEmpty() { return buff_.NumCols() - count_; }
  bool Empty() { return count_ == 0; }

  bool PushbackOneColumn(const float *data, size_t dim) {
    if (count_ == buff_.NumCols() || dim != buff_.NumRows())
      return(false);


    if (count_ == 0) {
      // empty buffer, push from beginning
      head_ = tail_ = 0;
    }

#ifdef _MSC_VER
    memcpy_s(buff_.Col(tail_), sizeof(float) * dim, data, sizeof(float) * dim);
#else
    memcpy(buff_.Col(tail_), data, sizeof(float) * dim);
#endif
    tail_ = (tail_ + 1) % buff_.NumCols();
    ++count_;

    return(true);
  }

  void PopfrontOneColumn() {
    if (count_ == 0)
      return;
    head_ = (head_ + 1) % buff_.NumCols();
    --count_;
  }

  void Reserve(size_t num_rows, size_t num_cols) {
    if (count_ == 0) {
      // empty buffer
      buff_.Resize(num_rows, num_cols);
    } else {
      if (num_rows != buff_.NumRows())
        IDEC_ERROR << "#rows are mismatch, " << num_rows << " vs. " << buff_.NumRows();

      if (num_cols > buff_.NumCols()) {
        // need to increase capacity
        if (tail_ > head_) {
          // easy case
          buff_.Resize(num_rows, num_cols);
        } else {
          // more difficult case
          size_t old_num_cols = buff_.NumCols();
          size_t inc_num_cols = std::max(tail_, num_cols - old_num_cols);
          buff_.Resize(num_rows, old_num_cols + inc_num_cols);
#ifdef _MSC_VER
          memcpy_s(buff_.Col(old_num_cols), tail_ * buff_.ColStride() * sizeof(float),
                   buff_.Col(0), tail_ * buff_.ColStride() * sizeof(float));
#else
          memcpy(buff_.Col(old_num_cols), buff_.Col(0),
                 tail_ * buff_.ColStride() * sizeof(float));
#endif
          tail_ = (tail_ + old_num_cols) % buff_.NumCols();
        }
      }
    }
  }

  void Clear() {
    head_ = tail_ = (size_t)-1;
    count_ = 0;
  }

  //float *GetHeadColumn() { return buff_.Col(head_); }

  float *Col(size_t col) {
    if (col >= count_)
      return(NULL);

    return buff_.Col((head_ + col) % buff_.NumCols());
  }

  ///////////////// add by zhuozhu.zz ////////////////
  const float *Col(size_t col) const {
    if (col >= count_)
      return(NULL);

    return buff_.Col((head_ + col) % buff_.NumCols());
  }

  size_t GetNFrames(size_t N, xnnFloatRuntimeMatrix &ret) {
    size_t n = std::min(N, count_);

    ret.Resize(buff_.NumRows(), n);

    for (size_t i = 0; i < n; ++i) {
#ifdef _MSC_VER
      memcpy_s(ret.Col(i), buff_.NumRows()*sizeof(float), Col(i),
               buff_.NumRows()*sizeof(float));
#else
      memcpy(ret.Col(i), Col(i), buff_.NumRows()*sizeof(float));
#endif
    }

    return(n);
  }

  //////////////////////zhuozhu.zz /////////////////////
  size_t GetPartFrames(size_t begin_frm, size_t end_frm,
                       xnnFloatRuntimeMatrix &ret) const {

    idec::IDEC_ASSERT(end_frm <= count_);

    ret.Resize(buff_.NumRows(), end_frm - begin_frm);

    for (size_t i = begin_frm, idx = 0; i < end_frm; ++i, ++idx) {
#ifdef _MSC_VER
      memcpy_s(ret.Col(idx), buff_.NumRows() * sizeof(float), Col(i),
               buff_.NumRows() * sizeof(float));
#else
      memcpy(ret.Col(idx), Col(i), buff_.NumRows() * sizeof(float));
#endif
    }

    return (end_frm - begin_frm);
  }

  size_t GetPartFrames(const std::vector<int> &frames_index,
                       xnnFloatRuntimeMatrix &ret) const {
    if (frames_index.empty()) {
      return 0;
    }

    int N = frames_index.size();
    idec::IDEC_ASSERT(frames_index[N-1] < count_);

    ret.Resize(buff_.NumRows(), N);
    for (size_t i = 0; i < N; ++i) {
#ifdef _MSC_VER
      memcpy_s(ret.Col(i), buff_.NumRows() * sizeof(float), Col(frames_index[i]),
               buff_.NumRows() * sizeof(float));
#else
      memcpy(ret.Col(i), Col(frames_index[i]), buff_.NumRows() * sizeof(float));
#endif
    }

    return N;
  }
  //zhuozhu.zz

  size_t PeekNFrames(size_t N, xnnFloatRuntimeMatrix &ret) {
    size_t n = GetNFrames(N, ret);

    return(n);
  }

  size_t PopNFrames(size_t N, xnnFloatRuntimeMatrix &ret) {
    size_t n = PeekNFrames(N, ret);

    for (size_t i = 0; i < n; ++i) {
      PopfrontOneColumn();
    }

    return(n);
  }

  //////////////////////zhuozhu.zz ////////////////////
  size_t PeekPartFrames(size_t begin_frm, size_t end_frm,
                        xnnFloatRuntimeMatrix &ret) const {
    size_t n = GetPartFrames(begin_frm, end_frm, ret);
    return(n);
  }

  size_t PeekPartFrames(const std::vector<int> &frames_index,
                        xnnFloatRuntimeMatrix &ret) const {
    size_t n = GetPartFrames(frames_index, ret);
    return(n);
  }
  //zhuozhu.zz

  size_t PopNFrames(size_t N) {
    size_t n = std::min(N, NumCols());

    for (size_t i = 0; i < n; ++i) {
      PopfrontOneColumn();
    }

    return(n);
  }
};

};

#endif
