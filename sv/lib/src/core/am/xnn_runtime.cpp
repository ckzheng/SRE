#include "xnn_runtime.h"

namespace idec {

void xnnFloatRuntimeMatrix::PlusMatTMat(const xnnFloat16RuntimeMatrix &Mt,
                                        const xnnFloat16RuntimeMatrix &V) {
  const size_t cacheablerowsV = 4096/*512*/; //magic number ???
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
#ifdef USE_AVX2
      __declspec(align(32)) int patchbuffer[rowstripehM *
                                            colstripewV];    // note: don't forget column rounding
#else
      __declspec(align(16)) int patchbuffer[rowstripehM *
                                            colstripewV];    // note: don't forget column rounding
#endif
#else
#ifdef USE_AVX2
      __attribute__((aligned(32))) int patchbuffer[rowstripehM *
          colstripewV];    // note: don't forget column rounding
#else
      __attribute__((aligned(16))) int patchbuffer[rowstripehM *
          colstripewV];    // note: don't forget column rounding
#endif
#endif
      // 128 * 16 -> 8 KB
      memset(patchbuffer, 0, rowstripehM * colstripewV * sizeof(int));

      for (size_t k0 = 0; k0 < V.NumRows(); k0 += dotprodstep) {
        const size_t k1 = std::min(k0 + dotprodstep, V.NumRows());
        //const bool first = k0 == 0;

        for (size_t i = i0; i < i1; ++i) {

          const size_t j14 = j1 & ~3;
          for (size_t j = j0; j < j14; j += 4) {  // grouped by 8
            const short *row = Mt.Col(i) + k0;	// of length k1-k0
            const short *cols4 = V.Col(j) + k0;	// of length k1-k0, stride = V.ColStride()
            int *patchij = patchbuffer + (j - j0)*rowstripehM + (i - i0);

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
        add(Col(j) + i0, patchbuffer + (j - j0)*rowstripehM, i1 - i0,
            Mt.q.scale_ * V.q.scale_);
      }
    }
  }

}
void xnnFloatRuntimeMatrix::PlusMatTMat(const xnnFloat8RuntimeMatrix &Mt,
                                        const xnnFloat8RuntimeMatrix &V) {
  const size_t cacheablerowsV = 4096/*512*/; //magic number ???
  const size_t cacheablecolsV = 16;
  // 512 * 16 -> 32 KB

  const size_t colstripewV = cacheablecolsV;
  const size_t rowstripehM = 128;
  const size_t dotprodstep = cacheablerowsV;
  // 128 * 512 -> 64 KB

  const size_t colstrideV = V.ColStride();

  float MtMinDividedScale = Mt.q.minf_ / Mt.q.scale_;
  float VMinDividedScale = V.q.minf_ / V.q.scale_;

  // loop over col stripes of V
  for (size_t j0 = 0; j0 < V.NumCols(); j0 += colstripewV) {
    const size_t j1 = std::min(j0 + colstripewV, V.NumCols());
    // stripe of V is columns [j0,j1)

    // loop over row stripes of M
    for (size_t i0 = 0; i0 < Mt.NumCols(); i0 += rowstripehM) {
      const size_t i1 = std::min(i0 + rowstripehM, Mt.NumCols());

      // loop over sub-ranges of the dot product (full dot product will exceed the L1 cache)
#ifdef _MSC_VER
#ifdef USE_AVX2
      __declspec(align(32)) int patchbuffer[rowstripehM *
                                            colstripewV];    // note: don't forget column rounding
#else
      __declspec(align(16)) int patchbuffer[rowstripehM *
                                            colstripewV];    // note: don't forget column rounding
#endif
#else
#ifdef USE_AVX2
      __attribute__((aligned(32))) int patchbuffer[rowstripehM *
          colstripewV];    // note: don't forget column rounding
#else
      __attribute__((aligned(16))) int patchbuffer[rowstripehM *
          colstripewV];    // note: don't forget column rounding
#endif
#endif
      // 128 * 16 -> 8 KB
      memset(patchbuffer, 0, rowstripehM * colstripewV * sizeof(int));

      for (size_t k0 = 0; k0 < V.NumRows(); k0 += dotprodstep) {
        const size_t k1 = std::min(k0 + dotprodstep, V.NumRows());
        //const bool first = k0 == 0;

        for (size_t i = i0; i < i1; ++i) {

          const size_t j14 = j1 & ~3;
          for (size_t j = j0; j < j14; j += 4) {  // grouped by 8
            const uint8_t *row = Mt.Col(i) + k0;	// of length k1-k0
            const uint8_t *cols4 = V.Col(j) +
                                   k0;	// of length k1-k0, stride = V.ColStride()
            int *patchij = patchbuffer + (j - j0)*rowstripehM + (i - i0);

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
        add(Col(j) + i0, patchbuffer + (j - j0)*rowstripehM, i1 - i0,
            Mt.q.scale_ * V.q.scale_);
      }
    }
  }
  float fac1 = Mt.q.scale_ * V.q.minf_, fac2 = V.q.scale_ * Mt.q.minf_,
        fac3 = Mt.NumRows() * V.q.minf_ * Mt.q.minf_;
  for (size_t i = 0; i < V.NumCols(); ++i) {
    for (size_t j = 0; j < Mt.NumCols(); ++j) {
      Col(i)[j] += Mt.col_sum[j] * fac1 + V.col_sum[i] * fac2 + fac3;
    }
  }

}


void xnnFloatRuntimeMatrix::PlusSmallMatTSmallMat(const xnnFloat16RuntimeMatrix
    &Mt, const xnnFloat16RuntimeMatrix &V) {
  for (size_t wcol = 0; wcol < Mt.NumCols(); ++wcol) {
    for (size_t vcol = 0; vcol < V.NumCols(); ++vcol) {
      int sum = 0;
      dotprod(Mt.Col(wcol), V.Col(vcol), &sum, Mt.NumRows());
      Col(vcol)[wcol] += sum * Mt.q.scale_ * V.q.scale_;
    }
  }
}
void xnnFloatRuntimeMatrix::PlusSmallMatTSmallMat(const xnnFloat8RuntimeMatrix
    &Mt, const xnnFloat8RuntimeMatrix &V) {
  float MtMinDividedScale = Mt.q.minf_ / Mt.q.scale_;
  float VMinDividedScale = V.q.minf_ / V.q.scale_;
  for (size_t wcol = 0; wcol < Mt.NumCols(); ++wcol) {
    for (size_t vcol = 0; vcol < V.NumCols(); ++vcol) {
      int sum = 0;
      dotprod(Mt.Col(wcol), V.Col(vcol), &sum, Mt.NumRows());
      Col(vcol)[wcol] += sum * Mt.q.scale_ * V.q.scale_;
    }
  }
  float fac1 = Mt.q.scale_ * V.q.minf_, fac2 = V.q.scale_ * Mt.q.minf_,
        fac3 = Mt.NumRows() * V.q.minf_ * Mt.q.minf_;
  for (size_t i = 0; i < V.NumCols(); ++i) {
    for (size_t j = 0; j < Mt.NumCols(); ++j) {
      Col(i)[j] += Mt.col_sum[j] * fac1 + V.col_sum[i] * fac2 + fac3;
    }
  }
}

}
