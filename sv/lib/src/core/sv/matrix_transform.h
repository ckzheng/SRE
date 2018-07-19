#ifndef _MATRIX_TRANSFORM_H_
#define _MATRIX_TRANSFORM_H_
#include "new_vector.h"
#include "new_matrix.h"
#include "am/xnn_runtime.h"

class MatrixTransform {
 public:
  static void EigenMatrix2XnnMatrix(const DoubleMatrix &mat,
                                    idec::xnnFloatRuntimeMatrix &xmat) {
    xmat.Resize(mat.Rows(), mat.Cols());
    unsigned int nRows = mat.Rows();
    unsigned int nCols = mat.Cols();
    float *pCol = NULL;
    for (int c = 0; c < nCols; ++c) {
      pCol = xmat.Col(c);
      for (int r = 0; r < nRows; ++r) {
        pCol[r] = mat(r, c);
      }
    }
  }

  static void EigenVector2XnnMatrix(const DoubleVector &mat,
                                    idec::xnnFloatRuntimeMatrix &xmat) {
    xmat.Resize(mat.Rows(), 1);
    unsigned int nRows = mat.Rows();
    unsigned int nCols = mat.Cols();
    float *pCol = NULL;
    for (int c = 0; c < nCols; ++c) {
      pCol = xmat.Col(c);
      for (int r = 0; r < nRows; ++r) {
        pCol[r] = mat(r);
      }
    }
  }

 private:

};

#endif
