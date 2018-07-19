#ifndef _IVECTOR_TRANSFORM_H_
#define _IVECTOR_TRANSFORM_H_
#include "new_vector.h"
#include "new_matrix.h"

class IvectorTransform {
 public:
  IvectorTransform() {

  }

  ~IvectorTransform() {

  }

  static void LengthNorm(DoubleVector &ivector) {
    double norm = ivector.Norm(2.0);
    double ratio = norm / sqrt(ivector.Size());
    // than it would be, in
    // expectation, if normally
    bool scaleup = true;
    bool normalize = true;
    if (!scaleup) {
      ratio = norm;
    }

    //idec::IDEC_INFO << "Ratio is " << ratio;
    if (ratio == 0.0) {
      idec::IDEC_WARN << "Zero iVector";
      return;
    }

    if (normalize) {
      ivector.Scale(1.0 / ratio);
    }
  }

  static void LdaTransform(const DoubleMatrix &transform, DoubleVector &ivector,
                           DoubleVector &transformed_ivector) {
    //LengthNorm(ivector);
    const DoubleMatrix &linear_term = transform.Crop(0, 0, transform.Rows(),
                                      transform.Cols() - 1);
    const DoubleVector &constant_term = transform.Colv(transform.Cols() - 1);
    if (ivector.Size() == transform.Cols()) {
      transformed_ivector = transform * ivector;
    } else {
      idec::IDEC_ASSERT(ivector.Size() == transform.Cols() - 1);
      transformed_ivector = constant_term + linear_term * ivector;
    }
  }
};
#endif