#ifndef _PLDA_H_
#define _PLDA_H_
#include "new_vector.h"
#include "new_matrix.h"
#include "base/log_message.h"
#include "resource_manager.h"

class Plda {
 public:
  Plda(const PldaOptions &config,
       const PldaResource &plda_res) :plda_opt_(config), mean_(plda_res.mean),
    transform_(plda_res.transform), psi_(plda_res.psi),
    offset_(plda_res.offset) {}
  ~Plda() {}

  int Dim() const {
    return mean_.Size();
  }

  double GetNormalizationFactor(const DoubleVector &transformed_ivector,
                                int num_examples) const {
    idec::IDEC_ASSERT(num_examples > 0);
    // Work out the normalization factor.  The covariance for an average over
    // "num_examples" training iVectors equals \Psi + I/num_examples.
    DoubleVector transformed_ivector_sq(transformed_ivector);
    transformed_ivector_sq.ApplyPow(2.0);
    DoubleVector inv_covar(psi_);
    inv_covar += (1.0 / num_examples);
    inv_covar.InvertElements();
    double dot_prod = inv_covar* transformed_ivector_sq;
    return sqrt(Dim() / dot_prod);
  }

  double TransformIvector(const DoubleVector &ivector, int num_examples,
                          DoubleVector &transformed_ivector) const {
    idec::IDEC_ASSERT(ivector.Size() == Dim());
    double normalization_factor;
    transformed_ivector = offset_;
    transformed_ivector += transform_ * ivector;
    if (plda_opt_.simple_length_norm) {
      normalization_factor = sqrt(transformed_ivector.Size()) /
                             transformed_ivector.Norm(2.0);
    } else {
      normalization_factor = GetNormalizationFactor(transformed_ivector,
                             num_examples);
    }

    if (plda_opt_.normalize_length) {
      transformed_ivector.Scale(normalization_factor);
    }

    return normalization_factor;
  }

  double LogLikelihoodRatio(const DoubleVector &train_ivector,
                            int n, const DoubleVector &test_ivector)  const {
    int dim = Dim();
    double loglike_given_class, loglike_without_class;
    const double M_LOG_2PI = 1.8378770664093454835606594728112;
    DoubleVector transformed_train_ivector, transformed_test_ivector;
    TransformIvector(train_ivector, n, transformed_train_ivector);
    TransformIvector(test_ivector, 1, transformed_test_ivector);
    {
      DoubleVector mean(dim);
      DoubleVector variance(dim);
      for (int i = 0; i < dim; i++) {
        mean(i) = n * psi_(i) / (n * psi_(i) + 1.0) * transformed_train_ivector(i);
        variance(i) = 1.0 + psi_(i) / (n * psi_(i) + 1.0);
      }
      double logdet = variance.SumLog();
      DoubleVector sqdiff(transformed_test_ivector);
      sqdiff -= mean;
      sqdiff.ApplyPow(2.0);
      variance.InvertElements();
      loglike_given_class = -0.5 * (logdet + M_LOG_2PI * dim + sqdiff * variance);
    }

    {
      // work out loglike_without_class.  Here the mean is zero and the variance
      // is I + \Psi.
      DoubleVector sqdiff(transformed_test_ivector);
      sqdiff.ApplyPow(2.0);
      DoubleVector variance(psi_);
      variance += 1.0;
      double logdet = variance.SumLog();
      variance.InvertElements();
      loglike_without_class = -0.5 * (logdet + M_LOG_2PI * dim + sqdiff * variance);
    }
    double loglike_ratio = loglike_given_class - loglike_without_class;
    return loglike_ratio;
  }

 private:
  // mean of samples in original space.
  const DoubleVector &mean_;
  // of dimension Dim() by Dim();
  const DoubleMatrix &transform_;
  // this transform makes within-class covar unit
  // and diagonalizes the between-class covar.
  const DoubleVector &psi_; // of dimension Dim().  The between-class
  // (diagonal) covariance elements, in decreasing order.
  // derived variable: -1.0 * transform_ * mean_.
  const DoubleVector &offset_;
  const PldaOptions &plda_opt_;
};

#endif // !_PLDA_H_
