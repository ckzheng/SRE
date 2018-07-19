#include "plda.h"

double Plda::GetNormalizationFactor(const DoubleVector &transformed_ivector,
                                    int num_examples) const {
  idec::IDEC_ASSERT(num_examples > 0);
  DoubleVector transformed_ivector_sq(transformed_ivector);
  transformed_ivector_sq.ApplyPow(2.0);
  DoubleVector inv_covar(psi_);
  inv_covar += (1.0 / num_examples);
  inv_covar.InvertElements();
  double dot_prod = inv_covar* transformed_ivector_sq;
  return sqrt(Dim() / dot_prod);
}

double Plda::TransformIvector(const DoubleVector &ivector, int num_examples,
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

double Plda::LogLikelihoodRatio(const DoubleVector &train_ivector,
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
