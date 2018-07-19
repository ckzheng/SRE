#include "ivector_estimator.h"
#include "base/time_utils.h"

void IvectorEstimator::InvertWithFlooring(DoubleMatrix &inverse_var,
    DoubleMatrix &var) const {
  DoubleMatrix dbl_var(inverse_var);
  int dim = inverse_var.Rows();
  DoubleVector s(dim);
  DoubleMatrix P(dim, dim);
  // Solve the symmetric eigenvalue problem, inverse_var = P diag(s) P^T.
  inverse_var.ComputeEigenProblem(P, s, dim);
  s.ApplyFloor(1.0);
  s.InvertElements();
  DoubleMatrix s_diag(dim, dim);
  s_diag.SetAllValues(0.0);
  for (int i = 0; i < s.Size(); ++i) {
    s_diag(i, i) = s(i);
  }
  var = P * s_diag;
}

double IvectorEstimator::GetLogDetNoFailure(const DoubleMatrix &var) {
  try {
    return var.LogDet();
  } catch (...) {
    DoubleVector s(var.Rows());
    DoubleMatrix P(var.Rows(), var.Rows());
    // Solve the symmetric eigenvalue problem, inverse_var = P diag(s) P^T.
    var.ComputeEigenProblem(P, s, var.Rows());
    s.ApplyFloor(1.0e-20);
    return s.Sum();
  }
}

void IvectorEstimator::GetIvectorDistMean(const UtteranceStats &utt_stats,
    DoubleVector &linear,
    DoubleMatrix &quadratic) const {
  const DoubleVector &utt_gamma = utt_stats.Gamma();
  const DoubleMatrix &utt_X = utt_stats.X();
  int I = NumGauss();
  for (int i = 0; i < I; i++) {
    double gamma = utt_gamma(i);
    if (gamma != 0.0) {
      const DoubleVector &x = utt_X.Rowv(i);
      //linear += Sigma_inv_M_[i].Transpose() * x;
      linear += Sigma_inv_M_trans_[i] * x;
    }
    quadratic += U_[i] * gamma;
  }
}

void IvectorEstimator::GetIvectorDistPrior(const UtteranceStats &utt_stats,
    DoubleVector &linear,
    DoubleMatrix &quadratic) const {
  linear(0) += prior_offset_;
  DoubleVector v(quadratic.Cols());
  v.SetAllValues(1.0);
  quadratic.DiagElementsAdd(v);
}

void IvectorEstimator::GetIvectorDistWeight(const UtteranceStats &utt_stats,
    const DoubleVector &mean, DoubleVector &linear,
    DoubleMatrix &quadratic) const {
  if (!IvectorDependentWeights()) {
    return;
  }

  DoubleVector logw_unnorm = w_ * mean;
  DoubleVector w(logw_unnorm);
  const DoubleVector &utt_gamma = utt_stats.Gamma();
  const DoubleMatrix &utt_X = utt_stats.X();
  w.ApplySoftMax();
  DoubleVector linear_coeff(NumGauss());
  DoubleVector quadratic_coeff(NumGauss());
  //double gamma = utt_stats.gamma_.Sum();
  double gamma = 0.0;
  for (int i = 0; i < utt_gamma.Size(); ++i) {
    gamma += utt_gamma(i);
  }

  for (int i = 0; i < NumGauss(); i++) {
    double gamma_i = utt_gamma(i);
    double max_term = std::max(gamma_i, gamma * w(i));
    linear_coeff(i) = gamma_i - gamma * w(i) + max_term * logw_unnorm(i);
    quadratic_coeff(i) = max_term;
  }

  DoubleMatrix quadratic_coeff_matrix(quadratic_coeff.Size(),
                                      quadratic_coeff.Size());
  quadratic_coeff_matrix.SetAllValues(0.0);
  for (int i = 0; i < quadratic_coeff.Size(); ++i) {
    quadratic_coeff_matrix(i, i) = quadratic_coeff(i);
  }
  quadratic += w_ * quadratic_coeff_matrix;
}

void IvectorEstimator::GetIvectorDistribution(const UtteranceStats &utt_stats,
    DoubleVector &mean, DoubleMatrix *var) {
  //double time_start, time_end;
  if (!IvectorDependentWeights()) {
    DoubleVector linear(IvectorDim());
    DoubleMatrix quadratic(IvectorDim(), IvectorDim());
    //time_start = idec::TimeUtils::GetTimeMilliseconds();
    GetIvectorDistMean(utt_stats, linear, quadratic);
    //time_end = idec::TimeUtils::GetTimeMilliseconds();
    //cout << time_end - time_start << " ms." << endl;
    //time_start = idec::TimeUtils::GetTimeMilliseconds();
    GetIvectorDistPrior(utt_stats, linear, quadratic);
    //time_end = idec::TimeUtils::GetTimeMilliseconds();
    //cout << time_end - time_start << " ms." << endl;
    if (var != NULL) {
      (*var) = quadratic;
      var->Invert();
      mean = (*var) * linear;
    } else {
      quadratic.Invert();
      mean = quadratic*linear;
    }
  } else {
    DoubleVector linear(IvectorDim());
    DoubleMatrix quadratic(IvectorDim());
    GetIvectorDistMean(utt_stats, linear, quadratic);
    GetIvectorDistPrior(utt_stats, linear, quadratic);

    DoubleVector cur_mean(IvectorDim());
    DoubleMatrix quadratic_inv(IvectorDim(), IvectorDim());
    InvertWithFlooring(quadratic, quadratic_inv);
    cur_mean += quadratic_inv * linear;
    // The loop is finding successively better approximation points
    // for the quadratic expansion of the weights.
    int num_iters = 4;
    double change_threshold = 0.1;
    // this (in 2-norm), we abort early.
    for (int iter = 0; iter < num_iters; iter++) {
      DoubleVector this_linear(linear);
      DoubleMatrix this_quadratic(quadratic);
      GetIvectorDistWeight(utt_stats, cur_mean, this_linear, this_quadratic);
      InvertWithFlooring(this_quadratic, quadratic_inv);
      DoubleVector mean_diff(cur_mean);
      cur_mean = quadratic_inv * this_linear;
      mean_diff = mean_diff - cur_mean;
      double change = mean_diff.Norm(2.0);
      cout << "On iter " << iter << ", iVector changed by " << change;
      if (change < change_threshold) {
        break;
      }
    }
    mean = cur_mean;
    if (var != NULL) {
      (*var) = quadratic_inv;
    }
  }
}

void IvectorEstimator::Run(const UtteranceStats &utt_stats,
                           DoubleVector &ivector) {
  if (ivector.Size() != IvectorDim()) {
    ivector.Resize(IvectorDim());
  }
  double time_start, time_end;
  //time_start = idec::TimeUtils::GetTimeMilliseconds();
  ivector(0) = PriorOffset();
  GetIvectorDistribution(utt_stats, ivector, NULL);
  ivector(0) -= PriorOffset();
  //time_end = idec::TimeUtils::GetTimeMilliseconds();
  //cout << time_end - time_start << " ms." << endl;
}

bool IvectorEstimator::IvectorDependentWeights() const {
  return w_.Rows() != 0;
}