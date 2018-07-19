#ifndef _DIAG_GMM_H_
#define _DIAG_GMM_H_

#include "new_matrix.h"
#include "new_vector.h"
#include "full_gmm.h"
#include "base/log_message.h"
#include <cmath>

using namespace std;

class DiagGmm {
 public:
  DiagGmm(DoubleVector &weights, DoubleVector &gconsts, DoubleMatrix &inv_vars,
          DoubleMatrix &means_invvars);
  DiagGmm() {}
  ~DiagGmm() { update_gconst_ = false; }
  int  Dim() const;
  int ComputeGconsts();
  void Resize(int nmix, int dim);
  void CopyFromFullGmm(const FullGmm &fullgmm);
  // Outputs the per-component log-likelihoods.
  void LogLikelihoods(const DoubleVector &feat, DoubleVector &loglikes)  const;
  void LogLikelihoods(const DoubleMatrix &data, DoubleMatrix &loglikes)  const;
  double LogLikelihood(const DoubleVector &data) const;
  int NumGauss() const;
  const DoubleVector& Gconsts() const { return gconsts_; }
  const DoubleVector& Weights() const { return weights_; }
  const DoubleMatrix& InvVars() const { return inv_vars_; }
  const DoubleMatrix& MeanInvVars() const { return means_invvars_; }
  /// Get gaussian selection information for one frame.
  double GaussianSelection(const DoubleVector &data, int num_gselect,
                           std::vector<int> &output) const;
  double GaussianSelection(const DoubleMatrix &data, int num_gselect,
                           std::vector<std::vector<int> > &output) const;
  void Read(const string &mdl_path);
  void GetVars(DoubleMatrix &vars);
  void GetMeans(DoubleMatrix &means);
  void WriteNLSFormat(const string &mdl_path);
 private:
  double LogAdd(double  x, double  y) const;
 private:
  DoubleVector weights_;
  //log(weight) - 0.5 * (log det(var) + mean*mean*inv(var))
  DoubleVector gconsts_;
  //Inverted(diagonal) variances.
  DoubleMatrix inv_vars_;
  //Means times inverted variance.
  DoubleMatrix means_invvars_;
  //Recompute gconsts_ if false.
  bool update_gconst_;
};

#endif // !_DIAG_GMM_H_
