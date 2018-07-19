#ifndef _DIAG_GMM_H_
#define _DIAG_GMM_H_

#include "full_gmm.h"
#include "new_vector.h"
#include "new_matrix.h"

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
  //void LogLikelihoods(const DoubleVector &feat, DoubleVector &loglikes)  const;
  void LogLikelihoods(const DoubleMatrix &data, DoubleMatrix &loglikes)  const;
  //double LogLikelihood(const DoubleVector &data) const;
  int NumGauss() const;
  /// Get gaussian selection information for one frame.
  //double GaussianSelection(const DoubleVector &data, int num_gselect, vector<int> &output) const;
  double GaussianSelection(const DoubleMatrix &data, int num_gselect,
                           vector< vector<int> > &output) const;
  //void GetVars(DoubleMatrix &vars) const;
  //void GetMeans(DoubleMatrix &means) const;
  //void Read(const string &mdl_path);
  //void WriteNLSFormat(const string &mdl_path);
  //void EigenMatrix2XnnMatrix(const DoubleMatrix &mat, idec::xnnFloatRuntimeMatrix& xmat) const;
  // void EigenVector2XnnMatrix(const DoubleVector &mat, idec::xnnFloatRuntimeMatrix& xmat) const;
 private:
  double LogAdd(double  x, double  y) const;
 private:
  unsigned int dim_;
  unsigned int mixture_;
  DoubleVector weights_;
  //idec::xnnFloatRuntimeMatrix weights_xnn_;
  //log(weight) - 0.5 * (log det(var) + mean*mean*inv(var))
  DoubleVector gconsts_;
  //idec::xnnFloatRuntimeMatrix gconsts_xnn_;
  //Inverted(diagonal) variances.
  DoubleMatrix inv_vars_;
  //idec::xnnFloatRuntimeMatrix inv_vars_xnn_;
  //Means times inverted variance.
  DoubleMatrix means_invvars_;
  //idec::xnnFloatRuntimeMatrix means_invvars_xnn_;
  //Recompute gconsts_ if false.
  bool update_gconst_;
};

#endif // !_DIAG_GMM_H_
