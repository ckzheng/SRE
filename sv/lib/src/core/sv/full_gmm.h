#ifndef _FULL_GMM_H_
#define _FULL_GMM_H_
#include <vector>
#include "new_matrix.h"
#include "new_vector.h"
#include "resource_loader.h"
using namespace std;

class FullGmm {
 public:
  FullGmm(const UbmResource &fgmm);

  //FullGmm() {}

  ~FullGmm() {}

  int Dim() const {
    return dim_;
  }

  void LogLikelihoodsPreselect(const DoubleVector &data,
	  const vector<int> &indices, DoubleVector &loglikes);

  const DoubleVector &Gconsts() const {
    return g_consts_;
  }

  const DoubleVector& Weights() const {
    return weights_;
  }

  const vector<DoubleMatrix> &InvCovars() const {
    return inv_covars_;
  }

  int NumGauss() const {
    return mixture_;
  }

  void GetMeans(DoubleMatrix &M) const;

  void Clear();

 private:
  // Equals log(weight) - 0.5 * (log det(var) + mean'*inv(var)*mean)
  unsigned int dim_;
  unsigned int mixture_;
  const DoubleVector& g_consts_ ;
  const idec::xnnFloatRuntimeMatrix& g_consts_xnn_;
  // weights (not log).
  const DoubleVector& weights_ ;
  const idec::xnnFloatRuntimeMatrix & weights_xnn_;
  // Means times inverse covariances.
  const DoubleMatrix& means_invcovars_;
  const idec::xnnFloatRuntimeMatrix& means_invcovars_xnn_;
  // Inverse covariances.
  const vector<DoubleMatrix>& inv_covars_ ;
  const vector<idec::xnnFloatRuntimeMatrix>& inv_covars_xnn_;
  idec::xnnRuntimeColumnMatrixView < idec::xnnFloatRuntimeMatrix > mean_invcovars_view_;
  idec::xnnFloatRuntimeMatrix data_xnn_, data_sq_, square_;
  // Recompute gconsts_ if false.
  bool update_gconst;
};

#endif // !_FULL_GMM_H_
