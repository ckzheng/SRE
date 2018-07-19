#ifndef _FULL_GMM_H_
#define _FULL_GMM_H_
#include <vector>
#include "new_matrix.h"
#include "new_vector.h"
#include "resource_loader.h"
#include "util/random.h"

using namespace std;

class DiagGmm;

class FullGmm {
 public:
  FullGmm(const UbmResource &fgmm);

  FullGmm() {}

  ~FullGmm() {}

  int Dim() const {
    return means_invcovars_.Cols();
  }

  void LogLikelihoodsPreselect(const DoubleVector &data,
                               const vector<int> &indices, DoubleVector &loglikes) const;

  int ComputeGconsts();

  void CopyFromDiagGmm(const DiagGmm &diag_gmm);

  const DoubleVector &Gconsts() const {
    return g_consts_;
  }

  void Gconsts(const DoubleVector &g_const ) {
    g_consts_ = g_const;
  }

  const DoubleVector Weights() const {
    return weights_;
  }

  void Weights(const DoubleVector &weight) {
    weights_ = weight;
  }

  const vector<DoubleMatrix> &InvCovars() const {
    return inv_covars_;
  }

  void InvCovars(const vector<DoubleMatrix> &inv_covars) {
    inv_covars_ = inv_covars;
  }

  const DoubleMatrix &MeansInvCovars() const {
    return means_invcovars_;
  }

  void MeansInvCovars(const DoubleMatrix &means_invcovars) {
    means_invcovars_ = means_invcovars;
  }

  int NumGauss() const {
    return weights_.Size();
  }

  void UpdateGconst(bool flag) {
	  update_gconst_ = flag;
  }

  void GetMeans(DoubleMatrix &M) const;

  void RemoveComponents(const vector<int> &gauss_in, bool renorm_weights);

  void Split(int target_components, float perturb_factor, vector<int>&history);

 private:
// Equals log(weight) - 0.5 * (log det(var) + mean'*inv(var)*mean)
  DoubleVector g_consts_ ;
// weights (not log).
  DoubleVector weights_ ;
// Means times inverse covariances.
  DoubleMatrix means_invcovars_;
// Inverse covariances.
  vector<DoubleMatrix> inv_covars_ ;

  idec::RandomGenerator rand_generator_;

// Recompute gconsts_ if false.
  bool update_gconst_;
};

#endif // !_FULL_GMM_H_