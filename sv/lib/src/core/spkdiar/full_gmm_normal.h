#ifndef _FULL_GMM_NORMAL_H_
#define _FULL_GMM_NORMAL_H_

#include "full_gmm.h"

class  FullGmmNormal {
 public:
  FullGmmNormal() {}
  explicit FullGmmNormal(const FullGmm &full_gmm) {

  }

  ~FullGmmNormal() {}

  void FullGmmNormal::Resize(int nmix, int dim) {
    idec::IDEC_ASSERT(nmix > 0 && dim > 0);
    if (weights_.Size() != nmix)
      weights_.Resize(nmix);

    if (means_.size() != nmix)
      means_.resize(nmix);

    if (vars_.size() != nmix)
      vars_.resize(nmix);
    for (int i = 0; i < nmix; i++) {
      if (vars_[i].Rows() != nmix ||
          vars_[i].Cols() != dim) {
        vars_[i].Resize(nmix, dim);
      }
    }
  }

  void FullGmmNormal::CopyFromFullGmm(const FullGmm &fullgmm) {
    /// resize the variables to fit the gmm
    size_t dim = fullgmm.Dim();
    size_t num_gauss = fullgmm.NumGauss();
    Resize(num_gauss, dim);

    /// copy weights
    weights_ = fullgmm.Weights();

    /// we need to split the natural components for each gaussian
    const vector<DoubleMatrix> &inv_covars = fullgmm.InvCovars();
    const DoubleMatrix &means_inv_covars = fullgmm.MeansInvCovars();
    for (size_t i = 0; i < num_gauss; i++) {
      // copy and invert (inverse) covariance matrix
      vars_[i] = inv_covars[i].Invert();
      means_[i] = vars_[i] * means_inv_covars.Rowv(i);
    }
  }

  void FullGmmNormal::CopyToFullGmm(FullGmm &fullgmm, int flags) {
    idec::IDEC_ASSERT(weights_.Size() == fullgmm.Weights().Size()
                      && means_.size() == fullgmm.Dim());

    FullGmmNormal oldg(fullgmm);
    const int kGmmMeans = 0x001, kGmmVariances = 0x002, kGmmWeights = 0x004;

    if (flags & kGmmWeights) {
      fullgmm.Weights(weights_);
    }

    size_t num_comp = fullgmm.NumGauss(), dim = fullgmm.Dim();
    vector<DoubleMatrix > inv_covars;
    DoubleMatrix mean_inv_covars;
    for (size_t i = 0; i < num_comp; i++) {
      if (flags & kGmmVariances) {
        inv_covars.push_back(vars_[i].Invert());
        if (!(flags & kGmmMeans)) {
          mean_inv_covars.Row(i, inv_covars[i] * means_[i]);
        }
      }
    }

    if (flags & kGmmVariances) {
      fullgmm.InvCovars(inv_covars);
    }

    if (flags & kGmmMeans) {
      fullgmm.MeansInvCovars(mean_inv_covars);
    }

    fullgmm.UpdateGconst(false);
  }

  const DoubleVector &Weights() const {
    return weights_;
  }

  const vector<DoubleVector> &Means() const {
    return means_;
  }

  const vector<DoubleMatrix> &Vars() const {
    return vars_;
  }

  void Weights(const DoubleVector &weights) {
    weights_ = weights;
  }

  void Means(const vector<DoubleVector> &means) {
    means_ = means;
  }

  void Vars(const vector<DoubleMatrix> &vars) {
    vars_ = vars;
  }

 private:
  DoubleVector weights_;              ///< weights (not log).
  vector<DoubleVector> means_;                ///< Means
  vector<DoubleMatrix> vars_;  ///< covariances
};

#endif
