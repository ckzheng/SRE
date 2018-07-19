#ifndef _IVECTOR_ESTIMATOR_H_
#define _IVECTOR_ESTIMATOR_H_

#include "new_matrix.h"
#include "new_vector.h"
#include "posterior.h"
#include "utterance_stats.h"
#include "full_gmm.h"
#include "config.h"

class IvectorEstimator {
 public:
  IvectorEstimator(const IvectorExtractOptions &opts,
                   const IvectorResource &iv_res) : prior_offset_(iv_res.prior_offset),
    Sigma_inv_(iv_res.sigma_inv), M_(iv_res.t_matrix), w_vec_(iv_res.w_vec),
    U_(iv_res.U), Sigma_inv_M_(iv_res.Sigma_inv_M), gconsts_(iv_res.gconsts),
    Sigma_inv_M_trans_(iv_res.Sigma_inv_M_trans) {
    //tot_auxf_change_ = NULL;
    verbose_mode_ = opts.verbose_mode;
  }

  ~IvectorEstimator() {};

  int FeatDim() const {
    idec::IDEC_ASSERT(!M_.empty());
    return M_[0].Rows();
  }

  int IvectorDim() const {
    if (M_.empty()) { return 0.0; }
    else { return M_[0].Cols(); }
  }

  int NumGauss() const {
    return M_.size();
  }

  void Run(const UtteranceStats &utt_stats, DoubleVector &ivector);

 private:
  double PriorOffset() const {
    return prior_offset_;
  }

  double GetLogDetNoFailure(const DoubleMatrix &var);
  void GetIvectorDistMean(const UtteranceStats &utt_stats, DoubleVector &linear,
                          DoubleMatrix &quadratic) const;
  void GetIvectorDistPrior(const UtteranceStats &utt_stats, DoubleVector &linear,
                           DoubleMatrix &quadratic) const;
  void GetIvectorDistWeight(const UtteranceStats &utt_stats,
                            const DoubleVector &mean, DoubleVector &linear, DoubleMatrix &quadratic) const;
  void GetIvectorDistribution(const UtteranceStats &utt_stats,
                              DoubleVector &mean, DoubleMatrix *var);
  bool IvectorDependentWeights() const;
  void InvertWithFlooring(DoubleMatrix &inverse_var, DoubleMatrix &var) const;

 private:
  const std::vector<DoubleMatrix> &M_;
  const std::vector<DoubleMatrix > &Sigma_inv_M_;
  const std::vector<DoubleMatrix > &Sigma_inv_M_trans_;
  const std::vector<DoubleMatrix > &Sigma_inv_;
  const DoubleMatrix w_;
  const DoubleVector &w_vec_;
  const DoubleVector &gconsts_;
  const vector<DoubleMatrix> &U_;
  //std::string  utt_;
  bool verbose_mode_;
  double prior_offset_;
};

#endif // !_IVECTOR_ESTIMATOR_H_