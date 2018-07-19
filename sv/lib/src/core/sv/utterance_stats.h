#ifndef _UTTERANCE_STATS_H_
#define _UTTERANCE_STATS_H_

#include <vector>
#include "posterior.h"
#include "new_matrix.h"
#include "new_vector.h"

using namespace std;

class UtteranceStats {
 public:
  UtteranceStats() {}
  UtteranceStats(int num_gauss, int feat_dim, bool need_2nd_order_stats);
  void AccStats(const DoubleMatrix &feats, const Posterior &post);
  void MergeStats(const UtteranceStats &utter_stats) {
    int feat_dim = FeatDim();
    int mix_num = MixtureNum();
    idec::IDEC_ASSERT(feat_dim == utter_stats.FeatDim());
    idec::IDEC_ASSERT(mix_num == utter_stats.MixtureNum());
    const DoubleVector &utt_gamma = utter_stats.Gamma();
    const DoubleMatrix &utt_X = utter_stats.X();
    for (int i = 0; i < mix_num; ++i) {
      gamma_(i) += utt_gamma(i);
      for (int j = 0; j < feat_dim; ++j) {
        X_(i, j) += utt_X(i, j);
      }
    }
  }

  void Scale(double scale);

  double NumFrames() const {
    return gamma_.Sum();
  }

  unsigned int FeatDim() const {
    return X_.Cols();
  }

  unsigned int MixtureNum() const {
    return gamma_.Size();
  }

  DoubleVector &Gamma() {
    return gamma_;
  }

  DoubleMatrix &X() {
    return X_;
  }

  vector<DoubleMatrix> &S() {
    return S_;
  }

  const DoubleVector &Gamma() const {
    return gamma_;
  }

  const DoubleMatrix &X() const {
    return X_;
  }

  const vector<DoubleMatrix> &S() const {
    return S_;
  }

  void Clear() {
    gamma_.SetAllValues(0.0);
    X_.SetAllValues(0.0);
    if (S_.size() != 0) {
      for (int i = 0; i < S_.size(); ++i) {
        S_[i].SetAllValues(0.0);
      }
    }
  }

 private:
  DoubleVector gamma_;
  DoubleMatrix X_;
  vector<DoubleMatrix> S_;
};

#endif
