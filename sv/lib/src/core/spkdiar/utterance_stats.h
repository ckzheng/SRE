#ifndef _UTTERANCE_STATS_H_
#define _UTTERANCE_STATS_H_

#include "posterior.h"
#include "new_matrix.h"
#include "new_vector.h"
#include <vector>

using namespace std;

class UtteranceStats {
 public:
  UtteranceStats() {}
  UtteranceStats(int num_gauss, int feat_dim, bool need_2nd_order_stats);
  void AccStats(const DoubleMatrix &feats, const Posterior &post);
  void MergeStats(const UtteranceStats &utter_stats);
  void Scale(double scale);
  double NumFrames() const { return gamma_.Sum(); }
  unsigned int FeatDim() const { return X_.Cols(); }
  unsigned int MixtureNum() const { return gamma_.Size(); }
  DoubleVector &Gamma() { return gamma_; }
  DoubleMatrix &X() { return X_; }
  vector<DoubleMatrix> &S() { return S_; }
  const DoubleVector &Gamma() const { return gamma_; }
  const DoubleMatrix &X() const { return X_; }
  const vector<DoubleMatrix> &S() const { return S_; }
  void Clear();

 private:
  //zeroth-order stats (summed posteriors), dimension [I]
  DoubleVector gamma_;
  //first-order stats, dimension [I][D]
  DoubleMatrix X_;
  //2nd-order stats, dimension [I][D][D], if required.
  vector<DoubleMatrix> S_;
};

#endif