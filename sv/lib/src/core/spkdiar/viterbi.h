#ifndef _VITERBI_H_
#define _VITERBI_H_

#include<vector>
#include "hmm/gmm.h"

namespace alspkdiar {

class Viterbi {
 public:
  Viterbi(const vector<alsid::GMM>& stat_vect, float *trans_matrix);
  int ComputeAndAccumulate(float **feature, int dims, int cols,
                                   unsigned long start, unsigned long count, double fudge);
  const vector<long> &GetPath();
  float LogTransition(unsigned long i, unsigned long j) const;
  float GetLlp();
  void Reset();
 private:
  int feature_count_;
  bool path_defined_;
  bool llp_defined_;
  float llp_;
  float *trans_matrix_;
  vector<long> path_;
  vector<int> tmp_tab_;
  vector<float> tmp_llk_vect_;
  vector<float> llp_vect_;
  vector<float> tmp_llp_vect_;
  vector<alsid::GMM> stat_vect_;
};
}

#endif
