#include "viterbi.h"

using std::vector;

namespace alspkdiar {

Viterbi::Viterbi(const vector<alsid::GMM> &stat_vect, float *trans_matrix) {
  this->stat_vect_ = stat_vect;
  this->trans_matrix_ = trans_matrix;
}

int Viterbi::ComputeAndAccumulate(float **feature, int dims, int cols,
    unsigned long start, unsigned long count, double fudge) {
  unsigned long i, j, states = stat_vect_.size();
  llp_defined_ = path_defined_ = false;
  double l;
  alsid::GMM gmm;
  // compute llk between the feature and each state
  tmp_llk_vect_.clear();
  for (i = 0; i < states; i++) {
    l = 0.0;
    for (int k = start; k < (start + count); ++k) {
      gmm = stat_vect_[i];
      l += gmm.OutP(feature[k], dims);
    }

    tmp_llk_vect_.push_back(l / count);
  }

  // if first feature in the viterbi path
  if (feature_count_ == 0) {
    llp_vect_ = tmp_llk_vect_;
    for (unsigned long c = 0; c < count; c++)
      for (i = 0; i < states; i++)
        tmp_tab_.push_back(i);
  } else {
    tmp_llp_vect_.clear();
    unsigned long maxInd = 0;
    float maxllp = 0;
    float llp;

    // For the first frame of the n block (n>0)- find the path
    for (i = 0; i < states; i++) {
      for (j = 0; j < states; j++) {
        llp = llp_vect_[j] + tmp_llk_vect_[i] + LogTransition(j, i)*fudge;
        if (j == 0 || llp > maxllp) {
          maxllp = llp;
          maxInd = j;
        }
      }

      tmp_llp_vect_.push_back(maxllp);
      tmp_tab_.push_back(maxInd);
    }

    for (unsigned long c = 1; c < count; c++) {
      for (i = 0; i < states; i++) {
        tmp_tab_.push_back(i);
      }
    }

    llp_vect_ = tmp_llp_vect_;
  }
  feature_count_ += count;
  return ALS_OK;
}

const std::vector<long> &Viterbi::GetPath() {
  if (!path_defined_) {
    unsigned long i, max = 0;
    int states = stat_vect_.size();
    // looks for the largest llp
    for (i = 0; i<states; i++) {
      if (llp_vect_[i] > llp_vect_[max])
        max = i;
    }

    llp_ = llp_vect_[max];
    llp_defined_ = true;
    path_ = std::vector<long>(feature_count_);
    // path_.reserve(feature_count_);
    if (feature_count_ != 0) {
      for (i = feature_count_ - 1; i > 0; i--) {
        path_[i] = max;
        max = tmp_tab_[(i - 1)*states + max];
      }
      path_[0] = max;
    }
    path_defined_ = true;
  }
  return path_;
}

float Viterbi::LogTransition(unsigned long i, unsigned long j) {
  unsigned long size = stat_vect_.size();
  return trans_matrix_[j*size + i];
}

float Viterbi::GetLlp() {
  return llp_;
}

void Viterbi::Reset() {
  tmp_tab_.clear();
  llp_vect_.clear();
  llp_defined_ = false;
  path_defined_ = false;
  feature_count_ = 0;
}
}