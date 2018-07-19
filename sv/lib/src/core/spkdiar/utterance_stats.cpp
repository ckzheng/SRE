#include "utterance_stats.h"

using namespace std;

UtteranceStats::UtteranceStats(int num_gauss, int feat_dim,
                               bool need_2nd_order_stats) {
  gamma_ = DoubleVector(num_gauss);
  gamma_.SetAllValues(0.0);
  X_ = DoubleMatrix(num_gauss, feat_dim);
  X_.SetAllValues(0.0);
  S_ = vector<DoubleMatrix>();
  if (need_2nd_order_stats) {
    S_.resize(num_gauss);
    for (int i = 0; i < num_gauss; i++) {
      S_[i].Resize(feat_dim, feat_dim);
      S_[i].SetAllValues(0.0);
    }
  }
}
void UtteranceStats::MergeStats(const UtteranceStats &utter_stats) {
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

void UtteranceStats::AccStats(const DoubleMatrix &feats,
                              const Posterior &post) {
  typedef std::vector<std::pair<int, double> > VecType;
  int num_frames, num_gauss, feat_dim;
  num_frames = feats.Rows(),
  num_gauss = X_.Rows(),
  feat_dim = feats.Cols();

  idec::IDEC_ASSERT(X_.Cols() == feat_dim);
  idec::IDEC_ASSERT(feats.Rows() == static_cast<int>(post.Size()));
  bool update_variance = (!S_.empty());
  DoubleMatrix outer_prod;
  for (int t = 0; t < num_frames; t++) {
    const DoubleMatrix &frame = feats.Row(t);
    const VecType &this_post = post[t].Data();
    if (update_variance) {
      outer_prod = frame * frame.Transpose();
    }

    for (VecType::const_iterator iter = this_post.begin(); iter != this_post.end();
         ++iter) {
      int i = iter->first; // Gaussian index.
      idec::IDEC_ASSERT(i >= 0 && i < num_gauss);
      double weight = iter->second;
      gamma_(i) += weight;
      for (int j = 0; j < X_.Cols(); j++) {
        X_(i, j) += weight * frame(0, j);
      }

      if (update_variance) {
        S_[i] += outer_prod * weight;
      }
    }
  }
}

void UtteranceStats::Scale(double scale) {
  gamma_.Scale(scale);
  X_.Scale(scale);
  for (size_t i = 0; i < S_.size(); i++) {
    S_[i].Scale(scale);
  }
}

void UtteranceStats::Clear() {
  gamma_.SetAllValues(0.0);
  X_.SetAllValues(0.0);
  if (S_.size() != 0) {
    for (int i = 0; i < S_.size(); ++i) {
      S_[i].SetAllValues(0.0);
    }
  }
}