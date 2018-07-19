#ifndef _BIC_H_
#define _BIC_H_

#include <vector>
#include "seg.h"
#include "mfcc.h"
#include "config.h"
#include "hmm/gmm.h"

namespace alspkdiar {

class Bic {
 public:
  //Bic(const SpeakerDiarizationOptions &opt) {
  //  this->mfcc_ = NULL;
  //  this->ubm_ = NULL;
  //  this->win_size_ = opt.win_size;
  //  this->step_size_ = opt.step_size;
  //  this->lambda_ = opt.lambda;
  //  this->alpha_ = opt.alpha;
  //  this->verbose_mode_ = opt.verbose_mode;
  //  this->buf_ = new float*[1024];
  //  this->buf_size_ = 1024;
  //}

  Bic(const BicOptions &opt) {
    this->mfcc_ = NULL;
    this->ubm_ = NULL;
    this->win_size_ = opt.win_size;
    this->step_size_ = opt.step_size;
    this->lambda_ = opt.lambda;
    this->alpha_ = opt.alpha;
    this->verbose_mode_ = opt.verbose_mode;
    this->buf_ = new float*[1024];
    this->buf_size_ = 1024;
  }

  void Init(const Mfcc *mfcc) {
    mfcc_ = mfcc;
  }

  ~Bic() {
    idec::IDEC_DELETE_ARRAY(buf_);
  }

  int GetLogDet(SegCluster &cluster, float &log_det) {
    FullGaussian *gauss = cluster.Gaussian();
    if (gauss->IsEmpty()) {
      vector<float *> feats;
      mfcc_->MergeFeature(cluster, feats);
      unsigned int len = 0;
      len = feats.size();
      gauss->Train(feats, mfcc_->GetDims(), 0, len);
      //cluster.SetGaussian(gauss);
    }
    //gauss = cluster.Gaussian();
    log_det = gauss->GetLDet();
    return ALS_OK;
  }

  int DeltaScore(SegCluster &cluster1,
                 SegCluster &cluster2, float &delta_score) {
    float log_det1, log_det2, log_det3;

    GetLogDet(cluster1, log_det1);
    GetLogDet(cluster2, log_det2);

    const FullGaussian *gauss1 = cluster1.Gaussian();
    const FullGaussian *gauss2 = cluster2.Gaussian();
    FullGaussian gauss3 = *gauss1;
    gauss3.Merge(gauss2);
    log_det3 = gauss3.GetLDet();

    int dim = gauss1->GetDim();
    float P = 0.5 * lambda_ * (dim + dim * (dim + 1) * 0.5) * log(
                gauss3.GetFrames());
    delta_score = gauss3.GetFrames() * log_det3 - gauss2->GetFrames() * log_det2 -
                  gauss1->GetFrames() * log_det1;
    //delta_score = 0.45 * delta_score - P;
    delta_score = delta_score - P;
    return ALS_OK;
  }

  int Score(SegCluster &cluster, alsid::HMM *&adapted_gmm, float &score) {
    vector<float *> feats;
    mfcc_->MergeFeature(cluster, feats);
    float P = 0.5 * ubm_->NumMix() * (1 + 2 * mfcc_->GetDims()) * lambda_;
    score = EvalSelfLikelihood(feats, 0, feats.size(),
                               adapted_gmm) - P * log(feats.size());
    return ALS_OK;
  }

  float EvalSelfLikelihood(vector<float *> &feat_buf, unsigned int start,
                           unsigned int len,
                           alsid::HMM *&adapted_gmm) {
    float lr = 0.0;
    if (adapted_gmm != NULL) {
      lr = EvalLikelihood(adapted_gmm, &feat_buf[0], start, len);
    } else {
      adapted_gmm = new alsid::HMM();
      adapted_gmm->InitFromUbm(*ubm_, 1);
      alsid::HMMTrainer  *adapted_gmm_trainer = new alsid::HMMTrainer(*adapted_gmm);
      float lr = 0.0;
      adapted_gmm_trainer->Reset();
      const double acc_threshold = 0.0001;
      adapted_gmm_trainer->Accumulate(&feat_buf[0], mfcc_->GetDims(),
                                      mfcc_->GetFrames(),
                                      -1.0, acc_threshold, start, start + len - 1);
      adapted_gmm_trainer->MapUpdate(gmm_update_opt_);
      lr = EvalLikelihood(adapted_gmm, &feat_buf[0], start, len);
      idec::IDEC_DELETE(adapted_gmm_trainer);
    }
    return lr;
  }

  /*
  float EvalSelfLikelihood(vector<float *>&feat_buf, unsigned int start,
  	 unsigned int len,
  	 alsid::HMM *&adapted_gmm) {
  	 float lr = 0.0;
  	 if (adapted_gmm != NULL) {
  		 lr = EvalLikelihood(adapted_gmm, &feat_buf[0], start, len);
  	 } else {
  		 adapted_gmm = new alsid::HMM();
  		 adapted_gmm->InitFromUbm(*ubm_, 1);
  		 alsid::HMMTrainer  *adapted_gmm_trainer = new alsid::HMMTrainer(*adapted_gmm);
  		 float lr = 0.0;
  		 adapted_gmm_trainer->Reset();
  		 const double acc_threshold = 0.0001;
  		 adapted_gmm_trainer->Accumulate(&feat_buf[0], mfcc_->GetDims(), mfcc_->GetFrames(),
  			 -1.0, acc_threshold, start, start + len - 1);
  		 adapted_gmm_trainer->MapUpdate(gmm_update_opt_);
  		 lr = EvalLikelihood(adapted_gmm, &feat_buf[0], start, len);
  		 idec::IDEC_DELETE(adapted_gmm_trainer);
  	 }
  	 return lr;
   }
   */

  float EvalSelfLikelihood(float **feat_buf, unsigned int start,
                           unsigned int len,
                           alsid::HMM *&adapted_gmm) {
    float lr = 0.0;
    if (adapted_gmm != NULL) {
      lr = EvalLikelihood(adapted_gmm, feat_buf, start, len);
    } else {
      adapted_gmm = new alsid::HMM();
      adapted_gmm->InitFromUbm(*ubm_, 1);
      alsid::HMMTrainer  *adapted_gmm_trainer = new alsid::HMMTrainer(*adapted_gmm);
      float lr = 0.0;
      adapted_gmm_trainer->Reset();
      const double acc_threshold = 0.0001;
      adapted_gmm_trainer->Accumulate(feat_buf, mfcc_->GetDims(), mfcc_->GetFrames(),
                                      -1.0, acc_threshold, start, start + len - 1);
      adapted_gmm_trainer->MapUpdate(gmm_update_opt_);
      lr = EvalLikelihood(adapted_gmm, feat_buf, start, len);
      idec::IDEC_DELETE(adapted_gmm_trainer);
    }
    return lr;
  }

  float EvalSelfLikelihood(float **feat_buf, unsigned int start,
                           unsigned int len) {
    alsid::HMM  *adapted_gmm = new alsid::HMM();
    adapted_gmm->InitFromUbm(*ubm_, 1);
    alsid::HMMTrainer  *adapted_gmm_trainer = new alsid::HMMTrainer(*adapted_gmm);
    float lr = 0.0;
    adapted_gmm_trainer->Reset();
    const double acc_threshold = 0.0001;
    adapted_gmm_trainer->Accumulate(feat_buf, mfcc_->GetDims(), mfcc_->GetFrames(),
                                    -1.0, acc_threshold, start, start + len - 1);
    adapted_gmm_trainer->MapUpdate(gmm_update_opt_);
    lr = EvalLikelihood(adapted_gmm, feat_buf, start, len);
    idec::IDEC_DELETE(adapted_gmm);
    idec::IDEC_DELETE(adapted_gmm_trainer);
    return lr;
  }

  float EvalLikelihood(alsid::HMM *mdl, float **feat_buf,
                       unsigned int start,
                       unsigned int len) {
    unsigned int T = 0;
    float tot_lr = 0;
    // unsigned int end = feat_buf.NumCols() > (start + len) ? (start + len) : feat_buf.NumCols();
    unsigned int end = start + len;
    float tot_lr_cur = LOG_ZERO;
    mdl->BeginEvaluate(-1);
    mdl->Evaluate(feat_buf, mfcc_->GetDims(), mfcc_->GetFrames(), &tot_lr, start,
                  end);
    // return (float)tot_lr / (float)len;
    return tot_lr;
  }

  int TurnDetect(const Seg &seg, std::vector<int> &change_point) {
    unsigned int N = seg.Length();

    int start = seg.begin;
    const int min_window = 100;
    if (N < 2 * min_window) {
      idec::IDEC_INFO << "total:" << N;
      return ALS_OK;
    }

    unsigned int start1 = start;
    unsigned int end1 = start1 + win_size_ - 1;
    unsigned int start2 = end1 + 1;
    unsigned int end2 = start2 + win_size_ - 1;

    std::vector<float> bic_scores;
    std::vector<int> possible_change_point;
    bic_scores.reserve(128);
    possible_change_point.reserve(128);
    float **feat_buf = mfcc_->GetFeature();
    float P = 0.5 * param_num_ * lambda_;
    float score1, score2, score3, bic_score, new_score;
    // calculate bic scores;
    while (end2 < start + N) {
      score1 = EvalSelfLikelihood(feat_buf, start1,
                                  end1 - start1) - P * log(end1 - start1);
      score2 = EvalSelfLikelihood(feat_buf, start2,
                                  end2 - start2) - P * log(end2 - start2);
      score3 = EvalSelfLikelihood(feat_buf, start1,
                                  end2 - start1) - P * log(end2 - start1);

      bic_score = score1 + score2 - score3;
      bic_scores.push_back(bic_score);
      possible_change_point.push_back(start1);

      start1 += step_size_;
      end1 += step_size_;
      start2 += step_size_;
      end2 += step_size_;
    }

    // bic score smoothing
    for (unsigned int i = 1; i < bic_scores.size() - 1; ++i) {
      new_score = 0.25 * bic_scores[i - 1] + 0.25 * bic_scores[i + 1] + 0.5*
                  bic_scores[i];
      bic_scores[i] = new_score;
    }

    float sum = 0.0;
    float sum2 = 0.0;
    for (unsigned int i = 0; i < bic_scores.size(); ++i) {
      sum += bic_scores[i];
    }

    double mean = sum / bic_scores.size();
    for (unsigned int i = 0; i < bic_scores.size(); ++i) {
      bic_scores[i] = bic_scores[i] - mean;
      sum2 = bic_scores[i] * bic_scores[i];
    }

    double std = sqrt(sum2 / bic_scores.size());

    // find true change point;
    for (unsigned int i = 1, j = 0; i < bic_scores.size() - 1; ++i) {
      //if (bic_scores[i] > 0) {
      j = i - 1;
      double min_left = bic_scores[i];
      bool ok = true;
      while (ok && (j > 0)) {
        if (bic_scores[j] < min_left) {
          min_left = bic_scores[j];
          --j;
        } else {
          ok = false;
        }
      }

      if (fabs(bic_scores[i] - min_left) > alpha_ * std) {
        // search right min
        int j = i + 1;
        double min_right = bic_scores[i];
        bool ok = true;
        while (ok && (j < bic_scores.size())) {
          if (bic_scores[j] < min_right) {
            min_right = bic_scores[j];
            j++;
          } else {
            ok = false;
          }
        }

        if (fabs(bic_scores[i] - min_right) > alpha_ * std) {
          change_point.push_back(possible_change_point[i]);
        }
      }
    }
    return ALS_OK;
  }

  int TurnDetectThreshLess(const Seg &seg, std::vector<int> &change_point) {
    float bic_score = -10e10;
    unsigned int N = seg.Length();
    unsigned int start = seg.begin;
    float **feat_buf = mfcc_->GetFeature();

    unsigned int start1 = start;
    unsigned int end1 = start1 + win_size_ / 2 - 1;
    unsigned int start2 = end1 + 1;
    unsigned int end2 = start2 + win_size_ / 2 - 1;

    std::vector<float> bic_scores;
    std::vector<unsigned int> possible_change_point;
    bic_scores.reserve(128);
    possible_change_point.reserve(128);
    SegCluster cluster1, cluster2;
    // calculate bic scores;
    while (end2 < start + N) {
      cluster1.Add(Seg(start1, end1));
      cluster2.Add(Seg(start2, end2));
      DeltaScore(cluster1, cluster2, bic_score);
      bic_scores.push_back(bic_score);
      possible_change_point.push_back(end1);
      start1 += step_size_;
      end1 += step_size_;
      start2 += step_size_;
      end2 += step_size_;
      cluster1.Clear();
      cluster2.Clear();
    }

    if (bic_scores.size() == 0) {
      return ALS_OK;
    }

    // bic score smoothing
    float new_score;
    for (unsigned int i = 1; i < bic_scores.size() - 1; ++i) {
      new_score = 0.25 * bic_scores[i - 1] + 0.25 * bic_scores[i + 1] + 0.5*
                  bic_scores[i];
      bic_scores[i] = new_score;
    }

    float sum = 0.0;
    float sum2 = 0.0;
    for (unsigned int i = 0; i < bic_scores.size(); ++i) {
      sum += bic_scores[i];
    }

    double mean = sum / bic_scores.size();
    for (unsigned int i = 0; i < bic_scores.size(); ++i) {
      bic_scores[i] = bic_scores[i] - mean;
      sum2 += bic_scores[i] * bic_scores[i];
    }

    double std = sqrt(sum2 / (bic_scores.size() - 1));

    // find true change point;
    bool ok = true;
    double min_left, min_right;
    for (unsigned int i = 1, j = 0; i < bic_scores.size() - 1; ++i) {
      j = i - 1;
      min_left = bic_scores[i];
      ok = true;
      while (ok && (j > 0)) {
        if (bic_scores[j] < min_left) {
          min_left = bic_scores[j];
          --j;
        } else {
          ok = false;
        }
      }

      if (fabs(bic_scores[i] - min_left) > alpha_ * std) {
        // search right min
        j = i + 1;
        min_right = bic_scores[i];
        ok = true;
        while (ok && (j < bic_scores.size())) {
          if (bic_scores[j] < min_right) {
            min_right = bic_scores[j];
            j++;
          } else {
            ok = false;
          }
        }

        if (fabs(bic_scores[i] - min_right) > alpha_ * std) {
          change_point.push_back(possible_change_point[i]);
        }
      }
    }
    return ALS_OK;
  }

  int TurnDetectFullGaussian(const Seg &seg,
                             std::vector<int> &change_point) {
    int S = seg.begin;
    unsigned int N = seg.Length();

    if (N < win_size_) {
      idec::IDEC_INFO << "total:" << N;
      return ALS_OK;
    }

    unsigned int win = win_size_;
    float delta_score, max_score;
    //Seg seg1, seg2;
    SegCluster cluster1, cluster2;
    unsigned int start = S;
    for (unsigned int cur_idx = S + win_size_; cur_idx <= S + N;) {
      unsigned int anchor = 0;
      float max_score = -10e6;
      for (unsigned int i = start + win_size_ / 2; i <= cur_idx - win_size_ / 2;
           i = i + step_size_) {
        float delta_score;
        cluster1.Add(Seg(start, i));
        cluster2.Add(Seg(i + 1, cur_idx));
        DeltaScore(cluster1, cluster2, delta_score);
        if (delta_score > max_score) {
          max_score = delta_score;
          anchor = i;
        }
      }

      if (max_score < 0.0) {
        cur_idx = cur_idx + step_size_ * 5;
      } else {
        start = anchor;
        cur_idx = start + win_size_;
        change_point.push_back(anchor);
      }
      cluster1.Clear();
      cluster2.Clear();
    }
    return ALS_OK;
  }

 private:
  int win_size_;
  double alpha_;
  double lambda_;
  int step_size_;
  bool verbose_mode_;
  int param_num_;
  float **buf_;
  int buf_size_;
  const Mfcc *mfcc_;
  alsid::GMM *ubm_;
  alsid::MapUpdateOption gmm_update_opt_;
};
}
#endif
