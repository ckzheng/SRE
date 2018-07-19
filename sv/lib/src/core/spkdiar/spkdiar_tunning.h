#ifndef _SPK_DIAR_TUNNING_H_
#define _SPK_DIAR_TUNNING_H_

#include <set>
#include <string>
#include "hmm/gmm.h"
#include "speaker_cluster.h"
#include "mfcc.h"
#include "viterbi.h"
#include "seg.h"
#include "ahc.h"

namespace alspkdiar {

class SpkDiarTunning {
 public:
  SpkDiarTunning(const alsid::GMM &ubm,
                 const alsid::MapUpdateOption &gmm_update_opt, const HmmOptions &opt) : gmm_update_opt_(gmm_update_opt) {
    this->mfcc_ = NULL;
    this->viterbi_buffer_length_ = opt.viterbi_buffer_length;
    this->epsilon_ = opt.epsilon;
    this->iterationNb_ = opt.iteration_nb;
    this->gamma_ = opt.gamma;	
    this->trans_method_ = opt.trans_method;
    this->verbose_mode_ = opt.verbose_mode;
	this->ubm_ = ubm;
    this->state_num_ = 0;
  }

  ~SpkDiarTunning() {
    //idec::IDEC_DELETE_ARRAY(transitions_);
  }

  double GetTransition(unsigned int i, unsigned int j) {
    if (i >= state_num_ || j >= state_num_) {
      idec::IDEC_ERROR << "Out of boundary in HMM transition matrix ";
      return ALSERR_UNHANDLED_EXCEPTION;
    }
    return transitions_[i + j*state_num_];
  }

  int SetTransition(double prob, unsigned int i,
                    unsigned int j) {
    if (i >= state_num_ || j >= state_num_) {
      idec::IDEC_ERROR << "Out of boundary in HMM transition matrix! ";
      return ALSERR_UNHANDLED_EXCEPTION;
    }

    if (prob >= 1.0 || prob <= 0.0) {
      idec::IDEC_ERROR << "Set invalid HMM transition probability! ";
      return ALSERR_UNHANDLED_EXCEPTION;
    }
    transitions_[i + j*state_num_] = log(prob);
    return ALS_OK;
  }

  int ComputeTransitions(const string &transition_method,
                         double gamma/*, const SpeakerCluster& spk_cluster*/) {
    if (transition_method == "Equiprob") {
      if ((gamma < 0) || (gamma>1)) {
        idec::IDEC_ERROR << "Gamma is " << gamma << " not in interval 0 1!";
        return ALSERR_UNHANDLED_EXCEPTION;
      }

      for (unsigned long i = 0; i < state_num_; i++) {
        for (unsigned long j = 0; j < state_num_; j++) {
          if (i == j) {
            SetTransition(gamma, i, j);
          } else {
            SetTransition((1.0 - gamma) / (double)(state_num_ - 1.0), i, j);
          }
        }
      }
    }

    if (transition_method == "RapConst") {
      double transij = ((double)1.0 / (gamma + (double)(state_num_ - 1.0)));
      double transii = (gamma * transij);
      for (unsigned long i = 0; i < state_num_; i++) {
        for (unsigned long j = 0; j < state_num_; j++) {
          if (i == j) {
            SetTransition(transii, i, j);
          } else {
            SetTransition(transij, i, j);
          }
        }
      }
    }

    if (transition_method == "Unity") {
      for (unsigned long i = 0; i < state_num_; i++) {
        for (unsigned long j = 0; j < state_num_; j++) {
          SetTransition(1.0, i, j);
        }
      }
    }

    return ALS_OK;
  }

  int ReSegmentProcess(Mfcc &mfcc, Ahc &ahc,
                       SpeakerCluster &spk_cluster, const SegCluster &seg_cluster) {

    if (spk_cluster.Size() <= 1) {
      idec::IDEC_ERROR << "Invalid parameter spk_cluster of ReSegmentProcess.";
      return ALSERR_UNHANDLED_EXCEPTION;
    }

    const string trainAlgo = "MAP";
    const string testAlgo = "Viterbi";
    //const bool verbose_mode_ = true;
    mfcc_ = &mfcc;
    ahc_ = &ahc;
	// 200s
	//const int total_length = 20000;
	//if (seg_cluster.Length() > total_length) {
	//	ubm_ = SegEM(seg_cluster);
	//}

    const int N_state = 2;
    if (state_num_ != N_state) {
      state_num_ = N_state;
      transitions_.resize(state_num_ * state_num_);
    }

    if (verbose_mode_) {
      idec::IDEC_INFO << ">> Read the segmentation <<";
      idec::IDEC_INFO << "HMM has " << state_num_ << " states";
    }

    int init_len_of_prob = seg_cluster.Size();
    vector<float> actual_viterbi_prob, previous_viterbi_prob;
    actual_viterbi_prob.resize(init_len_of_prob);

    int indice = 0;
    double fudge = 1.0;

    SegCluster new_cluster;
    SpeakerCluster cur_spk_cluster, new_spk_cluster, all_spk_cluster;

    new_spk_cluster = spk_cluster;

    ComputeTransitions(trans_method_, gamma_);

    do {
      indice++;
      if (verbose_mode_) {
        idec::IDEC_INFO << ">> Loop Decoding Pass - Adaptation : " << indice << " <<";
      }

      cur_spk_cluster = new_spk_cluster;

      if (trainAlgo == "MAP") {
        SegAdaptation(new_spk_cluster);
      }

      if (testAlgo == "Viterbi") {
        ViterbiDecoding(seg_cluster, new_cluster, actual_viterbi_prob, fudge);
      }

      InitializeSpeakerCluster(new_cluster, new_spk_cluster);

      // smoothing
      TwoSpkSmoothing(new_spk_cluster);

    } while ((IsStopReSeg(actual_viterbi_prob, previous_viterbi_prob,
                          cur_spk_cluster, new_spk_cluster)) && (indice < iterationNb_));

    spk_cluster = new_spk_cluster;

    return ALS_OK;
  }

  int InitializeAllSpeakerCluster(const SpeakerCluster &spk_cluster,
                                  SpeakerCluster &all_spk_cluster) {
    if (all_spk_cluster.Size() != 0) {
      all_spk_cluster.Clear();
    }

    SegCluster new_cluster;
    for (unsigned int i = 0; i < spk_cluster.Size(); ++i) {
      const SegCluster &cluster = spk_cluster.Get(i);
      for(unsigned int j =0; j < cluster.Size(); ++j) {
        Seg segment = cluster.GetSeg(j);
        if (segment.Length() < 100) {
          continue;
        }
        segment.label = 0;
        new_cluster.Add(segment);
        all_spk_cluster.Add(new_cluster);
        new_cluster.Clear();
      }
    }
    return ALS_OK;
  }

  int TwoSpkSmoothing(SpeakerCluster &spk_cluster) const {
    const int thresh = 30, label0 = 0, label1 = 1;
    idec::IDEC_ASSERT(spk_cluster.Size() == 2);
    const SegCluster &cluster0 = spk_cluster.Get(0);
    const SegCluster &cluster1 = spk_cluster.Get(1);
    vector<int> idxs;
    for (unsigned int i = 0; i < cluster0.Size(); ++i) {
      Seg seg = cluster0.GetSeg(i);
      if (seg.Length() < thresh) {
        seg.label = label1;
        spk_cluster.Get(1).Merge(seg);
        idxs.push_back(i);
      }
    }

    if (idxs.size() != 0) {
      sort(idxs.begin(), idxs.end());
      unique(idxs.begin(), idxs.end());
      for (int i = idxs.size() - 1; i >= 0; --i) {
        spk_cluster.Get(0).Remove(idxs[i]);
      }
      idxs.clear();
    }

    for (unsigned int i = 0; i < cluster1.Size(); ++i) {
      Seg seg = cluster1.GetSeg(i);
      if (seg.Length() < thresh) {
        seg.label = label0;
        spk_cluster.Get(0).Merge(seg);
        idxs.push_back(i);
      }
    }

    if (idxs.size() != 0) {
      sort(idxs.begin(), idxs.end());
      unique(idxs.begin(), idxs.end());
      for (int i = idxs.size() - 1; i >= 0; --i) {
        spk_cluster.Get(1).Remove(idxs[i]);
      }
      idxs.clear();
    }
    return ALS_OK;
  }

  bool IsStopReSeg(vector<float> &actual_viterbi_prob,
                   vector<float> &previous_viterbi_prob,
                   const SpeakerCluster &spk_cluster, const SpeakerCluster &new_spk_cluster) {
    if (spk_cluster.Size() != new_spk_cluster.Size()) {
      idec::IDEC_ERROR << "Problem with number of clusters!";
    }

    if (actual_viterbi_prob.size() != previous_viterbi_prob.size()) {
      unsigned int length = actual_viterbi_prob.size();
      if (previous_viterbi_prob.size() != length) {
        previous_viterbi_prob.resize(length);
      }

      for (unsigned long jseg = 0; jseg < length; jseg++) {
        previous_viterbi_prob[jseg] = actual_viterbi_prob[jseg];
      }
      return true;
    }

    for (unsigned int i = 0; i < spk_cluster.Size(); ++i) {
      if (IsDifferentSegmentation(spk_cluster.Get(i),
                                  new_spk_cluster.Get(i))) {
        if (!IsLessepsilon(actual_viterbi_prob, previous_viterbi_prob, epsilon_)) {
          return true;
        }
      }
    }
    return false;
  }

  // comparison between two viterbi paths according to likelihoods
  bool IsLessepsilon(vector<float> &actual_viterbi_prob,
                     vector<float> &previous_viterbi_prob, double epsilon) const {
    double diff;
    unsigned int length = actual_viterbi_prob.size();
    for (unsigned long iseg = 0; iseg < length; iseg++) {
      diff = fabs(actual_viterbi_prob[iseg] - previous_viterbi_prob[iseg]);
      if (diff > epsilon) {
        for (unsigned long jseg = 0; jseg < length; jseg++) {
          previous_viterbi_prob[jseg] = actual_viterbi_prob[jseg];
        }
        return false;
      }
    }
    return true;
  }

  bool IsDifferentSegmentation(const SegCluster &cluster1,
                               const SegCluster &cluster2) const {
    return cluster1.IsDiff(cluster2);
  }

  int InitializeCluster(const SpeakerCluster &spk_cluster,
                        SegCluster &cluster) {
    for (unsigned int i = 0; i < spk_cluster.Size(); ++i) {
      spk_cluster.Get(i).SetLabel(i);
      cluster.Merge(spk_cluster.Get(i));
    }
    return ALS_OK;
  }

  int InitializeSpeakerCluster(const SegCluster &cluster,
                               SpeakerCluster &spk_cluster) const {
    if (spk_cluster.Size() != 0) {
      spk_cluster.Clear();
    }

    set<int> labels;
    for (unsigned int i = 0; i < cluster.Size(); ++i) {
      const Seg &seg = cluster.GetSeg(i);
      labels.insert(seg.label);
    }

    set<int>::iterator iter = labels.begin();
    while (iter != labels.end()) {
      SegCluster tmp;
      for (unsigned int i = 0; i < cluster.Size(); ++i) {
        const Seg &seg = cluster.GetSeg(i);
        if (seg.label == *iter) {
          tmp.Add(seg);
        }
      }
      spk_cluster.Add(tmp);
      iter++;
    }
    return ALS_OK;
  }

  int ViterbiDecoding(const SegCluster &cluster,
                      SegCluster &new_cluster, vector<float> &viterbi_prob, double fudge) const {

    if (new_cluster.Size() != 0) {
      new_cluster.Clear();
    }

    Viterbi va = Viterbi(stat_vect_, &transitions_[0]);
    if (cluster.Size() == 0) {
      idec::IDEC_ERROR << "Not segments to decode in ViterbiDecoding.";
      return ALSERR_SPK_DIAR_SPEECH_TOO_SHORT;
    }

    unsigned long iprob = 0;
    viterbi_prob.resize(cluster.Size());
    // Loop on each segment comming from the macro-class acoustic segmentation
    for (unsigned int i = 0; i < cluster.Size(); ++i) {
      const Seg &segment = cluster.GetSeg(i);
      // Reset the viterbi accumulator
      va.Reset();
      // Accumulate the stats for a segment
      AccumulateStatViterbi(va, segment, fudge);
      // Get the path;
      const vector<long> &path = va.GetPath();
      double llp = va.GetLlp();
      // Get the log likelihood of the path
      viterbi_prob[iprob++] = llp;
      // Put the viterbi path (for the current macro segment)  in the cluster
      CopyPathInCluster(new_cluster, path, segment.begin);
    }
    return ALS_OK;
  }

  int AccumulateStatViterbi(Viterbi &va,
                            const Seg &seg, double fudge) const {
    //double fudge = 1.0;
    //unsigned long viterbi_buffer_length_ = 10;
    unsigned int begin_idx = seg.begin;
    unsigned int end_idx = seg.end;
    unsigned int length = end_idx - begin_idx;
    if (viterbi_buffer_length_ <= end_idx) {
      for (unsigned int i = begin_idx; i <= end_idx - viterbi_buffer_length_;
           i += viterbi_buffer_length_) {
        va.ComputeAndAccumulate(mfcc_->GetFeature(), mfcc_->GetDims(),
                                mfcc_->GetFrames(), i, viterbi_buffer_length_, fudge);
      }
    }

    if (length % viterbi_buffer_length_ != 0) {
      va.ComputeAndAccumulate(mfcc_->GetFeature(), mfcc_->GetDims(),
                              mfcc_->GetFrames(), begin_idx + (length / viterbi_buffer_length_) *
                              viterbi_buffer_length_, length%viterbi_buffer_length_,
                              fudge); // End of the segment
    }
    return ALS_OK;
  }

  int CopyPathInCluster(SegCluster &cluster,
                        const vector<long> &path, unsigned long start) const {
    // Begin/end of the current segment
    unsigned long start_v, stop_v, indice_loc;
    indice_loc = path[0];
    start_v = start;
    for (unsigned long i = 1; i < path.size(); i++) {
      // End of a segment
      if ((path[i] != indice_loc)) {
        stop_v = (i - 1) + start;
        cluster.Add(Seg(start_v, stop_v, indice_loc));
        indice_loc = path[i];
        start_v = i + start;
      } //if
    } //for

    // Deal with the last segment if needed
    stop_v = path.size() + start;
    cluster.Add(Seg(start_v, stop_v, indice_loc));
    return ALS_OK;
  }

  alsid::GMM SegEM(const SegCluster &cluster) const {
    unsigned int len = 0;    
    const double acc_threshold = 0.0001;
	vector<float *> feature;
    mfcc_->MergeFeature(cluster, feature);    
    len = feature.size();
    alsid::HMM adapted_gmm;
    adapted_gmm.InitFromUbm(ubm_, 1);
    alsid::HMMTrainer adapted_gmm_trainer(adapted_gmm);
    adapted_gmm_trainer.Reset();
    adapted_gmm_trainer.Accumulate(&feature[0], mfcc_->GetDims(), len, -1.0,
                                   acc_threshold, 0, 0 + len - 1);
	alsid::MapUpdateOption gmm_update_opt;
	adapted_gmm_trainer.MleUpdate(gmm_update_opt);
	return adapted_gmm.GetStates()[0];
  }

  int SegAdaptation(const SpeakerCluster &spk_cluster) {
    unsigned int len = 0;
    vector<float *> feature;
    const double acc_threshold = 0.0001;
    stat_vect_.clear();
    for (int i = 0; i < spk_cluster.Size(); ++i) {
      mfcc_->MergeFeature(spk_cluster.Get(i), feature);
      len = feature.size();
      alsid::HMM adapted_gmm;
      adapted_gmm.InitFromUbm(ubm_, 1);
      alsid::HMMTrainer adapted_gmm_trainer (adapted_gmm);
      adapted_gmm_trainer.Reset();
      adapted_gmm_trainer.Accumulate(&feature[0], mfcc_->GetDims(), len, -1.0,
                                     acc_threshold, 0, 0 + len - 1);
      adapted_gmm_trainer.MapUpdate(gmm_update_opt_);
      stat_vect_.push_back(adapted_gmm.GetStates()[0]);
    }
    return ALS_OK;
  }

 private:
  Mfcc *mfcc_;
  Ahc *ahc_;
  alsid::GMM ubm_;
  vector<alsid::GMM> stat_vect_;
  const alsid::MapUpdateOption &gmm_update_opt_;
  vector<float> transitions_;
  unsigned int state_num_;
  bool verbose_mode_;
  unsigned int viterbi_buffer_length_;
  unsigned int iterationNb_;
  double epsilon_;
  double gamma_;
  string trans_method_;
};
}

#endif
