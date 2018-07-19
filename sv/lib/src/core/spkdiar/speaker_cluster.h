#ifndef _SPEAKER_CLUSTER_H_
#define _SPEAKER_CLUSTER_H_

namespace alspkdiar {

#include <vector>
using std::vector;
#include "seg.h"

class SpeakerCluster {
 public:
  SpeakerCluster() {
    is_ivector_need_update_ = true;
  }

  unsigned int Size() const {
    return spks_.size();
  }

  SegCluster Get(unsigned int i) const {
    idec::IDEC_ASSERT(i < spks_.size());
    return spks_[i];
  }

  SegCluster &Get(unsigned int i) {
    idec::IDEC_ASSERT(i < spks_.size());
    return spks_[i];
  }

  unsigned int Length() const {
    unsigned int length = 0;
    for (unsigned int i = 0; i < spks_.size(); ++i) {
      length += spks_[i].Length();
    }
    return length;
  }

  void Remove(unsigned int i) {
    idec::IDEC_ASSERT(i < spks_.size());
    vector<SegCluster>::iterator iter = spks_.begin();
    spks_.erase(iter + i);
    is_ivector_need_update_ = true;
  }

  void Add(const SegCluster &seg_cluster) {
    spks_.push_back(seg_cluster);
    is_ivector_need_update_ = true;
  }

  void Clear() {
    spks_.clear();
    is_ivector_need_update_ = true;
  }

  void SetIvector(const DoubleVector &ivector) {
    ivector_ = ivector;
    is_ivector_need_update_ = false;
  }

  DoubleVector &Ivector() {
    return ivector_;
  }

  bool IsIvectorNeedUpdate() {
    return is_ivector_need_update_;
  }
 private:
  vector<SegCluster> spks_;
  DoubleVector ivector_;
  bool is_ivector_need_update_;
};
}
#endif
