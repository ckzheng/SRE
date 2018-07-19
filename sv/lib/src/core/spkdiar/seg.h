#ifndef _SEG_H_
#define _SEG_H_

#include <vector>
#include <algorithm>
#include "full_gaussian.h"
#include "new_vector.h"

using std::vector;

namespace alspkdiar {
struct Seg {
  int begin;
  int end;
  int label;
  FullGaussian gauss;
  DoubleVector ivector;

  Seg(int b = 0, int e = 0, int l = 0) {
    begin = b;
    end = e;
    label = l;
  }

  void Clear() {
    gauss.Clear();
  }

  unsigned int Length()const {
    return end - begin;
  }

  bool IsDiff(const Seg &seg) const {
    if ((begin == seg.begin) && (end == seg.end) && (label == seg.label)) {
      return false;
    }
    return true;
  }
};

class SegCluster {
 public:
  SegCluster() {
    is_need_update_ivector_ = false;
    utt_num_ = 1;
  }

  void Clear() {
    buf_.clear();
    gauss_.Clear();
    is_need_update_ivector_ = true;

  }

  void Add(const Seg &seg) {
    buf_.push_back(seg);
    is_need_update_ivector_ = true;
  }

  void Add(const SegCluster &cluster) {
    for (int i = 0; i < cluster.Size(); ++i) {
      const Seg &seg = cluster.GetSeg(i);
      buf_.push_back(seg);
    }
    is_need_update_ivector_ = true;
  }

  const Seg &GetSeg(unsigned int index) const {
    idec::IDEC_ASSERT(index < buf_.size());
    return buf_[index];
  }

  Seg &GetSeg(unsigned int index) {
    idec::IDEC_ASSERT(index < buf_.size());
    return buf_[index];
  }

  void Remove(unsigned int index) {
    idec::IDEC_ASSERT(index < buf_.size());
    vector<Seg>::iterator iter = buf_.begin();
    buf_.erase(iter + index);
    is_need_update_ivector_ = true;
  }

  void Merge(const Seg &seg) {
    int left_pos = -1, right_pos = -1;
    // look for merge
    for (unsigned int i = 0; i < buf_.size(); i++) {
      if (!seg.IsDiff(buf_[i])) {
        idec::IDEC_WARN << "Seg already in cluster!";
        return;
      }

      if (seg.label != buf_[i].label) {
        continue;
      }

      if (seg.begin == buf_[i].end + 1) {
        left_pos = i;
      }

      if (seg.end + 1 == buf_[i].begin) {
        right_pos = i;
      }
    }

    if ((right_pos != -1) && (left_pos != -1)) {
      unsigned int new_begin, new_end, pos;
      new_begin = std::min(buf_[left_pos].begin, buf_[right_pos].begin);
      new_end = std::max(buf_[right_pos].end, buf_[left_pos].end);
      Remove(std::max(left_pos, right_pos));
      pos = std::min(left_pos, right_pos);
      buf_[pos].begin = new_begin;
      buf_[pos].end = new_end;
    } else if ((right_pos == -1) && (left_pos != -1)) {
      buf_[left_pos].end = seg.end;
    } else if ((right_pos != -1) && (left_pos == -1)) {
      buf_[right_pos].begin = seg.begin;
    } else {
      buf_.push_back(seg);
    }

    is_need_update_ivector_ = true;
  }

  void Merge(const SegCluster &cluster) {
    for (int i = 0; i < cluster.Size(); ++i) {
      const Seg &seg = cluster.GetSeg(i);
      this->Merge(seg);
    }
  }

  unsigned int Length() const {
    unsigned int total_length = 0;
    for (int i = 0; i < buf_.size(); ++i) {
      total_length += buf_[i].Length();
    }
    return total_length;
  }

  unsigned int Size() const {
    return buf_.size();
  }

  const FullGaussian *Gaussian() const {
    return &gauss_;
  }

  FullGaussian *Gaussian() {
    return &gauss_;
  }

  const DoubleVector &Ivector() const {
    return ivector_ ;
  }

  void SetIvector(const DoubleVector &ivector) {
    ivector_ = ivector;
    is_need_update_ivector_ = false;
  }

  void SetGaussian(const FullGaussian &gauss) {
    gauss_ = gauss;
  }

  bool IsDiff(const SegCluster &cluster) const {
    if (this->Size() != cluster.Size()) {
      return true;
    }

    for (int i = 0; i < this->Size(); ++i) {
      if (buf_[i].IsDiff(cluster.GetSeg(i))) {
        return true;
      }
    }
    return false;
  }

  void SetLabel(const int label) {
    label_ = label;
    for (int i = 0; i < buf_.size(); ++i) {
      buf_[i].label = label;
    }
  }

  int Label() const {
    return label_;
  }

  bool IsIvectorNeedUpdate() const {
    return is_need_update_ivector_;
  }

 public:
  int utt_num_;
 private:
  int label_;
  vector<Seg> buf_;
  FullGaussian gauss_;
  DoubleVector ivector_;
  bool is_need_update_ivector_;
};
}

#endif
