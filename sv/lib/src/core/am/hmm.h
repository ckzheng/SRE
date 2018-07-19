#ifndef ASR_DECODER_SRC_CORE_AM_HMM_H_
#define ASR_DECODER_SRC_CORE_AM_HMM_H_

#include <vector>
#include <string>
#include <iostream>
#include "base/idec_types.h"

namespace idec {

static const int32 kNoTranId = -1;
// kNoPdf is used where pdf_class or pdf would be used, to indicate,
// none is there.
static const int32 kNoPdf = -1;

class PhysicalHmm {
 public:
  struct HmmState {
    int32 pdf_id;
    std::vector<int32> trans_ids;
    explicit HmmState(int32 p) : pdf_id(p) { }
    bool operator == (const HmmState &other) const {
      return (pdf_id == other.pdf_id && trans_ids == other.trans_ids);
    }
    HmmState() : pdf_id() { }
  };
  std::string ToHashKeyPhysicalModel();
  void Read(std::istream &is, bool binary);
  void Write(std::ostream &os, bool binary) const;
  const std::vector<HmmState> &GetStates() const { return states_; }
  void SetStates(std::vector<HmmState> &states) { states_ = states; }
  PhysicalHmm() {}
  // Copy constructor
  PhysicalHmm(const PhysicalHmm &other) : states_(other.states_) { }
  bool operator == (const PhysicalHmm &other) const {
    return states_ == other.states_;
  }

 private:
  std::vector<HmmState> states_;
};

}  // end namespace idec

#endif  // ASR_DECODER_SRC_CORE_AM_HMM_H_

