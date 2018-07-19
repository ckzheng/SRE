#include "am/hmm.h"
#include <cstdlib>
#include <iostream>
#include <string>
#include "base/log_message.h"
#include "util/io_base.h"

namespace idec {

void PhysicalHmm::Write(std::ostream &os, bool binary) const {
  if (!binary) {
    IDEC_ERROR << "cannot support write LogicalHmm in text format";
    exit(-1);
  }
  int32 state_num = static_cast<int32>(states_.size());
  IOBase::Write(os, state_num);
  for (size_t i = 0; i < states_.size(); i++) {
    IOBase::Write(os, states_[i].pdf_id);
    IOBase::Write(os, states_[i].trans_ids);
  }
}

void PhysicalHmm::Read(std::istream &is, bool binary) {
  int32 state_number = 0;
  IOBase::Read(is, &state_number);
  states_.resize(state_number);
  for (size_t i = 0; i < states_.size(); i++) {
    IOBase::Read(is, &states_[i].pdf_id);
    IOBase::Read(is, &states_[i].trans_ids);
  }
}

std::string PhysicalHmm::ToHashKeyPhysicalModel() {
  std::string key = "";
  for (size_t i = 0; i < states_.size(); i++) {
    if (i != 0)
      key += "_";
    key += std::to_string(states_[i].pdf_id);
  }
  return key;
}

}  // namespace idec

