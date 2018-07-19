// hmm/transition-model.cc

// Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
//        Johns Hopkins University (author: Guoguo Chen)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <vector>
#include "kaldi/transition-model.h"
#include "kaldi/context-dep.h"
#include "kaldi/kaldi-error.h"

namespace idec {
namespace kaldi {

void TransitionModel::ComputeTriples(const ContextDependency &ctx_dep) {
  const std::vector<int32> &phones = topo_.GetPhones();
  std::vector<std::vector<std::pair<int32, int32> > > pdf_info;
  KALDI_ASSERT(!phones.empty());
  std::vector<int32> num_pdf_classes( 1 + *std::max_element(phones.begin(),
                                      phones.end()), -1);
  for (size_t i = 0; i < phones.size(); i++)
    num_pdf_classes[phones[i]] = topo_.NumPdfClasses(phones[i]);
  ctx_dep.GetPdfInfo(phones, num_pdf_classes, &pdf_info);
  // pdf_info is list indexed by pdf of which (phone, pdf_class) it
  // can correspond to.

  std::map<std::pair<int32, int32>, std::vector<int32> > to_hmm_state_list;
  // to_hmm_state_list is a map from (phone, pdf_class) to the list
  // of hmm-states in the HMM for that phone that that (phone, pdf-class)
  // can correspond to.
  for (size_t i = 0; i < phones.size(); i++) {  // setting up to_hmm_state_list.
    int32 phone = phones[i];
    const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(phone);
    for (int32 j = 0; j < static_cast<int32>(entry.size());
         j++) {  // for each state...
      int32 pdf_class = entry[j].pdf_class;
      if (pdf_class != kNoPdf) {
        to_hmm_state_list[std::make_pair(phone, pdf_class)].push_back(j);
      }
    }
  }
  for (int32 pdf = 0; pdf < static_cast<int32>(pdf_info.size()); pdf++) {
    for (size_t j = 0; j < pdf_info[pdf].size(); j++) {
      int32 phone = pdf_info[pdf][j].first,
            pdf_class = pdf_info[pdf][j].second;
      const std::vector<int32> &state_vec = to_hmm_state_list[std::make_pair(phone,
                                            pdf_class)];
      KALDI_ASSERT(!state_vec.empty());
      // state_vec is a list of the possible HMM-states that emit this
      // pdf_class.
      for (size_t k = 0; k < state_vec.size(); k++) {
        int32 hmm_state = state_vec[k];
        triples_.push_back(Triple(phone, hmm_state, pdf));
      }
    }
  }

  // now triples_ is populated with all possible triples of (phone, hmm_state, pdf).
  std::sort(triples_.begin(), triples_.end());  // sort to enable reverse lookup.
  // this sorting defines the transition-ids.
}

void TransitionModel::ComputeDerived() {
  state2id_.resize(triples_.size()+2);  // indexed by transition-state, which
  // is one based, but also an entry for one past end of list.

  int32 cur_transition_id = 1;
  num_pdfs_ = 0;
  for (int32 tstate = 1;
       tstate <= static_cast<int32>(triples_.size()+1);  // not a typo.
       tstate++) {
    state2id_[tstate] = cur_transition_id;
    if (static_cast<size_t>(tstate) <= triples_.size()) {
      int32 phone = triples_[tstate-1].phone,
            hmm_state = triples_[tstate-1].hmm_state,
            pdf = triples_[tstate-1].pdf;
      num_pdfs_ = std::max(num_pdfs_, 1+pdf);
      const HmmTopology::HmmState &state = topo_.TopologyForPhone(phone)[hmm_state];
      int32 my_num_ids = static_cast<int32>(state.transitions.size());
      cur_transition_id += my_num_ids;  // # trans out of this state.
    }
  }

  id2state_.resize(
    cur_transition_id);   // cur_transition_id is #transition-ids+1.
  for (int32 tstate = 1; tstate <= static_cast<int32>(triples_.size()); tstate++)
    for (int32 tid = state2id_[tstate]; tid < state2id_[tstate+1]; tid++)
      id2state_[tid] = tstate;

}
void TransitionModel::InitializeProbs() {
  log_probs_.Resize(NumTransitionIds()
                    +1);  // one-based array, zeroth element empty.
  for (int32 trans_id = 1; trans_id <= NumTransitionIds(); trans_id++) {
    int32 trans_state = id2state_[trans_id];
    int32 trans_index = trans_id - state2id_[trans_state];
    const Triple &triple = triples_[trans_state-1];
    const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(triple.phone);
    KALDI_ASSERT(static_cast<size_t>(triple.hmm_state) < entry.size());
    BaseFloat prob = entry[triple.hmm_state].transitions[trans_index].second;
    if (prob <= 0.0)
      KALDI_ERR << "TransitionModel::InitializeProbs, zero "
                "probability [should remove that entry in the topology]";
    if (prob > 1.0)
      KALDI_WARN << "TransitionModel::InitializeProbs, prob greater than one.";
    log_probs_(trans_id) = log(prob);
  }
  ComputeDerivedOfProbs();
}

void TransitionModel::Check() const {
  KALDI_ASSERT(NumTransitionIds() != 0 && NumTransitionStates() != 0);
  {
    int32 sum = 0;
    for (int32 ts = 1; ts <= NumTransitionStates();
         ts++) sum += NumTransitionIndices(ts);
    KALDI_ASSERT(sum == NumTransitionIds());
  }
  for (int32 tid = 1; tid <= NumTransitionIds(); tid++) {
    int32 tstate = TransitionIdToTransitionState(tid),
          index = TransitionIdToTransitionIndex(tid);
    KALDI_ASSERT(tstate > 0 && tstate <=NumTransitionStates() && index >= 0);
    KALDI_ASSERT(tid == PairToTransitionId(tstate, index));
    int32 phone = TransitionStateToPhone(tstate),
          hmm_state = TransitionStateToHmmState(tstate),
          pdf = TransitionStateToPdf(tstate);
    KALDI_ASSERT(tstate == TripleToTransitionState(phone, hmm_state, pdf));
    KALDI_ASSERT(log_probs_(tid) <= 0.0
                 && log_probs_(tid) - log_probs_(tid) == 0.0);
    // checking finite and non-positive (and not out-of-bounds).
  }
}

TransitionModel::TransitionModel(const ContextDependency &ctx_dep,
                                 const HmmTopology &hmm_topo): topo_(hmm_topo) {
  // First thing is to get all possible triples.
  ComputeTriples(ctx_dep);
  ComputeDerived();
  InitializeProbs();
  Check();
}

int32 TransitionModel::TripleToTransitionState(int32 phone, int32 hmm_state,
    int32 pdf) const {
  Triple triple(phone, hmm_state, pdf);
  // Note: if this ever gets too expensive, which is unlikely, we can refactor
  // this code to sort first on pdf, and then index on pdf, so those
  // that have the same pdf are in a contiguous range.
  std::vector<Triple>::const_iterator iter =
    std::lower_bound(triples_.begin(), triples_.end(), triple);
  if (iter == triples_.end() || !(*iter == triple)) {
    KALDI_ERR << "TransitionModel::TripleToTransitionState, triple not found."
              << " (incompatible tree and model?)";
  }
  // triples_ is indexed by transition_state-1, so add one.
  return static_cast<int32>((iter - triples_.begin())) + 1;
}


int32 TransitionModel::NumTransitionIndices(int32 trans_state) const {
  KALDI_ASSERT(static_cast<size_t>(trans_state) <= triples_.size());
  return static_cast<int32>(state2id_[trans_state+1]-state2id_[trans_state]);
}

int32 TransitionModel::TransitionIdToTransitionState(int32 trans_id) const {
  KALDI_ASSERT(trans_id != 0
               &&  static_cast<size_t>(trans_id) < id2state_.size());
  return id2state_[trans_id];
}

int32 TransitionModel::TransitionIdToTransitionIndex(int32 trans_id) const {
  KALDI_ASSERT(trans_id != 0
               && static_cast<size_t>(trans_id) < id2state_.size());
  return trans_id - state2id_[id2state_[trans_id]];
}

int32 TransitionModel::TransitionStateToPhone(int32 trans_state) const {
  KALDI_ASSERT(static_cast<size_t>(trans_state) <= triples_.size());
  return triples_[trans_state-1].phone;
}

int32 TransitionModel::TransitionStateToPdf(int32 trans_state) const {
  KALDI_ASSERT(static_cast<size_t>(trans_state) <= triples_.size());
  return triples_[trans_state-1].pdf;
}

int32 TransitionModel::TransitionStateToHmmState(int32 trans_state) const {
  KALDI_ASSERT(static_cast<size_t>(trans_state) <= triples_.size());
  return triples_[trans_state-1].hmm_state;
}

int32 TransitionModel::PairToTransitionId(int32 trans_state,
    int32 trans_index) const {
  KALDI_ASSERT(static_cast<size_t>(trans_state) <= triples_.size());
  KALDI_ASSERT(trans_index < state2id_[trans_state+1] - state2id_[trans_state]);
  return state2id_[trans_state] + trans_index;
}

int32 TransitionModel::NumPhones() const {
  int32 num_trans_state = triples_.size();
  int32 max_phone_id = 0;
  for (int32 i = 0; i < num_trans_state; ++i) {
    if (triples_[i].phone > max_phone_id)
      max_phone_id = triples_[i].phone;
  }
  return max_phone_id;
}


bool TransitionModel::IsFinal(int32 trans_id) const {
  KALDI_ASSERT(static_cast<size_t>(trans_id) < id2state_.size());
  int32 trans_state = id2state_[trans_id];
  int32 trans_index = trans_id - state2id_[trans_state];
  const Triple &triple = triples_[trans_state-1];
  const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(triple.phone);
  KALDI_ASSERT(static_cast<size_t>(triple.hmm_state) < entry.size());
  KALDI_ASSERT(static_cast<size_t>(triple.hmm_state) < entry.size());
  KALDI_ASSERT(static_cast<size_t>(trans_index) <
               entry[triple.hmm_state].transitions.size());
  // return true if the transition goes to the final state of the
  // topology entry.
  return (entry[triple.hmm_state].transitions[trans_index].first + 1 ==
          static_cast<int32>(entry.size()));
}


bool TransitionModel::IsSelfLoop(int32 trans_id) const {
  KALDI_ASSERT(static_cast<size_t>(trans_id) < id2state_.size());
  int32 trans_state = id2state_[trans_id];
  int32 trans_index = trans_id - state2id_[trans_state];
  const Triple &triple = triples_[trans_state-1];
  int32 phone = triple.phone, hmm_state = triple.hmm_state;
  const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(phone);
  KALDI_ASSERT(static_cast<size_t>(hmm_state) < entry.size());
  return (static_cast<size_t>(trans_index) < entry[hmm_state].transitions.size()
          && entry[hmm_state].transitions[trans_index].first == hmm_state);
}

int32 TransitionModel::SelfLoopOf(int32 trans_state)
const {  // returns the self-loop transition-id,
  KALDI_ASSERT(static_cast<size_t>(trans_state-1) < triples_.size());
  const Triple &triple = triples_[trans_state-1];
  // or zero if does not exist.
  int32 phone = triple.phone, hmm_state = triple.hmm_state;
  const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(phone);
  KALDI_ASSERT(static_cast<size_t>(hmm_state) < entry.size());
  for (int32 trans_index = 0;
       trans_index < static_cast<int32>(entry[hmm_state].transitions.size());
       trans_index++)
    if (entry[hmm_state].transitions[trans_index].first == hmm_state)
      return PairToTransitionId(trans_state, trans_index);
  return 0;  // invalid transition id.
}

void TransitionModel::ComputeDerivedOfProbs() {
  non_self_loop_log_probs_.Resize(NumTransitionStates()
                                  +1);  // this array indexed
  //  by transition-state with nothing in zeroth element.
  for (int32 tstate = 1; tstate <= NumTransitionStates(); tstate++) {
    int32 tid = SelfLoopOf(tstate);
    if (tid == 0) {  // no self-loop
      non_self_loop_log_probs_(tstate) = 0.0;  // log(1.0)
    } else {
      BaseFloat self_loop_prob = exp(GetTransitionLogProb(tid)),
                non_self_loop_prob = 1.0 - self_loop_prob;
      if (non_self_loop_prob <= 0.0) {
        KALDI_WARN << "ComputeDerivedOfProbs(): non-self-loop prob is " <<
                   non_self_loop_prob;
        non_self_loop_prob = 1.0e-10;  // just so we can continue...
      }
      non_self_loop_log_probs_(tstate) = log(
                                           non_self_loop_prob);  // will be negative.
    }
  }
}

void TransitionModel::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<TransitionModel>");
  topo_.Read(is, binary);
  ExpectToken(is, binary, "<Triples>");
  int32 size;
  ReadBasicType(is, binary, &size);
  triples_.resize(size);
  for (int32 i = 0; i < size; i++) {
    ReadBasicType(is, binary, &(triples_[i].phone));
    ReadBasicType(is, binary, &(triples_[i].hmm_state));
    ReadBasicType(is, binary, &(triples_[i].pdf));
  }
  ExpectToken(is, binary, "</Triples>");
  ComputeDerived();
  ExpectToken(is, binary, "<LogProbs>");
  log_probs_.Read(is, binary);
  ExpectToken(is, binary, "</LogProbs>");
  ExpectToken(is, binary, "</TransitionModel>");
  ComputeDerivedOfProbs();
  Check();
}

void TransitionModel::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<TransitionModel>");
  if (!binary) os << "\n";
  topo_.Write(os, binary);
  WriteToken(os, binary, "<Triples>");
  WriteBasicType(os, binary, static_cast<int32>(triples_.size()));
  if (!binary) os << "\n";
  for (int32 i = 0; i < static_cast<int32> (triples_.size()); i++) {
    WriteBasicType(os, binary, triples_[i].phone);
    WriteBasicType(os, binary, triples_[i].hmm_state);
    WriteBasicType(os, binary, triples_[i].pdf);
    if (!binary) os << "\n";
  }
  WriteToken(os, binary, "</Triples>");
  if (!binary) os << "\n";
  WriteToken(os, binary, "<LogProbs>");
  if (!binary) os << "\n";
  log_probs_.Write(os, binary);
  WriteToken(os, binary, "</LogProbs>");
  if (!binary) os << "\n";
  WriteToken(os, binary, "</TransitionModel>");
  if (!binary) os << "\n";
}

BaseFloat TransitionModel::GetTransitionProb(int32 trans_id) const {
  return exp(log_probs_(trans_id));
}

BaseFloat TransitionModel::GetTransitionLogProb(int32 trans_id) const {
  return log_probs_(trans_id);
}

BaseFloat TransitionModel::GetNonSelfLoopLogProb(int32 trans_state) const {
  KALDI_ASSERT(trans_state != 0);
  return non_self_loop_log_probs_(trans_state);
}

BaseFloat TransitionModel::GetTransitionLogProbIgnoringSelfLoops(
  int32 trans_id) const {
  KALDI_ASSERT(trans_id != 0);
  KALDI_PARANOID_ASSERT(!IsSelfLoop(trans_id));
  return log_probs_(trans_id) - GetNonSelfLoopLogProb(
           TransitionIdToTransitionState(trans_id));
}


int32 TransitionModel::TransitionIdToPhone(int32 trans_id) const {
  KALDI_ASSERT(trans_id != 0
               && static_cast<size_t>(trans_id) < id2state_.size());
  int32 trans_state = id2state_[trans_id];
  return triples_[trans_state-1].phone;
}

int32 TransitionModel::TransitionIdToPdfClass(int32 trans_id) const {
  KALDI_ASSERT(trans_id != 0
               && static_cast<size_t>(trans_id) < id2state_.size());
  int32 trans_state = id2state_[trans_id];

  const Triple &t = triples_[trans_state-1];
  const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(t.phone);
  KALDI_ASSERT(static_cast<size_t>(t.hmm_state) < entry.size());
  return entry[t.hmm_state].pdf_class;
}


int32 TransitionModel::TransitionIdToHmmState(int32 trans_id) const {
  KALDI_ASSERT(trans_id != 0
               && static_cast<size_t>(trans_id) < id2state_.size());
  int32 trans_state = id2state_[trans_id];
  const Triple &t = triples_[trans_state-1];
  return t.hmm_state;
}


bool GetPdfsForPhones(const TransitionModel &trans_model,
                      const std::vector<int32> &phones,
                      std::vector<int32> *pdfs) {
  KALDI_ASSERT(IsSortedAndUniq(phones));
  KALDI_ASSERT(pdfs != NULL);
  pdfs->clear();
  for (int32 tstate = 1; tstate <= trans_model.NumTransitionStates(); tstate++) {
    if (std::binary_search(phones.begin(), phones.end(),
                           trans_model.TransitionStateToPhone(tstate)))
      pdfs->push_back(trans_model.TransitionStateToPdf(tstate));
  }
  SortAndUniq(pdfs);

  for (int32 tstate = 1; tstate <= trans_model.NumTransitionStates(); tstate++)
    if (std::binary_search(pdfs->begin(), pdfs->end(),
                           trans_model.TransitionStateToPdf(tstate))
        && !std::binary_search(phones.begin(), phones.end(),
                               trans_model.TransitionStateToPhone(tstate)))
      return false;
  return true;
}

bool GetPhonesForPdfs(const TransitionModel &trans_model,
                      const std::vector<int32> &pdfs,
                      std::vector<int32> *phones) {
  KALDI_ASSERT(IsSortedAndUniq(pdfs));
  KALDI_ASSERT(phones != NULL);
  phones->clear();
  for (int32 tstate = 1; tstate <= trans_model.NumTransitionStates(); tstate++) {
    if (std::binary_search(pdfs.begin(), pdfs.end(),
                           trans_model.TransitionStateToPdf(tstate)))
      phones->push_back(trans_model.TransitionStateToPhone(tstate));
  }
  SortAndUniq(phones);

  for (int32 tstate = 1; tstate <= trans_model.NumTransitionStates(); tstate++)
    if (std::binary_search(phones->begin(), phones->end(),
                           trans_model.TransitionStateToPhone(tstate))
        && !std::binary_search(pdfs.begin(), pdfs.end(),
                               trans_model.TransitionStateToPdf(tstate)))
      return false;
  return true;
}

} // End namespace kaldi
}
