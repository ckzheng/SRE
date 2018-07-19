#include "am/kaldi_am.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include "base/log_message.h"
#include "util/file_input.h"
#include "util/file_output.h"
#include "util/dir_utils.h"

namespace idec {

using namespace std;

KaldiAM::KaldiAM() {
  max_num_states_ = 0;
  trans_model_ = NULL;
  ctx_dep_ = NULL;
  num_pdfs_ = 0;
}

KaldiAM::~KaldiAM() {
  IDEC_DELETE(trans_model_);
  IDEC_DELETE(ctx_dep_);
}

bool KaldiAM::IsValidKaldiAm(const char *file_name) {
  string tree_file = string(file_name) + ".tree";
  string model_file = string(file_name) + ".mdl";
  if (File::IsReadable(tree_file.c_str()) ||
      File::IsReadable(model_file.c_str())) {
    return true;
  } else {
    return false;
  }
}

void KaldiAM::ReadBinary(const char *file_name) {
  string tree_file = string(file_name) + ".tree";
  string model_file = string(file_name) + ".mdl";

  ReadBinaryCtxDep(tree_file.c_str());
  ReadBinaryTransModel(model_file.c_str());

  // now scan the topology for the max-num-states
  max_num_states_ = 0;
  const kaldi::HmmTopology &topo = trans_model_->GetTopo();
  for (int32 phone_id = 1; phone_id <= trans_model_->NumPhones(); phone_id++) {
    const kaldi::HmmTopology::TopologyEntry &entry =
      topo.TopologyForPhone(phone_id);
    max_num_states_ = std::max(static_cast<int32>(entry.size()),
                               max_num_states_);
  }

  num_pdfs_ = ctx_dep_->NumPdfs();
}

void KaldiAM::Output(const char *file_name, bool output_binary) {
  string tree_file = string(file_name) + ".tree";
  string model_file = string(file_name) + ".mdl";

  OutputBinaryCtxDep(tree_file.c_str());
  OutputBinaryTransModel(model_file.c_str());
}

void KaldiAM::ReadBinaryTransModel(const char *model_file) {
  FileInput fi(model_file, true);
  fi.Open();
  std::istream & ki = fi.GetStream();

  // make sure the input file is binary
  if (ki.peek() != '\0')
    IDEC_ERROR << "only support kaldi binary format";
  ki.get();
  if (ki.peek() != 'B')
    IDEC_ERROR << "only support kaldi binary format";
  ki.get();

  IDEC_DELETE(trans_model_);
  trans_model_ = new kaldi::TransitionModel();
  trans_model_->Read(ki, true);
  fi.Close();
}

void KaldiAM::ReadBinaryCtxDep(const char *tree_file) {
  FileInput fi(tree_file, true);
  fi.Open();
  std::istream & ki = fi.GetStream();

  // make sure the input file is binary
  if (ki.peek() != '\0')
    IDEC_ERROR << "only support kaldi binary format";
  ki.get();
  if (ki.peek() != 'B')
    IDEC_ERROR << "only support kaldi binary format";
  ki.get();

  IDEC_DELETE(ctx_dep_);
  ctx_dep_ = new kaldi::ContextDependency();
  ctx_dep_->Read(ki, true);
  fi.Close();
}

void KaldiAM::OutputBinaryTransModel(const char *model_file) {
  FileOutput fo(model_file, true);
  fo.Open();
  std::ostream & ko = fo.GetStream();
  const char*mark = "\0B";
  ko.write(mark, 2);
  trans_model_->Write(ko, true);
  fo.Close();
}

void KaldiAM::OutputBinaryCtxDep(const char *tree_file) {
  FileOutput fo(tree_file, true);
  fo.Open();
  std::ostream & ko = fo.GetStream();
  const char*mark = "\0B";
  ko.write(mark, 2);
  ctx_dep_->Write(ko, true);
  fo.Close();
}

PhysicalHmm &KaldiAM::GetHMM(const std::vector<PhoneId> &phone_seq,
                             PhysicalHmm &hmm) {
  std::vector<int32>phone_window(phone_seq.size());
  for (size_t p = 0; p < phone_seq.size(); p++) {
    phone_window[p] = phone_seq[p];
  }

  if (static_cast<int32>(phone_window.size()) != ctx_dep_->ContextWidth())
    KALDI_ERR << "Context size mismatch, ilabel-info [from context FST is "
      << (phone_window.size()) << ", context-dependency object "
      "expects " << (ctx_dep_->ContextWidth());

  int P = ctx_dep_->CentralPosition();
  int32 phone = phone_window[P];

  const kaldi::HmmTopology &topo = trans_model_->GetTopo();
  const kaldi::HmmTopology::TopologyEntry &entry = topo.TopologyForPhone(phone);

  // vector of the pdfs, indexed by pdf-class (pdf-classes must start from zero
  // and be contiguous).
  std::vector<int32> pdfs(topo.NumPdfClasses(phone));
  for (int32 pdf_class = 0;
       pdf_class < static_cast<int32>(pdfs.size());
       pdf_class++) {
    if (!ctx_dep_->Compute(phone_window, pdf_class, &(pdfs[pdf_class]))) {
      std::ostringstream ctx_ss;
      for (size_t i = 0; i < phone_window.size(); i++)
        ctx_ss << phone_window[i] << ' ';
      KALDI_ERR << "GetHmmAsFst: context-dependency object could not produce "
        << "an answer: pdf-class = " << pdf_class << " ctx-window = "
        << ctx_ss.str() << ".  This probably points "
        "to either a coding error in some graph-building process, "
        "a mismatch of topology with context-dependency object, the "
        "wrong FST being passed on a command-line, or something of "
        " that general nature.";
    }
  }

  std::vector<PhysicalHmm::HmmState> states;
  states.resize(static_cast<int32>(entry.size()));
  for (int32 hmm_state = 0;
       hmm_state < static_cast<int32>(entry.size());
       hmm_state++) {
    int32 pdf_class = entry[hmm_state].pdf_class, pdf;
    if (pdf_class == kNoPdf) {
      pdf = kNoPdf;  // nonemitting state.
    } else {
      KALDI_ASSERT(pdf_class < static_cast<int32>(pdfs.size()));
      pdf = pdfs[pdf_class];
    }
    states[hmm_state].pdf_id = pdf;
    states[hmm_state].trans_ids.resize(entry[hmm_state].transitions.size());
    int32 trans_idx;
    for (trans_idx = 0;
         trans_idx < static_cast<int32>(entry[hmm_state].transitions.size());
         trans_idx++) {
      int32 trans_id;
      // int32 dest_state = entry[hmm_state].transitions[trans_idx].first;
      if (pdf_class == kNoPdf) {
        // no pdf, hence non-estimated probability.
        // [would not happen with normal topology] .
        // There is no transition-state involved in this case.
        // log_prob = log(entry[hmm_state].transitions[trans_idx].second);
        trans_id = 0;
      } else {  // normal probability.
        int32 trans_state =
          trans_model_->TripleToTransitionState(phone, hmm_state, pdf);
        trans_id =
          trans_model_->PairToTransitionId(trans_state, trans_idx);
        // log_prob =
        // trans_model_->GetTransitionLogProbIgnoringSelfLoops(trans_id);
        // log_prob is a negative number (or zero)...
      }
      states[hmm_state].trans_ids[trans_idx] = trans_id;
    }
  }
  hmm.SetStates(states);
  return hmm;
}

bool KaldiAM::IsLeftRight(int32 trans_id) const {
  using namespace kaldi;
  int32 phone = trans_model_->TransitionIdToPhone(trans_id);
  const HmmTopology &topo = trans_model_->GetTopo();
  const HmmTopology::TopologyEntry &entry = topo.TopologyForPhone(phone);
  int32 src_state = trans_model_->TransitionIdToHmmState(trans_id);
  int32 trans_idx = trans_model_->TransitionIdToTransitionIndex(trans_id);
  int32 dest_state = entry[src_state].transitions[trans_idx].first;
  return (dest_state == src_state + 1);
}

}  // namespace idec

