#ifndef ASR_DECODER_SRC_CORE_AM_KALDI_AM_H_
#define ASR_DECODER_SRC_CORE_AM_KALDI_AM_H_

#include <stdexcept>  // std::exception
#include <vector>
#include "am/acoustic_model.h"
#include "am/hmm.h"
#include "kaldi/context-dep.h"
#include "kaldi/transition-model.h"

namespace idec {

// wrapper of kalid format acoustic model
class KaldiAM : public AcousticModel {
 public:
  KaldiAM();
  virtual ~KaldiAM();
  virtual PhysicalHmm & GetHMM(const std::vector<PhoneId>&phoneSeq,
                               PhysicalHmm &hmm);
  virtual int32 GetFeatDim() const {
    throw std::runtime_error("not implemented");
  }

  // io functions
  virtual void ReadBinary(const char *file_name);
  virtual void Output(const char *file_name, bool output_binary);

  //) interface for decision tree
  virtual int32 ContextSize() {
    return ctx_dep_->ContextWidth()/2;
  }  // 1 for triphone
  virtual int32 CentralPhonePosition() const {
    return ctx_dep_->CentralPosition();
  }
  virtual int32 NumPdfs() const { return num_pdfs_;}
  virtual int32 NumPhones() const {
    return trans_model_->NumPhones();
  }
  // interface for transition ids
  virtual int32 SelfLoopOfSrcHmmState(int32 trans_id) const {
    return trans_model_->SelfLoopOf(TransitionIdToTransitionState(trans_id));
  }
  virtual int32 TransitionIdToPhone(int32 trans_id) const {
    return trans_model_->TransitionIdToPhone(trans_id);
  }
  virtual int32 TransitionIdToPdfClass(int32 trans_id) const {
    throw std::runtime_error("not implemented");
  }
  virtual int32 TransitionIdToHmmState(int32 trans_id) const {
    return trans_model_->TransitionIdToHmmState(trans_id);
  }
  virtual int32 TransitionIdToTransitionState(int32 trans_id) const {
    return trans_model_->TransitionIdToTransitionState(trans_id);
  }
  virtual int32 TransitionIdToTransitionIndex(int32 trans_id) const {
    return trans_model_->TransitionIdToTransitionIndex(trans_id);
  }
  virtual inline int32 TransitionIdToPdf(int32 trans_id) const {
    return trans_model_->TransitionIdToPdf(trans_id);
  }
  virtual bool IsSelfLoop(int32 trans_id) const {
    return trans_model_->IsSelfLoop(trans_id);
  }
  virtual bool IsLeftRight(int32 trans_id) const;
  virtual bool NotBackSkip(int32 trans_id) const {
    throw std::runtime_error("not implemented");
  }
  virtual inline int32 NumTransitionIds() const {
    return trans_model_->NumTransitionIds();
  }  // Returns the total number of transition-ids (note, these are one-based).
  virtual BaseFloat GetTransitionProb(int32 trans_id) const {
    throw std::runtime_error("not implemented");
  }
  virtual BaseFloat GetTransitionLogProb(int32 trans_id) const {
    return trans_model_->GetTransitionLogProb(trans_id);
  }
  virtual int32 GetMaxEmittingState() {
    return max_num_states_ - 1;
  }  // we assume only the last state is the non-emitting state

 public:
  static bool IsValidKaldiAm(const char *file_name);

 private:
  void ReadBinaryTransModel(const char *model_file);
  void ReadBinaryCtxDep(const char *tree_file);

  void OutputBinaryTransModel(const char *model_file);
  void OutputBinaryCtxDep(const char *tree_file);

 private:
  kaldi::TransitionModel *trans_model_;
  kaldi::ContextDependency *ctx_dep_;
  int32 max_num_states_;
  int32 num_pdfs_;
};

}  // namespace idec

#endif  // ASR_DECODER_SRC_CORE_AM_KALDI_AM_H_

