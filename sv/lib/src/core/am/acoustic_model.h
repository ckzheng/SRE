#ifndef ABSTRACT_AM_H
#define ABSTRACT_AM_H

#include <map>
#include <string>
#include "base/idec_types.h"
#include "lex/phone_set.h"
#include "am/hmm.h"


namespace idec {

// the structural information part of acoustic model, the observation model is not here
// think it as the transition model + decision tree in the Kaldi
// or the MMF file + decision tree w/o the GMM definition
class AcousticModel {
 public:
  virtual PhysicalHmm &GetHMM(const std::vector<PhoneId> &phn_ctx_win, PhysicalHmm &hmm) = 0;
  virtual int32 GetFeatDim() const = 0;

  //) io functions
  virtual void ReadBinary(const char *file_name) = 0;
  virtual void Output(const char *file_name, bool binary) = 0;

  //) interface for decision tree
  virtual int32 ContextSize() = 0; //symmetric context window only, 1 for triphone
  virtual int32 CentralPhonePosition() const = 0; //central phone position, 1 for triphone
  virtual int32 NumPdfs() const = 0;
  virtual int32 NumPhones() const = 0;

  // interface for transition ids
  virtual int32 SelfLoopOfSrcHmmState(int32 trans_id) const = 0; // get the self-loop transition id of srcHMM state
  virtual int32 TransitionIdToPhone(int32 trans_id) const = 0;
  virtual int32 TransitionIdToPdfClass(int32 trans_id) const = 0;
  virtual int32 TransitionIdToHmmState(int32 trans_id) const = 0;
  virtual int32 TransitionIdToTransitionState(int32 trans_id) const = 0;
  virtual int32 TransitionIdToTransitionIndex(int32 trans_id) const = 0;
  virtual int32 TransitionIdToPdf(int32 trans_id) const = 0;
  virtual bool  IsSelfLoop(int32 trans_id) const = 0;  // return true if this trans_id corresponds to a self-loop.
  virtual bool  NotBackSkip(int32  trans_id) const = 0;
  virtual bool  IsLeftRight(int32  trans_id) const = 0;
  virtual int32 NumTransitionIds() const = 0;    /// Returns the total number of transition-ids (note, these are one-based).
  virtual BaseFloat GetTransitionProb(int32 trans_id) const = 0;
  virtual BaseFloat GetTransitionLogProb(int32 trans_id) const = 0;

  // other stats
  virtual int32 GetMaxEmittingState() = 0;

  AcousticModel() {};
  virtual ~AcousticModel() {};
};

// factory class to create Acoustic models
class AcousticModelMaker {
 public:
  static AcousticModel *MakeFromBinaryFile(const std::string &file_name);
};

};

#endif
