#ifndef ASR_DECODER_SRC_CORE_AM_AM_SCORER_H_
#define ASR_DECODER_SRC_CORE_AM_AM_SCORER_H_

#include <map>
#include <string>
#include "base/idec_types.h"
#include "lex/phone_set.h"
#include "am/hmm.h"
#include "am/xnn_runtime.h"

namespace idec {

// the observation model part of acoustic model
// e.g. think it as the decode-able + AmDiagGmm in kaldi
class AcousticModelScorer {
 public:
  // number of pdfs
  virtual size_t NumPdfs() const = 0;

  // push in an array of nFrames data into the scorer
  // which span time [startFrame, startFrame + nFrame)
  virtual void PushFeatures(int start_frame,
                            const xnnFloatRuntimeMatrix &feat) = 0;

  // get the score of (frame, pdfId)
  virtual float GetFrameScore(int frame, int pdf_id) = 0;
  virtual float *GetFrameScores(int frame) = 0;

  // functions to do when start a utterances, e.g. reset the cache
  virtual int BeginUtterance() = 0;

  // functions to do when end a utterances
  virtual int EndUtterance() = 0;

  // implement the lazy evaluation mode
  void SetLazyMode() { lazy_mode_ = true; }
  bool IsLazyMode() const { return lazy_mode_;}

  AcousticModelScorer() { lazy_mode_ = false; }
  virtual ~AcousticModelScorer() {}

  bool lazy_mode_;
};
}  // namespace idec

#endif  // ASR_DECODER_SRC_CORE_AM_AM_SCORER_H_
