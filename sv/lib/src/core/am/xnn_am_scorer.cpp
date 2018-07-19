#include <map>
#include <string>
#include "am/xnn_am_scorer.h"

namespace idec {
using namespace std;
XNNAcousticModelScorer::XNNAcousticModelScorer(const XNNAcousticModelScorerOpt &opt, xnnNet *net) : opts_(opt) {
  net_ = net;
  if (opt.lazy_evaluation) {
    SetLazyMode();
  }
  // sync mode
  if (opts_.input_block_size == opts_.output_block_size) {
    evaluator_ = new xnnAmEvaluator(*net_, opts_.ac_scale, opts_.output_block_size,
                                    0, opts_.input_block_size, opts_.output_block_size);
  } else { // async mode for blstm
    evaluator_ = new xnnAmEvaluator(*net_, opts_.ac_scale,
                                    opt.lazy_evaluation ? 0 :
                                    opts_.output_block_size, // for lazy, we make the whole network in the same block with dnn
                                    0, opts_.input_block_size, opts_.output_block_size);
  }
}

XNNAcousticModelScorer::~XNNAcousticModelScorer() {
  if (evaluator_ != NULL)
    delete evaluator_;
}

void XNNAcousticModelScorer::PushFeatures(int start_frame, const xnnFloatRuntimeMatrix &feat) {
  evaluator_->pushFeatures(start_frame, feat);
}

float XNNAcousticModelScorer::GetFrameScore(int frame, int pdf_id) {
  if (lazy_mode_) {
    return evaluator_->logLikelihood_lazy(frame, pdf_id);
  } else {
    return evaluator_->logLikelihood(frame, pdf_id);
  }
}

float *XNNAcousticModelScorer::GetFrameScores(int frame) {
  return evaluator_->logLikelihood(frame);
}
};
