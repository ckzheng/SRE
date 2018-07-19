#include "am/async_am_scorer.h"

namespace idec {

std::auto_ptr<SharedBatchDispatchXnnEvaluator>SharedBatchDispatchXnnEvaluator::instance_;

BatchDispatchAsyncAcousticModelScorer::BatchDispatchAsyncAcousticModelScorer(xnnNet *net) :AsyncAcousticModelScorer(net) {
  // register it with the shared batch-dispatch
  SharedBatchDispatchXnnEvaluator::GetInstance(10, net)->RegisterAcousticModelScorer(this);
}

BatchDispatchAsyncAcousticModelScorer::~BatchDispatchAsyncAcousticModelScorer() {}

};

