#ifndef ASYNC_AM_SCORER_H
#define ASYNC_AM_SCORER_H

#include <map>
#include <string>
#include <deque>
#include <vector>
#include "base/idec_types.h"
#include "lex/phone_set.h"
#include "am/hmm.h"
#include "am/xnn_runtime.h"
#include "am/am_scorer.h"
#include "am/xnn_net.h"
#include "util/thread.h"


namespace idec {

// the observation model part of acoustic model
enum AsyncModelScorerState {
  kAsyncModelScorerStateIdle = 0,
  kAsyncModelScorerStateEvaluating = 1
};

class BatchDispatchXnnEvaluator;
class SharedBatchDispatchXnnEvaluator;
class AsyncAcousticModelScorer :public AcousticModelScorer {
 public:
  AsyncAcousticModelScorer(xnnNet *net) :net_(net) {};
  virtual ~AsyncAcousticModelScorer() {};

  // push in an array of nFrames data into the scorer
  // which span time [startFrame, startFrame + nFrame)
  virtual void    PushFeatures(int start_frame,
                               const xnnFloatRuntimeMatrix &feat) {
    xnnFloatRuntimeMatrix *feat_copy = new xnnFloatRuntimeMatrix(feat.NumRows(),
        feat.NumCols());
    size_t  num_rows_  = feat_copy->NumRows();
    size_t  num_cols_ = feat_copy->NumCols();
    for (size_t col = 0; col < num_cols_; ++col) {
      memcpy(feat_copy->Col(col), feat.Col(col), sizeof(float) * num_rows_);
    }

    // thread-safe push_back of the feature buffer
    thread::lock_guard<thread::recursive_mutex> lock(feat_mutex_);
    FeaBuftElem feat_elem(feat_copy, start_frame, (int)feat_copy->NumCols());
    feat_buf_.push_back(feat_elem);
  }

  virtual size_t  NumPdfs() const { return net_->uDim(); };

  // get the score of (frame, pdfId)
  virtual float   GetFrameScore(int frame, int pdf_id) {
    // sanity to make sure it is in the available score range
    IDEC_ASSERT(score_buf_.size() > 0);
    int32 start_frame = score_buf_.front().start_frame;
    IDEC_ASSERT(frame >= start_frame);
    return (score_buf_.front().score->Col(frame - start_frame))[pdf_id];
  }

  virtual float  *GetFrameScores(int frame) {
    // sanity to make sure it is in the available score range
    IDEC_ASSERT(score_buf_.size() > 0);
    int32 start_frame = score_buf_.front().start_frame;
    IDEC_ASSERT(frame >= start_frame);
    return (score_buf_.front().score->Col(frame - start_frame));
  }

  // functions to do when start a utterances, e.g. reset the cache
  virtual int     BeginUtterance() {
    return 0;
  }

  // functions to do when end a utterances
  virtual int     EndUtterance() {
    return 0;
  }

  // [start_frame, start_frame + num_frame)
  virtual void    AvailableScoreRange(int start_frame, int num_frame) {
  }


  // drop the score range
  virtual void    DropScoreRange(int start_frame, int num_frame) {
    if (score_buf_.size() == 0) {
      return;
    }
    if (score_buf_.front().start_frame != start_frame
        || num_frame != score_buf_.front().num_frame) {
      return;
    }
    delete score_buf_.front().score;

    score_buf_.pop_front();

  }

 protected:
  friend class BatchDispatchXnnEvaluator;

  // one element in feature buffer
  // it a feature spanning [start_frame, start_fram + number_fram)
  struct FeaBuftElem {
   public:
    FeaBuftElem(xnnFloatRuntimeMatrix *_feat,
                int                   _start_frame,
                int                   _num_frame) {
      feat = _feat;
      _start_frame = start_frame;
      _num_frame = num_frame;
    }
    xnnFloatRuntimeMatrix *feat;
    int                   start_frame;
    int                   num_frame;
  };

  //
  struct ScoreBufElem {
    xnnFloatRuntimeMatrix *score;
    int                   start_frame;
    int                   num_frame;
  };

  // feature buffer
  std::deque<FeaBuftElem>            feat_buf_;
  thread::recursive_mutex            feat_mutex_;


  // score buffer
  std::deque<ScoreBufElem>           score_buf_;
  thread::recursive_mutex            score_mutex_;



  AsyncModelScorerState              state_;      // the state of the model score
  xnnNet                             *net_;       // the neural network

  void SetState(AsyncModelScorerState state) {
    state_ = state;
  }
  AsyncModelScorerState GetState() { return state_; }
};

class BatchDispatchAsyncAcousticModelScorer :public AsyncAcousticModelScorer {
 public:
  BatchDispatchAsyncAcousticModelScorer(xnnNet *net);
  virtual ~BatchDispatchAsyncAcousticModelScorer();
};

// the actual xnn evaluator + a thread
class BatchDispatchXnnEvaluator {
 public:
  BatchDispatchXnnEvaluator(xnnNet *net) {};
  void Start();
  void Stop();
  void RunEvaluator() {
    while (true) {

      // collect the feature into batches
      for (size_t i = 0; i < registed_async_scorers.size(); i++) {
        if (registed_async_scorers[i]->GetState() == kAsyncModelScorerStateIdle) {
          // append feature to feat_buf_
        }
      }
      // feat_buf_ ==> score_buf_;
      ForwardPropogate();
      // write the score back into async_scorers
    }
  }

  void ForwardPropogate() {
  }

  void RegisterScorer(BatchDispatchAsyncAcousticModelScorer *scorer) {
    registed_async_scorers.push_back(scorer);
  }

  //
  std::vector<BatchDispatchAsyncAcousticModelScorer *> registed_async_scorers;
  xnnFloatRuntimeMatrix               feat_buf_;
  xnnFloatRuntimeMatrix               score_buf_;
  xnnNet                              *net_;
  std::vector<xnnFloatRuntimeMatrix>  activations_;
};


// the thread-pool of xnn evaluators
// it will be a thread-safe singleton
class SharedBatchDispatchXnnEvaluator {
 public:
  static SharedBatchDispatchXnnEvaluator *GetInstance(size_t num_evaluator,
      xnnNet *net) {
    if (0 == instance_.get()) {
      if (0 == instance_.get()) {
        instance_.reset(new SharedBatchDispatchXnnEvaluator(num_evaluator, net));
      }
    }
    return instance_.get();
  }

  SharedBatchDispatchXnnEvaluator(size_t num_evaluator, xnnNet *net) {
    evaluators_.resize(num_evaluator);
    for (size_t i = 0; i < num_evaluator; i++) {
      evaluators_[i] = new BatchDispatchXnnEvaluator(net);
    }
  }

  virtual ~SharedBatchDispatchXnnEvaluator() {
    for (size_t i = 0; i < evaluators_.size(); i++) {
      delete (evaluators_[i]);
    }
    evaluators_.clear();
  }

  void RegisterAcousticModelScorer(BatchDispatchAsyncAcousticModelScorer
                                   *scorer) {
    for (size_t i = 0; i < evaluators_.size(); i++) {
      evaluators_[i]->RegisterScorer(scorer);
    }
  }

  // the thread pool of xnn-evaluators
  std::vector<BatchDispatchXnnEvaluator *>               evaluators_;
  static std::auto_ptr<SharedBatchDispatchXnnEvaluator> instance_;
};


};

#endif
