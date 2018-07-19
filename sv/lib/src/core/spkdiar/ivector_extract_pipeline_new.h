#ifndef _IVECTOR_EXTRACTOR_PIPELINE_H_
#define _IVECTOR_EXTRACTOR_PIPELINE_H_

#include "base/log_message.h"
#include "diag_gmm.h"
#include "posterior.h"
#include "ivector_estimator.h"
#include "resource_manager.h"

class IvectorExtractPipeline {
 public:
  IvectorExtractPipeline(const ResourceManager *res):opt_
    (res->IvectorConfigOption()) {
    res_ = res;
    utt_stats_ = UtteranceStats(res_->GaussNum(), res_->FeatDim(), false);
    verbose_mode_ = opt_.verbose_mode;
  }

  ~IvectorExtractPipeline() {}
  void ExtractIvector(const DoubleMatrix &feats, DoubleVector &ivector);
  int ExtractIvector(const UtteranceStats &utt_stats,
                     DoubleVector &ivector) const;
  void ExtractIvector(DoubleVector &ivector);
  int AccumulateStats(const DoubleMatrix &feats);
  const UtteranceStats &UttStats() {return utt_stats_;}
  void Clear() { utt_stats_.Clear(); }
 private:
  int DiagGmmgSelect(const DoubleMatrix &feats, const DiagGmm &gmm,
                     int num_gselect, vector<vector<int> > &gselect) const;
  int FullGmmgSelectToPost(const DoubleMatrix &feats, const FullGmm &fgmm,
                           Posterior &post, vector<vector<int> > &gselect, double min_post) const;
  int ScalePost(const DoubleMatrix &feats, Posterior &post) const;

 private:

  bool verbose_mode_;
  const IvectorExtractOptions &opt_;
  const ResourceManager *res_;
  UtteranceStats utt_stats_;
};


#endif // !_ACC_STATES_H_
