#include "ivector_extract_pipeline_new.h"
#include "base/time_utils.h"

void IvectorExtractPipeline::ExtractIvector(const DoubleMatrix &feats,
    DoubleVector &ivector) {
  vector<vector<int> > gselect;
  int num_gselect = opt_.number_gmm_gselect;

  double time_start, time_end;
  //time_start = idec::TimeUtils::GetTimeMilliseconds();
  DiagGmmgSelect(feats, res_->DGmm(), num_gselect, gselect);
  //time_end = idec::TimeUtils::GetTimeMilliseconds();
  //cout << "diag gmm select spend " << time_end - time_start << endl;

  Posterior post;
  double min_post = opt_.min_post;

  //time_start = idec::TimeUtils::GetTimeMilliseconds();
  FullGmmgSelectToPost(feats, res_->FGmm(), post, gselect, min_post);
  //time_end = idec::TimeUtils::GetTimeMilliseconds();
  //cout << "full gmm select spend " << time_end - time_start << " ms." << endl;

  //time_start = idec::TimeUtils::GetTimeMilliseconds();
  ScalePost(feats, post);
  //time_end = idec::TimeUtils::GetTimeMilliseconds();
  //cout << "scale post spend " << time_end - time_start << " ms." << endl;

  bool need_2nd_order_stats = false;
  UtteranceStats utt_stats(res_->GaussNum(), res_->FeatDim(),
                           need_2nd_order_stats);
  //time_start = idec::TimeUtils::GetTimeMilliseconds();
  utt_stats.AccStats(feats, post);
  //time_end = idec::TimeUtils::GetTimeMilliseconds();
  //cout << "accumulate state spend " << time_end - time_start << " ms." << endl;

  //time_start = idec::TimeUtils::GetTimeMilliseconds();
  ExtractIvector(utt_stats, ivector);
  //time_end = idec::TimeUtils::GetTimeMilliseconds();
  //cout << "extract ivector spend " << time_end - time_start << " ms." << endl;
  return;
}

void IvectorExtractPipeline::ExtractIvector(DoubleVector &ivector) {
  ExtractIvector(utt_stats_, ivector);
}

int IvectorExtractPipeline::ExtractIvector(const UtteranceStats &utt_stats,
    DoubleVector &ivector) const {
  IvectorEstimator ivector_estimator = IvectorEstimator(opt_, res_->IvRes());
  ivector_estimator.Run(utt_stats, ivector);
  // ivector -= res_->IvRes().mean;
  return 0;
}

int IvectorExtractPipeline::ScalePost(const DoubleMatrix &feats,
                                      Posterior &post) const {
  if (static_cast<int>(post.Size()) != feats.Rows()) {
    idec::IDEC_WARN << "Size mismatch between posterior " << post.Size()
                    << " and features " << feats.Rows() << " for utterance ";
  }

  double max_count_scale = 1.0;
  double max_count = opt_.max_count;
  double acoustic_weight = opt_.acoustic_weight;
  double this_t = acoustic_weight * post.Total();
  if (max_count > 0 && this_t > max_count) {
    max_count_scale = max_count / this_t;
    idec::IDEC_INFO << "Scaling stats for utterance by scale "
                    << max_count_scale << " due to --max-count=" << max_count;
    this_t = max_count;
  }

  double scale = acoustic_weight * max_count_scale;
  if (scale != 1.0) {
    post.Scale(scale);
  }
  return 0;
}

int IvectorExtractPipeline::AccumulateStats(const DoubleMatrix &feats) {
  int num_gselect = opt_.number_gmm_gselect;
  vector<vector<int> > gselect;
  DiagGmmgSelect(feats, res_->DGmm(), num_gselect, gselect);

  Posterior post;
  double min_post = opt_.min_post;
  FullGmmgSelectToPost(feats, res_->FGmm(), post, gselect, min_post);
  ScalePost(feats, post);
  utt_stats_.AccStats(feats, post);
  return 0;
}

int IvectorExtractPipeline::DiagGmmgSelect(const DoubleMatrix &feats,
    const DiagGmm &gmm, int num_gselect,
    vector<vector<int> > &gselect) const {
  idec::IDEC_ASSERT(num_gselect > 0);
  int num_gauss = gmm.NumGauss();
  if (num_gselect > num_gauss) {
    idec::IDEC_WARN << "You asked for " << num_gselect << " Gaussians but GMM "
                    << "only has " << num_gauss << ", returning this many. ";
    num_gselect = num_gauss;
  }

  if (feats.Rows() != gselect.size()) {
    gselect.resize(feats.Rows());
  }

  int tot_t_this_file = feats.Rows();
  double tot_like_this_file = gmm.GaussianSelection(feats, num_gselect, gselect);
  if (verbose_mode_) {
    idec::IDEC_INFO << tot_t_this_file << " frames is " << (tot_like_this_file /
                    tot_t_this_file);
  }
  return 0;
}

int IvectorExtractPipeline::FullGmmgSelectToPost(const DoubleMatrix &feats,
    const FullGmm &fgmm, Posterior &post,
    vector<vector<int> > &gselect, double min_post) const {
  int tot_posts = 0;
  int num_frames = feats.Rows();
  int dims = feats.Cols();
  if (post.Size() != num_frames) {
    post.Resize(num_frames);
  }

  double this_tot_loglike = 0;
  bool utt_ok = true;
  DoubleVector frame;
  DoubleVector loglikes;
  for (int t = 0; t < num_frames; t++) {
    frame = feats.Rowv(t);
    const std::vector<int> &this_gselect = gselect[t];
    idec::IDEC_ASSERT(!gselect[t].empty());
    fgmm.LogLikelihoodsPreselect(frame, this_gselect, loglikes);
    this_tot_loglike += loglikes.ApplySoftMax();
    if (fabs(loglikes.Sum() - 1.0) > 0.01) {
      idec::IDEC_ERROR <<
                       "Skipping utterance because bad posterior-sum encountered (NaN?)";
    } else {
      if (min_post != 0.0) {
        int max_index = 0;
        loglikes.Max(&max_index);
        for (int i = 0; i < loglikes.Size(); i++) {
          loglikes(i) = (loglikes(i) < min_post) ? 0.0 : loglikes(i);
        }
        double sum = loglikes.Sum();
        if (sum == 0.0) {
          loglikes(max_index) = 1.0;
        } else {
          loglikes.Scale(1.0 / sum);
        }
      }

      for (int i = 0; i < loglikes.Size(); i++) {
        if (loglikes(i) != 0.0) {
          post[t].Add(std::make_pair(this_gselect[i], loglikes(i)));
          tot_posts++;
        }
      }
      idec::IDEC_ASSERT(!post[t].Empty());
    }
  }
  return 0;
}