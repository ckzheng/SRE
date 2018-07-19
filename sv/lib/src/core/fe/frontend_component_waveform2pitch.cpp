#include "am/xnn_runtime.h"
#include "am/xnn_kaldi_utility.h"
#include "base/log_message.h"
#include "util/parse-options.h"
#include "fe/frontend_component_waveform2pitch.h"

namespace idec {

FrontendComponent_Waveform2Pitch::OnlineProcessPitch::OnlineProcessPitch(
  const ProcessPitchOptions &opts,
  OnlineFeatureInterface *src) :
  opts_(opts), src_(src),
  dim_((opts.add_pov_feature ? 1 : 0)
       + (opts.add_normalized_log_pitch ? 1 : 0)
       + (opts.add_delta_pitch ? 1 : 0)
       + (opts.add_raw_log_pitch ? 1 : 0)) {
  IDEC_ASSERT(dim_ > 0 &&
              " At least one of the pitch features should be chosen. "
              "Check your post-process-pitch options.");
  IDEC_ASSERT(src->Dim() == kRawFeatureDim &&
              "Input feature must be pitch feature (should have dimension 2)");

  // default control parameters
  window_ = 2;
  order_ = 1;

  // check
  if (window_ < 1)
    IDEC_ERROR << "delta window size " << window_ <<
               " must be greater than or equal to 1";

  IDEC_ASSERT(order_ >= 0
              && order_ < 1000);  // just make sure we don't get binary junk.
  // opts will normally be 2 or 3.
  IDEC_ASSERT(window_ > 0 && window_ < 1000);  // again, basic sanity check.
  // normally the window size will be two.


  scales_.resize(order_ + 1);
  scales_[0].resize(1);
  scales_[0][0] =
    1.0;  // trivial window for 0th order delta [i.e. baseline feats]

  for (int32 i = 1; i <= order_; i++) {
    std::vector<float> &prev_scales = scales_[i - 1],
                        &cur_scales = scales_[i];
    int32 window = window_;  // this code is designed to still
    // work if instead we later make it an array and do opts.window[i-1],
    // or something like that. "window" is a parameter specifying delta-window
    // width which is actually 2*window + 1.
    IDEC_ASSERT(window != 0);
    int32 prev_offset = (static_cast<int32>(prev_scales.size() - 1)) / 2,
          cur_offset = prev_offset + window;
    cur_scales.resize(prev_scales.size() + 2 * window);  // also zeros it.

    float normalizer = 0.0;
    for (int32 j = -window; j <= window; j++) {
      normalizer += j*j;
      for (int32 k = -prev_offset; k <= prev_offset; k++) {
        cur_scales[j + k + cur_offset] +=
          static_cast<float>(j)* prev_scales[k + prev_offset];
      }
    }
    for (int i = 0; i < (int)cur_scales.size(); ++i) {
      cur_scales[i] /= normalizer;
    }
  }

}

void FrontendComponent_Waveform2Pitch::OnlineProcessPitch::GetFrame(int frame,
    std::vector<float> &feat) {

  int frame_delayed = frame < opts_.delay ? 0 : frame - opts_.delay;
  IDEC_ASSERT((int)feat.size() == dim_ &&
              frame_delayed < NumFramesReady());
  int index = 0;
  if (opts_.add_pov_feature)
    feat[index++] = GetPovFeature(frame_delayed);
  if (opts_.add_normalized_log_pitch)
    feat[index++] = GetNormalizedLogPitchFeature(frame_delayed);
  if (opts_.add_delta_pitch)
    feat[index++] = GetDeltaPitchFeature(frame_delayed);
  if (opts_.add_raw_log_pitch)
    feat[index++] = GetRawLogPitchFeature(frame_delayed);
  IDEC_ASSERT(index == dim_);

}



float FrontendComponent_Waveform2Pitch::OnlineProcessPitch::GetPovFeature(
  int frame) const {
  std::vector<float> tmp(kRawFeatureDim);
  src_->GetFrame(frame, tmp);  // (NCCF, pitch) from pitch extractor
  float nccf = tmp[0];
  return opts_.pov_scale * NccfToPovFeature(nccf)
         + opts_.pov_offset;
}


float FrontendComponent_Waveform2Pitch::OnlineProcessPitch::GetDeltaPitchFeature(
  int frame) {

  // Rather than computing the delta pitch directly in code here,
  // which might seem easier, we accumulate a small window of features
  // and call ComputeDeltas.  This might seem like overkill; the reason
  // we do it this way is to ensure that the end effects (at file
  // beginning and end) are handled in a consistent way.
  int context = opts_.delta_window;
  int start_frame = std::max(0, frame - context),
      end_frame = std::min(frame + context + 1, src_->NumFramesReady()),
      frames_in_window = end_frame - start_frame;

  //std::vector<float > feats(frames_in_window);
  xnnFloatRuntimeMatrix feats(1, frames_in_window), delta_feats;

  for (int32 f = start_frame; f < end_frame; f++)
    //feats[f - start_frame] = GetRawLogPitchFeature(f);
    feats.Col(f - start_frame)[0] = GetRawLogPitchFeature(f);



  DeltaFeaturesOptions delta_opts;
  delta_opts.order = 1;
  delta_opts.window = opts_.delta_window;

  //std::vector<std::vector<float > >delta_feats;
  //delta_feats.resize(feats.size());

  //for (size_t t = 0; t < delta_feats.size(); t++)
  //{
  //    delta_feats[t].resize(delta_opts.order + 1);
  //}

  ComputeDeltas(delta_opts, feats, delta_feats);
  while (delta_feature_noise_.size() <= static_cast<size_t>(frame)) {
    delta_feature_noise_.push_back(RandGauss() *
                                   opts_.delta_pitch_noise_stddev);
  }
  // note: delta_feats will have two columns, second contains deltas.
  return (delta_feats.Col(frame - start_frame)[1] + delta_feature_noise_[frame])
         *
         opts_.delta_pitch_scale;
}


float FrontendComponent_Waveform2Pitch::OnlineProcessPitch::GetRawLogPitchFeature(
  int frame) const {
  std::vector<float> tmp(kRawFeatureDim);
  src_->GetFrame(frame, tmp);
  float pitch = tmp[1];
  IDEC_ASSERT(pitch > 0);
  return log(pitch);
}

float FrontendComponent_Waveform2Pitch::OnlineProcessPitch::GetNormalizedLogPitchFeature(
  int frame) {
  UpdateNormalizationStats(frame);
  double log_pitch = GetRawLogPitchFeature(frame),
         avg_log_pitch = normalization_stats_[frame].sum_log_pitch_pov /
                         normalization_stats_[frame].sum_pov,
                         normalized_log_pitch = log_pitch - avg_log_pitch;
  return (float)normalized_log_pitch * opts_.pitch_scale;
}

// inline
void FrontendComponent_Waveform2Pitch::OnlineProcessPitch::GetNormalizationWindow(
  int t,
  int src_frames_ready,
  int *window_begin,
  int *window_end) const {
  int left_context = opts_.normalization_left_context;
  int right_context = opts_.normalization_right_context;
  *window_begin = std::max(0, t - left_context);
  *window_end = std::min(t + right_context + 1, src_frames_ready);
}


void FrontendComponent_Waveform2Pitch::OnlineProcessPitch::UpdateNormalizationStats(
  int frame) {
  IDEC_ASSERT(frame >= 0);
  if ((int)normalization_stats_.size() <= frame)
    normalization_stats_.resize(frame + 1);
  int cur_num_frames = src_->NumFramesReady();
  bool input_finished = src_->IsLastFrame(cur_num_frames - 1);

  NormalizationStats &this_stats = normalization_stats_[frame];
  if (this_stats.cur_num_frames == cur_num_frames &&
      this_stats.input_finished == input_finished) {
    // Stats are fully up-to-date.
    return;
  }
  int this_window_begin, this_window_end;
  GetNormalizationWindow(frame, cur_num_frames,
                         &this_window_begin, &this_window_end);

  if (frame > 0) {
    const NormalizationStats &prev_stats = normalization_stats_[frame - 1];
    if (prev_stats.cur_num_frames == cur_num_frames &&
        prev_stats.input_finished == input_finished) {
      // we'll derive this_stats efficiently from prev_stats.
      // Checking that cur_num_frames and input_finished have not changed
      // ensures that the underlying features will not have changed.
      this_stats = prev_stats;
      int32 prev_window_begin, prev_window_end;
      GetNormalizationWindow(frame - 1, cur_num_frames,
                             &prev_window_begin, &prev_window_end);
      if (this_window_begin != prev_window_begin) {
        IDEC_ASSERT(this_window_begin == prev_window_begin + 1);
        std::vector<float> tmp(kRawFeatureDim);
        src_->GetFrame(prev_window_begin, tmp);
        float accurate_pov = NccfToPov(tmp[0]),
              log_pitch = log(tmp[1]);
        this_stats.sum_pov -= accurate_pov;
        this_stats.sum_log_pitch_pov -= accurate_pov * log_pitch;
      }
      if (this_window_end != prev_window_end) {
        IDEC_ASSERT(this_window_end == prev_window_end + 1);
        std::vector<float> tmp(kRawFeatureDim);
        src_->GetFrame(prev_window_end, tmp);
        float accurate_pov = NccfToPov(tmp[0]),
              log_pitch = log(tmp[1]);
        this_stats.sum_pov += accurate_pov;
        this_stats.sum_log_pitch_pov += accurate_pov * log_pitch;
      }
      return;
    }
  }
  // The way we do it here is not the most efficient way to do it;
  // we'll see if it becomes a problem.  The issue is we have to redo
  // this computation from scratch each time we process a new chunk, which
  // may be a little inefficient if the chunk-size is very small.
  this_stats.cur_num_frames = cur_num_frames;
  this_stats.input_finished = input_finished;
  this_stats.sum_pov = 0.0f;
  this_stats.sum_log_pitch_pov = 0.0f;
  std::vector<float> tmp(kRawFeatureDim);
  for (int32 f = this_window_begin; f < this_window_end; f++) {
    src_->GetFrame(f, tmp);
    float accurate_pov = NccfToPov(tmp[0]),
          log_pitch = log(tmp[1]);
    this_stats.sum_pov += accurate_pov;
    this_stats.sum_log_pitch_pov += accurate_pov * log_pitch;
  }
}


int FrontendComponent_Waveform2Pitch::OnlineProcessPitch::NumFramesReady()
const {
  int src_frames_ready = src_->NumFramesReady();
  if (src_frames_ready == 0) {
    return 0;
  } else if (src_->IsLastFrame(src_frames_ready - 1)) {
    return src_frames_ready + opts_.delay;
  } else {
    return std::max(0, src_frames_ready -
                    opts_.normalization_right_context + opts_.delay);
  }
}


FrontendComponent_Waveform2Pitch::PitchFrameInfo::PitchFrameInfo(
  int num_states)
  :state_info_(num_states), state_offset_(0),
   cur_best_state_(-1), prev_info_(NULL) {
  //pitch_use_naive_search = false;  // This is used in unit-tests
}


bool pitch_use_naive_search = false;  // This is used in unit-tests.



FrontendComponent_Waveform2Pitch::PitchFrameInfo::PitchFrameInfo(
  PitchFrameInfo *prev_info) :
  state_info_(prev_info->state_info_.size()), state_offset_(0),
  cur_best_state_(-1), prev_info_(prev_info) {
}

void FrontendComponent_Waveform2Pitch::PitchFrameInfo::SetNccfPov(
  const std::vector<float> &nccf_pov) {
  size_t num_states = nccf_pov.size();
  IDEC_ASSERT(num_states == state_info_.size());
  for (size_t i = 0; i < num_states; i++)
    state_info_[i].pov_nccf = nccf_pov[i];
}


void FrontendComponent_Waveform2Pitch::PitchFrameInfo::ComputeLocalCost(
  const std::vector<float> &nccf_pitch,
  const std::vector<float> &lags,
  const PitchExtractionOptions &opts,
  std::vector<float> *local_cost) {
  // from the paper, eq. 5, local_cost = 1 - Phi(t,i)(1 - soft_min_f0 L_i)
  // nccf is the nccf on this frame measured at the lags in "lags".
  IDEC_ASSERT(nccf_pitch.size() == local_cost->size() &&
              nccf_pitch.size() == lags.size());
  //local_cost->Set(1.0);
  for (size_t t = 0; t < local_cost->size(); t++) {
    (*local_cost)[t] = 1.0f;
  }
  // add the term -Phi(t,i):
  //local_cost->AddVec(-1.0, nccf_pitch);
  for (size_t t = 0; t < local_cost->size(); t++) {
    (*local_cost)[t] += (-1.0f)*nccf_pitch[t];
  }
  // add the term soft_min_f0 Phi(t,i) L_i
  //local_cost->AddVecVec(opts.soft_min_f0, lags, nccf_pitch, 1.0);
  for (size_t t = 0; t < lags.size(); t++) {
    (*local_cost)[t] += opts.soft_min_f0 *lags[t] * nccf_pitch[t];
  }
}

void FrontendComponent_Waveform2Pitch::PitchFrameInfo::ComputeBacktraces(
  const PitchExtractionOptions &opts,
  const std::vector<float> &nccf_pitch,
  const std::vector<float> &lags,
  const std::vector<float> &prev_forward_cost_vec,
  std::vector<std::pair<int, int> > *index_info,
  std::vector<float> *this_forward_cost_vec) {
  int num_states = (int)nccf_pitch.size();

  //std::vector<float> local_cost(num_states, kUndefined);
  std::vector<float> local_cost(num_states); // need to set initial value here
  ComputeLocalCost(nccf_pitch, lags, opts, &local_cost);

  const float delta_pitch_sq = pow(log(1.0f + opts.delta_pitch), 2.0f),
              inter_frame_factor = delta_pitch_sq * opts.penalty_factor;

  // index local_cost, prev_forward_cost and this_forward_cost using raw pointer
  // indexing not operator (), since this is the very inner loop and a lot of
  // time is taken here.

  //const float *prev_forward_cost = prev_forward_cost_vec.Data();
  //float *this_forward_cost = this_forward_cost_vec->Data();
  // TODO:
  //std::vector <float> prev_forward_cost;
  //prev_forward_cost.assign(prev_forward_cost_vec.begin(), prev_forward_cost_vec.end());
  //std::vector <float> this_forward_cost;
  //this_forward_cost.assign(this_forward_cost_vec->begin(), this_forward_cost_vec->end());

  if (index_info->empty())
    index_info->resize(num_states);

  // make it a reference for more concise indexing.
  std::vector<std::pair<int32, int32> > &bounds = *index_info;

  /* bounds[i].first will be a lower bound on the backpointer for state i,
  bounds[i].second will be an upper bound on it.  We progressively tighten
  these bounds till we know the backpointers exactly.
  */

  if (pitch_use_naive_search) {
    // This branch is only taken in unit-testing code.
    for (int i = 0; i < num_states; i++) {
      float best_cost = std::numeric_limits<float>::infinity();
      int best_j = -1;
      for (int j = 0; j < num_states; j++) {
        //float this_cost = (j - i) * (j - i) * inter_frame_factor
        //    + prev_forward_cost[j];

        float this_cost = (j - i) * (j - i) * inter_frame_factor
                          + prev_forward_cost_vec[j];
        if (this_cost < best_cost) {
          best_cost = this_cost;
          best_j = j;
        }
      }
      //this_forward_cost[i] = best_cost;
      (*this_forward_cost_vec)[i] = best_cost;
      state_info_[i].backpointer = best_j;
    }
  } else {
    int last_backpointer = 0;
    for (int i = 0; i < num_states; i++) {
      int start_j = last_backpointer;
      float best_cost = (start_j - i) * (start_j - i) * inter_frame_factor
                        + prev_forward_cost_vec[start_j];
      int best_j = start_j;

      for (int j = start_j + 1; j < num_states; j++) {
        float this_cost = (j - i) * (j - i) * inter_frame_factor
                          + prev_forward_cost_vec[j];
        if (this_cost < best_cost) {
          best_cost = this_cost;
          best_j = j;
        } else { // as soon as the costs stop improving, we stop searching.
          break;  // this is a loose lower bound we're getting.
        }
      }
      state_info_[i].backpointer = best_j;
      (*this_forward_cost_vec)[i] = best_cost;
      bounds[i].first = best_j;  // this is now a lower bound on the
      // backpointer.
      bounds[i].second = num_states - 1;  // we have no meaningful upper bound
      // yet.
      last_backpointer = best_j;
    }

    // We iterate, progressively refining the upper and lower bounds until they
    // meet and we know that the resulting backtraces are optimal.  Each
    // iteration takes time linear in num_states.  We won't normally iterate as
    // far as num_states; normally we only do two iterations; when printing out
    // the number of iterations, it's rarely more than that (once I saw seven
    // iterations).  Anyway, this part of the computation does not dominate.
    for (int iter = 0; iter < num_states; iter++) {
      bool changed = false;
      if (iter % 2 == 0) {  // go backwards through the states
        last_backpointer = num_states - 1;
        for (int i = num_states - 1; i >= 0; i--) {
          int lower_bound = bounds[i].first,
              upper_bound = std::min(last_backpointer, bounds[i].second);
          if (upper_bound == lower_bound) {
            last_backpointer = lower_bound;
            continue;
          }
          float best_cost = (*this_forward_cost_vec)[i];
          int best_j = state_info_[i].backpointer, initial_best_j = best_j;

          if (best_j == upper_bound) {
            // if best_j already equals upper bound, don't bother tightening the
            // upper bound, we'll tighten the lower bound when the time comes.
            last_backpointer = best_j;
            continue;
          }
          // Below, we have j > lower_bound + 1 because we know we've already
          // evaluated lower_bound and lower_bound + 1 [via knowledge of
          // this algorithm.]
          for (int j = upper_bound; j > lower_bound + 1; j--) {
            float this_cost = (j - i) * (j - i) * inter_frame_factor
                              + prev_forward_cost_vec[j];
            if (this_cost < best_cost) {
              best_cost = this_cost;
              best_j = j;
            } else { // as soon as the costs stop improving, we stop searching,
              // unless the best j is still lower than j, in which case
              // we obviously need to keep moving.
              if (best_j > j)
                break;  // this is a loose lower bound we're getting.
            }
          }
          // our "best_j" is now an upper bound on the backpointer.
          bounds[i].second = best_j;
          if (best_j != initial_best_j) {
            (*this_forward_cost_vec)[i] = best_cost;
            state_info_[i].backpointer = best_j;
            changed = true;
          }
          last_backpointer = best_j;
        }
      } else { // go forwards through the states.
        last_backpointer = 0;
        for (int i = 0; i < num_states; i++) {
          int lower_bound = std::max(last_backpointer, bounds[i].first),
              upper_bound = bounds[i].second;
          if (upper_bound == lower_bound) {
            last_backpointer = lower_bound;
            continue;
          }
          float best_cost = (*this_forward_cost_vec)[i];
          int best_j = state_info_[i].backpointer, initial_best_j = best_j;

          if (best_j == lower_bound) {
            // if best_j already equals lower bound, we don't bother tightening
            // the lower bound, we'll tighten the upper bound when the time
            // comes.
            last_backpointer = best_j;
            continue;
          }
          // Below, we have j < upper_bound because we know we've already
          // evaluated that point.
          for (int j = lower_bound; j < upper_bound - 1; j++) {
            float this_cost = (j - i) * (j - i) * inter_frame_factor
                              + prev_forward_cost_vec[j];
            if (this_cost < best_cost) {
              best_cost = this_cost;
              best_j = j;
            } else { // as soon as the costs stop improving, we stop searching,
              // unless the best j is still higher than j, in which case
              // we obviously need to keep moving.
              if (best_j < j)
                break;  // this is a loose lower bound we're getting.
            }
          }
          // our "best_j" is now a lower bound on the backpointer.
          bounds[i].first = best_j;
          if (best_j != initial_best_j) {
            (*this_forward_cost_vec)[i] = best_cost;
            state_info_[i].backpointer = best_j;
            changed = true;
          }
          last_backpointer = best_j;
        }
      }
      if (!changed)
        break;
    }
  }
  // The next statement is needed due to RecomputeBacktraces: we have to
  // invalidate the previously computed best-state info.
  cur_best_state_ = -1;
  //this_forward_cost_vec->AddVec(1.0, local_cost);
  for (size_t t = 0; t < (*this_forward_cost_vec).size(); t++) {
    (*this_forward_cost_vec)[t] += (1 * local_cost[t]);
  }
}


void FrontendComponent_Waveform2Pitch::PitchFrameInfo::SetBestState(
  int best_state,
  std::vector<std::pair<int, float> >::iterator iter) {
  // This function would naturally be recursive, but we have coded this to avoid
  // recursion, which would otherwise eat up the stack.  Think of it as a static
  // member function, except we do use "this" right at the beginning.
  PitchFrameInfo *this_info = this;  // it will change in the loop.
  while (this_info != NULL) {
    PitchFrameInfo *prev_info = this_info->prev_info_;
    if (best_state == this_info->cur_best_state_)
      return;  // no change
    if (prev_info != NULL)  // don't write anything for frame -1.
      iter->first = best_state;
    size_t state_info_index = best_state - this_info->state_offset_;
    IDEC_ASSERT(state_info_index < this_info->state_info_.size());
    this_info->cur_best_state_ = best_state;
    best_state = this_info->state_info_[state_info_index].backpointer;
    if (prev_info != NULL)  // don't write anything for frame -1.
      iter->second = this_info->state_info_[state_info_index].pov_nccf;
    this_info = prev_info;
    if (this_info != NULL && this_info->prev_info_ != NULL)
      iter--;

  }
}


int FrontendComponent_Waveform2Pitch::PitchFrameInfo::ComputeLatency(
  int max_latency) {
  if (max_latency <= 0) return 0;

  int latency = 0;

  // This function would naturally be recursive, but we have coded this to avoid
  // recursion, which would otherwise eat up the stack.  Think of it as a static
  // member function, except we do use "this" right at the beginning.
  // This function is called only on the most recent PitchFrameInfo object.
  int num_states = (int)state_info_.size();
  int min_living_state = 0, max_living_state = num_states - 1;
  PitchFrameInfo *this_info = this;  // it will change in the loop.


  for (; this_info != NULL && latency < max_latency;) {
    int offset = this_info->state_offset_;
    IDEC_ASSERT(min_living_state >= offset &&
                (max_living_state - offset) < (int)this_info->state_info_.size());
    min_living_state =
      this_info->state_info_[min_living_state - offset].backpointer;
    max_living_state =
      this_info->state_info_[max_living_state - offset].backpointer;
    if (min_living_state == max_living_state) {
      return latency;
    }
    this_info = this_info->prev_info_;
    if (this_info != NULL)  // avoid incrementing latency for frame -1,
      latency++;            // as it's not a real frame.
  }
  return latency;
}

void FrontendComponent_Waveform2Pitch::PitchFrameInfo::Cleanup(
  PitchFrameInfo *prev_frame) {
  IDEC_ERROR << "Cleanup not implemented.";
}



void FrontendComponent_Waveform2Pitch::OnlinePitchFeatureImpl::SelectLags(
  const PitchExtractionOptions &opts,
  std::vector<float> &lags) {
  // choose lags relative to acceptable pitch tolerance
  float min_lag = (float)(1.0 / opts.max_f0), max_lag = float(1.0 / opts.min_f0);

  std::vector<float> tmp_lags;
  for (float lag = min_lag; lag <= max_lag; lag *= 1.0 + opts.delta_pitch)
    tmp_lags.push_back(lag);
  lags.resize(tmp_lags.size());
  lags.assign(tmp_lags.begin(), tmp_lags.end());
}




FrontendComponent_Waveform2Pitch::LinearResample::LinearResample(
  int samp_rate_in_hz,
  int32 samp_rate_out_hz,
  float filter_cutoff_hz,
  int32 num_zeros) :
  samp_rate_in_(samp_rate_in_hz),
  samp_rate_out_(samp_rate_out_hz),
  filter_cutoff_(filter_cutoff_hz),
  num_zeros_(num_zeros) {
  IDEC_ASSERT(samp_rate_in_hz > 0.0f &&
              samp_rate_out_hz > 0.0f &&
              filter_cutoff_hz > 0.0f &&
              filter_cutoff_hz * 2 < samp_rate_in_hz &&
              filter_cutoff_hz * 2 < samp_rate_out_hz &&
              num_zeros > 0);

  // base_freq is the frequency of the repeating unit, which is the gcd
  // of the input frequencies.
  int32 base_freq = Gcd(samp_rate_in_, samp_rate_out_);
  input_samples_in_unit_ = samp_rate_in_ / base_freq;
  output_samples_in_unit_ = samp_rate_out_ / base_freq;

  SetIndexesAndWeights();
  Reset();
}


int64 FrontendComponent_Waveform2Pitch::LinearResample::GetNumOutputSamples(
  int64 input_num_samp,
  bool flush) const {
  // For exact computation, we measure time in "ticks" of 1.0 / tick_freq,
  // where tick_freq is the least common multiple of samp_rate_in_ and
  // samp_rate_out_.
  int tick_freq = Lcm(samp_rate_in_, samp_rate_out_);
  int ticks_per_input_period = tick_freq / samp_rate_in_;

  // work out the number of ticks in the time interval
  // [ 0, input_num_samp/samp_rate_in_ ).
  int64 interval_length_in_ticks = input_num_samp * ticks_per_input_period;
  if (!flush) {
    float window_width = num_zeros_ / (2.0f * filter_cutoff_);
    // To count the window-width in ticks we take the floor.  This
    // is because since we're looking for the largest integer num-out-samp
    // that fits in the interval, which is open on the right, a reduction
    // in interval length of less than a tick will never make a difference.
    // For example, the largest integer in the interval [ 0, 2 ) and the
    // largest integer in the interval [ 0, 2 - 0.9 ) are the same (both one).
    // So when we're subtracting the window-width we can ignore the fractional
    // part.
    int window_width_ticks = (int)floor(window_width * tick_freq);
    // The time-period of the output that we can sample gets reduced
    // by the window-width (which is actually the distance from the
    // center to the edge of the windowing function) if we're not
    // "flushing the output".
    interval_length_in_ticks -= window_width_ticks;
  }
  if (interval_length_in_ticks <= 0)
    return 0;
  int ticks_per_output_period = tick_freq / samp_rate_out_;
  // Get the last output-sample in the closed interval, i.e. replacing [ ) with
  // [ ].  Note: integer division rounds down.  See
  // http://en.wikipedia.org/wiki/Interval_(mathematics) for an explanation of
  // the notation.
  int64 last_output_samp = interval_length_in_ticks / ticks_per_output_period;
  // We need the last output-sample in the open interval, so if it takes us to
  // the end of the interval exactly, subtract one.
  if (last_output_samp * ticks_per_output_period == interval_length_in_ticks)
    last_output_samp--;
  // First output-sample index is zero, so the number of output samples
  // is the last output-sample plus one.
  int64 num_output_samp = last_output_samp + 1;
  return num_output_samp;
}

void FrontendComponent_Waveform2Pitch::LinearResample::SetIndexesAndWeights() {
  first_index_.resize(output_samples_in_unit_);
  weights_.resize(output_samples_in_unit_);

  double window_width = num_zeros_ / (2.0f * filter_cutoff_);

  for (int32 i = 0; i < output_samples_in_unit_; i++) {
    double output_t = i / static_cast<double>(samp_rate_out_);
    double min_t = output_t - window_width, max_t = output_t + window_width;
    // we do ceil on the min and floor on the max, because if we did it
    // the other way around we would unnecessarily include indexes just
    // outside the window, with zero coefficients.  It's possible
    // if the arguments to the ceil and floor expressions are integers
    // (e.g. if filter_cutoff_ has an exact ratio with the sample rates),
    // that we unnecessarily include something with a zero coefficient,
    // but this is only a slight efficiency issue.
    int32 min_input_index = (int)ceil(min_t * samp_rate_in_),
          max_input_index = (int)floor(max_t * samp_rate_in_),
          num_indices = max_input_index - min_input_index + 1;
    first_index_[i] = min_input_index;
    weights_[i].resize(num_indices);
    for (int32 j = 0; j < num_indices; j++) {
      int32 input_index = min_input_index + j;
      double input_t = input_index / static_cast<double>(samp_rate_in_),
             delta_t = input_t - output_t;
      // sign of delta_t doesn't matter.
      weights_[i][j] = (float)(FilterFunc((float)delta_t) / samp_rate_in_);
    }
  }
}


// inline
void FrontendComponent_Waveform2Pitch::LinearResample::GetIndexes(
  int64 samp_out,
  int64 *first_samp_in,
  int32 *samp_out_wrapped) const {
  // A unit is the smallest nonzero amount of time that is an exact
  // multiple of the input and output sample periods.  The unit index
  // is the answer to "which numbered unit we are in".
  int64 unit_index = samp_out / output_samples_in_unit_;
  // samp_out_wrapped is equal to samp_out % output_samples_in_unit_
  *samp_out_wrapped = static_cast<int32>(samp_out -
                                         unit_index * output_samples_in_unit_);
  *first_samp_in = first_index_[*samp_out_wrapped] +
                   unit_index * input_samples_in_unit_;
}


void FrontendComponent_Waveform2Pitch::LinearResample::Resample(
  const std::vector<float> &input,
  bool flush,
  std::vector<float> *output) {
  int32 input_dim = (int)input.size();
  int64 tot_input_samp = input_sample_offset_ + input_dim,
        tot_output_samp = GetNumOutputSamples(tot_input_samp, flush);

  IDEC_ASSERT(tot_output_samp >= output_sample_offset_);

  output->resize(tot_output_samp - output_sample_offset_);

  // samp_out is the index into the total output signal, not just the part
  // of it we are producing here.
  for (int64 samp_out = output_sample_offset_;
       samp_out < tot_output_samp;
       samp_out++) {
    int64 first_samp_in;
    int32 samp_out_wrapped;
    GetIndexes(samp_out, &first_samp_in, &samp_out_wrapped);
    const std::vector<float> &weights = weights_[samp_out_wrapped];
    // first_input_index is the first index into "input" that we have a weight
    // for.
    int32 first_input_index = static_cast<int32>(first_samp_in -
                              input_sample_offset_);
    float this_output;
    if (first_input_index >= 0 &&
        first_input_index + (int)weights.size() <= input_dim) {
      //std::vector<float> input_part(input, first_input_index, weights.size());
      //this_output = VecVec(input_part, weights);
      std::vector<float> input_part;
      for (size_t t = first_input_index; t < first_input_index + weights.size();
           t++) {
        input_part.push_back(input[t]);
      }
      this_output = vec_dot(input_part, weights);
    } else { // Handle edge cases.
      this_output = 0.0f;
      for (int32 i = 0; i < (int)weights.size(); i++) {
        float weight = weights[i];
        int32 input_index = first_input_index + i;
        int tmp_value = (int)input_remainder_.size() + input_index;
        if (input_index < 0 && tmp_value >= 0) {
          this_output += weight *
                         input_remainder_[input_remainder_.size() + input_index];
        } else if (input_index >= 0 && input_index < input_dim) {
          this_output += weight * input[input_index];
        } else if (input_index >= input_dim) {
          // We're past the end of the input and are adding zero; should only
          // happen if the user specified flush == true, or else we would not
          // be trying to output this sample.
          IDEC_ASSERT(flush);
        }
      }
    }
    int32 output_index = static_cast<int32>(samp_out - output_sample_offset_);
    (*output)[output_index] = this_output;
  }

  if (flush) {
    Reset();  // Reset the internal state.
  } else {
    SetRemainder(input);
    input_sample_offset_ = tot_input_samp;
    output_sample_offset_ = tot_output_samp;
  }
}

void FrontendComponent_Waveform2Pitch::LinearResample::SetRemainder(
  const std::vector<float> &input) {
  std::vector<float> old_remainder(input_remainder_);
  // max_remainder_needed is the width of the filter from side to side,
  // measured in input samples.  you might think it should be half that,
  // but you have to consider that you might be wanting to output samples
  // that are "in the past" relative to the beginning of the latest
  // input... anyway, storing more remainder than needed is not harmful.
  int32 max_remainder_needed = (int)ceil(samp_rate_in_ * num_zeros_ /
                                         filter_cutoff_);
  input_remainder_.resize(max_remainder_needed);
  for (int index = (int)(-1 * input_remainder_.size()); index < 0; index++) {
    // we interpret "index" as an offset from the end of "input" and
    // from the end of input_remainder_.
    int32 input_index = index + (int)input.size();
    if (input_index >= 0)
      input_remainder_[index + input_remainder_.size()] = input[input_index];
    else if (input_index + old_remainder.size() >= 0)
      input_remainder_[index + input_remainder_.size()] =
        old_remainder[input_index + old_remainder.size()];
    // else leave it at zero.
  }
}


void FrontendComponent_Waveform2Pitch::LinearResample::Reset() {
  input_sample_offset_ = 0;
  output_sample_offset_ = 0;
  input_remainder_.resize(0);
}


float FrontendComponent_Waveform2Pitch::LinearResample::FilterFunc(
  float t) const {
  float window,  // raised-cosine (Hanning) window of width
        // num_zeros_/2*filter_cutoff_
        filter;  // sinc filter function
  if (fabs(t) < num_zeros_ / (2.0f * filter_cutoff_))
    window = 0.5f * (float)(1 + cos(M_2PI * filter_cutoff_ / num_zeros_ * t));
  else
    window = 0.0f;  // outside support of window function
  if (t != 0)
    filter = (float)(sin(M_2PI * filter_cutoff_ * t) / (M_PI * t));
  else
    filter = 2 * filter_cutoff_;  // limit of the function at t = 0
  return filter * window;
}


FrontendComponent_Waveform2Pitch::ArbitraryResample::ArbitraryResample(
  int num_samples_in, float samp_rate_in,
  float filter_cutoff, const std::vector<float> &sample_points,
  int32 num_zeros) :
  num_samples_in_(num_samples_in),
  samp_rate_in_(samp_rate_in),
  filter_cutoff_(filter_cutoff),
  num_zeros_(num_zeros) {
  IDEC_ASSERT(num_samples_in > 0 && samp_rate_in > 0.0f &&
              filter_cutoff > 0.0f &&
              filter_cutoff * 2.0f <= samp_rate_in
              && num_zeros > 0);
  // set up weights_ and indices_.  Please try to keep all functions short and
  SetIndexes(sample_points);
  SetWeights(sample_points);
}

void FrontendComponent_Waveform2Pitch::ArbitraryResample::SetWeights(
  const std::vector<float> &sample_points) {
  int32 num_samples_out = (int)NumSamplesOut();
  for (int32 i = 0; i < num_samples_out; i++) {
    for (int32 j = 0; j < (int)weights_[i].size(); j++) {
      float delta_t = sample_points[i] -
                      (first_index_[i] + j) / samp_rate_in_;
      // Include at this point the factor of 1.0 / samp_rate_in_ which
      // appears in the math.
      weights_[i][j] = FilterFunc(delta_t) / samp_rate_in_;
    }
  }
}

void FrontendComponent_Waveform2Pitch::ArbitraryResample::SetIndexes(
  const std::vector<float> &sample_points) {
  int num_samples = (int)sample_points.size();
  first_index_.resize(num_samples);
  weights_.resize(num_samples);
  float filter_width = num_zeros_ / (2.0f * filter_cutoff_);
  for (int i = 0; i < num_samples; i++) {
    // the t values are in seconds.
    float t = sample_points[i],
          t_min = t - filter_width, t_max = t + filter_width;
    int index_min = (int)ceil(samp_rate_in_ * t_min),
        index_max = (int)floor(samp_rate_in_ * t_max);
    // the ceil on index min and the floor on index_max are because there
    // is no point using indices just outside the window (coeffs would be zero).
    if (index_min < 0)
      index_min = 0;
    if (index_max >= num_samples_in_)
      index_max = num_samples_in_ - 1;
    first_index_[i] = index_min;
    weights_[i].resize(index_max - index_min + 1);
  }
}

float FrontendComponent_Waveform2Pitch::ArbitraryResample::FilterFunc(
  float t) const {
  float window,  // raised-cosine (Hanning) window of width
        // num_zeros_/2*filter_cutoff_
        filter;  // sinc filter function
  if (fabs(t) < num_zeros_ / (2.0f * filter_cutoff_))
    window = 0.5f * (float)(1 + cos(M_2PI * filter_cutoff_ / num_zeros_ * t));
  else
    window = 0.0f;  // outside support of window function
  if (t != 0.0)
    filter = (float)(sin(M_2PI * filter_cutoff_ * t) / (M_PI * t));
  else
    filter = 2.0f * filter_cutoff_;  // limit of the function at zero.
  return filter * window;
}


void FrontendComponent_Waveform2Pitch::ArbitraryResample::Resample(
  const xnnFloatRuntimeMatrix &input,
  xnnFloatRuntimeMatrix *output) const {
  // each row of "input" corresponds to the data to resample;
  // the corresponding row of "output" is the resampled data.

#if 1
  IDEC_ASSERT(input.NumCols() == output->NumCols() &&
              input.NumRows() == (size_t)num_samples_in_ &&
              output->NumRows() == weights_.size());

  //std::vector<float> output_col(output->NumRows());
  xnnFloatRuntimeMatrix output_col(1, output->NumCols());
  for (size_t i = 0; i < NumSamplesOut(); i++) {

    /* just for dealing one frame per chunk*/
    xnnFloatRuntimeMatrix input_part(weights_[i].size(), input.NumCols());

    for (size_t col = 0; col < input.NumCols(); col++) {
      for (size_t row = 0; row < weights_[i].size(); row++) {
        input_part.Col(col)[row] = input.Col(col)[row + first_index_[i]];
      }
    }

    //const std::vector<float> &weight_vec(weights_[i]);
    xnnFloatRuntimeMatrix weight_vec(weights_[i].size(), 1);
    for (int j = 0; j < (int)weights_[i].size(); j++) {
      weight_vec.Col(0)[j] = weights_[i][j];
    }

    //output_col.AddMatVec(1.0, input_part,
    //   kNoTrans, weight_vec, 0.0);
    // output->CopyColFromVec(output_col, i);

    output_col.ScalePlusMatTMat(0, weight_vec, input_part);
    for (int j = 0; j < (int)output->NumCols(); j++) {
      output->Col(j)[i] = output_col.Col(j)[0];
    }


    //outputBuff_.Resize(outputDim_, inputMatrix_.NumCols());
    //outputBuff_.ScalePlusMatTMat(0, dct_matrix_, inputMatrix_);

  }





#endif
}


void FrontendComponent_Waveform2Pitch::ArbitraryResample::Resample(
  const std::vector<float> &input,
  std::vector<float> *output) const {
  IDEC_ASSERT((int)input.size() == num_samples_in_ &&
              output->size() == weights_.size());

  int32 output_dim = (int)output->size();
  for (int32 i = 0; i < output_dim; i++) {
    //std::vector<float> input_part(input, first_index_[i], weights_[i].Dim());
    //std::vector<float> input_part(weights_[i].size() - first_index_[i]);
    //std::copy(input.begin() + first_index_[i], input.begin() + first_index_[i] + weights_[i].size(), input_part.begin());
    std::vector<float> input_part;
    input_part.assign(input.begin() + first_index_[i],
                      input.begin() + first_index_[i] + weights_[i].size());
    (*output)[i] = vec_dot(input_part, weights_[i]);
  }
}


FrontendComponent_Waveform2Pitch::OnlinePitchFeatureImpl::OnlinePitchFeatureImpl(
  const PitchExtractionOptions &opts) :
  opts_(opts), forward_cost_remainder_(0.0f), input_finished_(false),
  signal_sumsq_(0.0f), signal_sum_(0.0f), downsampled_samples_processed_(0) {
  signal_resampler_ = new LinearResample((int)opts.samp_freq,
                                         (int)opts.resample_freq,
                                         opts.lowpass_cutoff,
                                         opts.lowpass_filter_width);

  double outer_min_lag = 1.0f / opts.max_f0 -
                         (opts.upsample_filter_width / (2.0f * opts.resample_freq));
  double outer_max_lag = 1.0f / opts.min_f0 +
                         (opts.upsample_filter_width / (2.0f * opts.resample_freq));
  nccf_first_lag_ = (int)ceil(opts.resample_freq * outer_min_lag);
  nccf_last_lag_ = (int)floor(opts.resample_freq * outer_max_lag);

  frames_latency_ = 0;  // will be set in AcceptWaveform()

  // Choose the lags at which we resample the NCCF.
  SelectLags(opts, lags_);


  float upsample_cutoff = opts.resample_freq * 0.5f;


  std::vector<float> lags_offset(lags_);

  //lags_offset.Add(-nccf_first_lag_ / opts.resample_freq);
  for (size_t i = 0; i < lags_offset.size(); i++) {
    lags_offset[i] += (-nccf_first_lag_ / opts.resample_freq);
  }

  int num_measured_lags = nccf_last_lag_ + 1 - nccf_first_lag_;

  nccf_resampler_ = new ArbitraryResample(num_measured_lags, opts.resample_freq,
                                          upsample_cutoff, lags_offset,
                                          opts.upsample_filter_width);

  // add a PitchInfo object for frame -1 (not a real frame).
  frame_info_.push_back(new PitchFrameInfo((int)lags_.size()));
  // zeroes forward_cost_; this is what we want for the fake frame -1.
  forward_cost_.resize(lags_.size());
}

int FrontendComponent_Waveform2Pitch::OnlinePitchFeatureImpl::NumFramesAvailable(
  int64 num_downsampled_samples, bool snip_edges) const {
  int frame_shift = opts_.NccfWindowShift(),
      frame_length = opts_.NccfWindowSize();
  // Use the "full frame length" to compute the number
  // of frames only if the input is not finished.
  if (!input_finished_)
    frame_length += nccf_last_lag_;
  if (num_downsampled_samples < frame_length) return 0;
  else if (input_finished_ && !snip_edges) {
    return (int32)(num_downsampled_samples * 1.0f / frame_shift + 0.5f);
  } else
    return (int32)(((num_downsampled_samples - frame_length) / frame_shift) + 1);
}

void FrontendComponent_Waveform2Pitch::OnlinePitchFeatureImpl::UpdateRemainder(
  const std::vector<float> &downsampled_wave_part) {
  // frame_info_ has an extra element at frame-1, so subtract
  // one from the length.
  int64 num_frames = static_cast<int64>(frame_info_.size()) - 1,
        next_frame = num_frames,
        frame_shift = opts_.NccfWindowShift(),
        next_frame_sample = frame_shift * next_frame;

  //signal_sumsq_ += VecVec(downsampled_wave_part, downsampled_wave_part);
  //signal_sum_ += downsampled_wave_part.Sum();
  signal_sumsq_ += vec_dot(downsampled_wave_part, downsampled_wave_part);
  for (size_t t = 0; t < downsampled_wave_part.size(); t++) {
    signal_sum_ += downsampled_wave_part[t];
  }

  // next_frame_sample is the first sample index we'll need for the
  // next frame.
  int64 next_downsampled_samples_processed =
    downsampled_samples_processed_ + downsampled_wave_part.size();

  if (next_frame_sample > next_downsampled_samples_processed) {
    // this could only happen in the weird situation that the full frame length
    // is less than the frame shift.
    int32 full_frame_length = opts_.NccfWindowSize() + nccf_last_lag_;
    IDEC_ASSERT(full_frame_length < frame_shift && "Code error");
    downsampled_signal_remainder_.resize(0);
  } else {
    std::vector<float> new_remainder(next_downsampled_samples_processed -
                                     next_frame_sample);
    // note: next_frame_sample is the index into the entire signal, of
    // new_remainder(0).
    // i is the absolute index of the signal.
    for (int64 i = next_frame_sample;
         i < next_downsampled_samples_processed; i++) {
      if (i >= downsampled_samples_processed_) {  // in current signal.
        new_remainder[i - next_frame_sample] =
          downsampled_wave_part[i - downsampled_samples_processed_];
      } else { // in old remainder; only reach here if waveform supplied is
        new_remainder[i - next_frame_sample] =                      //  tiny.
          downsampled_signal_remainder_[i - downsampled_samples_processed_ +
                                        downsampled_signal_remainder_.size()];
      }
    }
    downsampled_signal_remainder_.swap(new_remainder);
  }
  downsampled_samples_processed_ = next_downsampled_samples_processed;
}


void FrontendComponent_Waveform2Pitch::OnlinePitchFeatureImpl::ExtractFrame(
  const std::vector<float> &downsampled_wave_part,
  int64 sample_index,
  std::vector<float> *window) {

  int32 full_frame_length = (int)window->size();
  int32 offset = static_cast<int32>(sample_index -
                                    downsampled_samples_processed_);

  if (offset + full_frame_length > (int)downsampled_wave_part.size()) {
    // Requested frame is past end of the signal.  This should only happen if
    // input_finished_ == true, when we're flushing out the last couple of
    // frames of signal.  In this case we pad with zeros.
    IDEC_ASSERT(input_finished_);
    int32 new_full_frame_length = (int)downsampled_wave_part.size() - offset;
    IDEC_ASSERT(new_full_frame_length > 0);
    //window->setZero();
    for (size_t t = 0; t < window->size(); t++) {
      (*window)[t] = 0;
    }
    //std::vector<float> sub_window(*window, 0, new_full_frame_length);
    sub_window_.assign(window->begin(), window->begin() + new_full_frame_length);
    ExtractFrame(downsampled_wave_part, sample_index, &sub_window_);
    for (int i = 0; i < new_full_frame_length; i++) {
      (*window)[i] = sub_window_[i];
    }

    return;
  }

  // "offset" is the offset of the start of the frame, into this
  // signal.
  if (offset >= 0) {
    // frame is full inside the new part of the signal.
    //window->CopyFromVec(downsampled_wave_part.Range(offset, full_frame_length));
    for (int t = offset; t < full_frame_length + offset; t++) {
      (*window)[t - offset] = downsampled_wave_part[t];
    }

  } else {
    // frame is partly in the remainder and partly in the new part.
    int32 remainder_offset = (int)downsampled_signal_remainder_.size() + offset;
    IDEC_ASSERT(remainder_offset >= 0);  // or we didn't keep enough remainder.
    IDEC_ASSERT(offset + full_frame_length > 0);  // or we should have
    // processed this frame last
    // time.

    int32 old_length = -offset, new_length = offset + full_frame_length;
    //window->Range(0, old_length).CopyFromVec(
    //    downsampled_signal_remainder_.Range(remainder_offset, old_length));
    //window->Range(old_length, new_length).CopyFromVec(
    //    downsampled_wave_part.Range(0, new_length));

    for (int t = remainder_offset; t < remainder_offset + old_length; t++) {
      (*window)[t - remainder_offset] = downsampled_signal_remainder_[t];
    }
    for (int t = 0; t < new_length; t++) {
      (*window)[t + old_length] = downsampled_wave_part[t];
    }
  }
  if (opts_.preemph_coeff != 0.0f) {
    float preemph_coeff = opts_.preemph_coeff;
    for (int32 i = (int)window->size() - 1; i > 0; i--)
      (*window)[i] -= preemph_coeff * (*window)[i - 1];
    (*window)[0] *= (1.0f - preemph_coeff);
  }
}


bool FrontendComponent_Waveform2Pitch::OnlinePitchFeatureImpl::IsLastFrame(
  int32 frame) const {
  int32 T = NumFramesReady();
  IDEC_ASSERT(frame < T);
  return (input_finished_ && frame + 1 == T);// Trace back the best-path.
}

int FrontendComponent_Waveform2Pitch::OnlinePitchFeatureImpl::NumFramesReady()
const {
  int num_frames = (int)lag_nccf_.size(),
      latency = frames_latency_;
  IDEC_ASSERT(latency <= num_frames);
  return num_frames - latency;
}


void FrontendComponent_Waveform2Pitch::OnlinePitchFeatureImpl::GetFrame(
  int32 frame,
  std::vector<float> *feat) {
  IDEC_ASSERT(frame < NumFramesReady() && feat->size() == 2);
  (*feat)[0] = lag_nccf_[frame].second;
  (*feat)[1] = 1.0f / lags_[lag_nccf_[frame].first];
}


void FrontendComponent_Waveform2Pitch::OnlinePitchFeatureImpl::InputFinished() {
  input_finished_ = true;
  // Process an empty waveform; this has an effect because
  // after setting input_finished_ to true, NumFramesAvailable()
  // will return a slightly larger number.
  AcceptWaveform(opts_.samp_freq, std::vector<float>());
  int num_frames = (int)(static_cast<size_t>(frame_info_.size() - 1));
  if (num_frames < opts_.recompute_frame && !opts_.nccf_ballast_online)
    RecomputeBacktraces();
  frames_latency_ = 0;
  IDEC_INFO << "Pitch-tracking Viterbi cost is "
            << (forward_cost_remainder_ / num_frames)
            << " per frame, over " << num_frames << " frames.";
}



FrontendComponent_Waveform2Pitch::OnlinePitchFeatureImpl::~OnlinePitchFeatureImpl() {
  delete nccf_resampler_;
  delete signal_resampler_;
  for (size_t i = 0; i < frame_info_.size(); i++)
    delete frame_info_[i];
  for (size_t i = 0; i < nccf_info_.size(); i++)
    delete nccf_info_[i];
}



// see comment with declaration.  This is only relevant for online
// operation (it gets called for non-online mode, but is a no-op).
void FrontendComponent_Waveform2Pitch::OnlinePitchFeatureImpl::RecomputeBacktraces() {

  IDEC_ASSERT(!opts_.nccf_ballast_online);
  int num_frames = static_cast<int32>(frame_info_.size()) - 1;

  // The assertion reflects how we believe this function will be called.
  IDEC_ASSERT(num_frames <= opts_.recompute_frame);
  IDEC_ASSERT(nccf_info_.size() == static_cast<size_t>(num_frames));
  if (num_frames == 0)
    return;
  double num_samp = (double)downsampled_samples_processed_, sum = signal_sum_,
         sumsq = signal_sumsq_, mean = sum / num_samp;
  float mean_square = (float)(sumsq / num_samp - mean * mean);

  bool must_recompute = false;
  float threshold = 0.01f;
  for (int32 frame = 0; frame < num_frames; frame++)
    if (!ApproxEqual(nccf_info_[frame]->mean_square_energy,
                     mean_square, threshold))
      must_recompute = true;

  if (!must_recompute) {
    // Nothing to do.  We'll reach here, for instance, if everything was in one
    // chunk and opts_.nccf_ballast_online == false.  This is the case for
    // offline processing.
    for (size_t i = 0; i < nccf_info_.size(); i++)
      delete nccf_info_[i];
    nccf_info_.clear();
    return;
  }

  int32 num_states = (int)forward_cost_.size(),
        basic_frame_length = opts_.NccfWindowSize();

  float new_nccf_ballast = pow(mean_square * basic_frame_length, 2) *
                           opts_.nccf_ballast;

  double forward_cost_remainder = 0.0f;
  std::vector<float> forward_cost(num_states),  // start off at zero.
      next_forward_cost(forward_cost);
  std::vector<std::pair<int32, int32 > > index_info;

  for (int32 frame = 0; frame < num_frames; frame++) {
    NccfInfo &nccf_info = *nccf_info_[frame];
    float old_mean_square = nccf_info_[frame]->mean_square_energy,
          avg_norm_prod = nccf_info_[frame]->avg_norm_prod,
          old_nccf_ballast = pow(old_mean_square * basic_frame_length, 2) *
                             opts_.nccf_ballast,
                             nccf_scale = pow((old_nccf_ballast + avg_norm_prod) /
                                          (new_nccf_ballast + avg_norm_prod),
                                          static_cast<float>(0.5f));
    // The "nccf_scale" is an estimate of the scaling factor by which the NCCF
    // would change on this frame, on average, by changing the ballast term from
    // "old_nccf_ballast" to "new_nccf_ballast".  It's not exact because the
    // "avg_norm_prod" is just an average of the product e1 * e2 of frame
    // energies of the (frame, shifted-frame), but these won't change that much
    // within a frame, and even if they do, the inaccuracy of the scaled NCCF
    // will still be very small if the ballast term didn't change much, or if
    // it's much larger or smaller than e1*e2.  By doing it as a simple scaling,
    // we save the overhead of the NCCF resampling, which is a considerable part
    // of the whole computation.

    //nccf_info.nccf_pitch_resampled.Scale(nccf_scale);
    for (size_t t = 0; t < nccf_info.nccf_pitch_resampled.size(); t++) {
      nccf_info.nccf_pitch_resampled[t] *= nccf_scale;
    }

    frame_info_[frame + 1]->ComputeBacktraces(
      opts_, nccf_info.nccf_pitch_resampled, lags_,
      forward_cost, &index_info, &next_forward_cost);

    forward_cost.swap(next_forward_cost);
    //float remainder = forward_cost.Min();
    float remainder = forward_cost[0];
    for (size_t t = 1; t < forward_cost.size(); t++) {
      if (forward_cost[t] < remainder) {
        remainder = forward_cost[t];
      }
    }
    forward_cost_remainder += remainder;
    //forward_cost.Add(-remainder);
    for (size_t t = 0; t < forward_cost.size(); t++) {
      forward_cost[t] += (-remainder);

    }
  }
  IDEC_INFO << "Forward-cost per frame changed from "
            << (forward_cost_remainder_ / num_frames) << " to "
            << (forward_cost_remainder / num_frames);

  forward_cost_remainder_ = forward_cost_remainder;
  forward_cost_.swap(forward_cost);

  //int32 best_final_state;
  //forward_cost_.Min(&best_final_state);
  int best_final_state = 0;
  float tmp_forward_cost_ = forward_cost_[0];
  for (size_t t = 1; t < forward_cost.size(); t++) {
    if (forward_cost[t] < tmp_forward_cost_) {
      tmp_forward_cost_ = forward_cost[t];
      best_final_state = (int)t;
    }
  }

  if (lag_nccf_.size() != static_cast<size_t>(num_frames))
    lag_nccf_.resize(num_frames);

  std::vector<std::pair<int32, float> >::iterator last_iter =
    lag_nccf_.end() - 1;
  frame_info_.back()->SetBestState(best_final_state, last_iter);
  frames_latency_ =
    frame_info_.back()->ComputeLatency(opts_.max_frames_latency);
  for (size_t i = 0; i < nccf_info_.size(); i++)
    delete nccf_info_[i];
  nccf_info_.clear();

}


void FrontendComponent_Waveform2Pitch::OnlinePitchFeatureImpl::ComputeCorrelation(
  const std::vector<float> &wave,
  int first_lag, int last_lag,
  int nccf_window_size,
  std::vector<float> *inner_prod,
  std::vector<float> *norm_prod) {
  std::vector<float> zero_mean_wave(wave);
  // TODO: possibly fix this, the mean normalization is done in a strange way.
  //std::vector<float> wave_part(wave, 0, nccf_window_size);
  std::vector<float> wave_part;
  wave_part.assign(wave.begin(), wave.begin() + nccf_window_size);
  // subtract mean-frame from wave
  //zero_mean_wave.Add(-wave_part.Sum() / nccf_window_size);
  float wav_part_sum = 0;
  for (size_t t = 0; t < wave_part.size(); t++) {
    wav_part_sum += wave_part[t];
  }
  for (size_t t = 0; t < zero_mean_wave.size(); t++) {
    zero_mean_wave[t] += (-wav_part_sum / nccf_window_size);
  }
  float e1, e2, sum;
  //std::vector<float> sub_vec1(zero_mean_wave, 0, nccf_window_size);
  std::vector<float> sub_vec1;
  //std::copy(zero_mean_wave.begin(), zero_mean_wave.begin() + nccf_window_size, sub_vec1.begin());
  sub_vec1.assign(zero_mean_wave.begin(),
                  zero_mean_wave.begin() + nccf_window_size);
  e1 = vec_dot(sub_vec1, sub_vec1);
  for (int lag = first_lag; lag <= last_lag; lag++) {
    //std::vector<float> sub_vec2(zero_mean_wave, lag, nccf_window_size);
    std::vector<float> sub_vec2;
    sub_vec2.assign(zero_mean_wave.begin() + lag,
                    zero_mean_wave.begin() + lag + nccf_window_size);
    e2 = vec_dot(sub_vec2, sub_vec2);
    sum = vec_dot(sub_vec1, sub_vec2);
    (*inner_prod)[lag - first_lag] = sum;
    (*norm_prod)[lag - first_lag] = e1 * e2;
  }
}


void FrontendComponent_Waveform2Pitch::OnlinePitchFeatureImpl::ComputeNccf(
  const std::vector<float> &inner_prod,
  const std::vector<float> &norm_prod,
  float nccf_ballast,
  // std::vector<float> &nccf_vec)
  float *nccf_vec) {
  //IDEC_ASSERT(inner_prod.size() == norm_prod.size() &&
  //    inner_prod.size() == nccf_vec->size());

  for (int lag = 0; lag < (int)inner_prod.size(); lag++) {
    float numerator = inner_prod[lag],
          denominator = pow(norm_prod[lag] + nccf_ballast, 0.5f),
          nccf;
    if (denominator != 0.0f) {
      nccf = numerator / denominator;
    } else {
      IDEC_ASSERT(numerator == 0.0f);
      nccf = 0.0f;
    }
    IDEC_ASSERT(nccf < 1.01f && nccf > -1.01f);
    nccf_vec[lag] = nccf;
  }
}


void FrontendComponent_Waveform2Pitch::OnlinePitchFeatureImpl::AcceptWaveform(
  float sampling_rate,
  const std::vector<float> &wave) {
  // flush out the last few samples of input waveform only if input_finished_ ==
  // true.
  const bool flush = input_finished_;

  std::vector<float> downsampled_wave;
  signal_resampler_->Resample(wave, flush, &downsampled_wave);

  // these variables will be used to compute the root-mean-square value of the
  // signal for the ballast term.
  double cur_sumsq = signal_sumsq_, cur_sum = signal_sum_;
  int64 cur_num_samp = downsampled_samples_processed_,
        prev_frame_end_sample = 0;
  if (!opts_.nccf_ballast_online) {
    //cur_sumsq += VecVec(downsampled_wave, downsampled_wave);
    //cur_sum += downsampled_wave.Sum();
    //cur_num_samp += downsampled_wave.Dim();
    cur_sumsq += vec_dot(downsampled_wave, downsampled_wave);
    for (size_t t = 0; t < downsampled_wave.size(); t++) {
      cur_sum += downsampled_wave[t];
    }
    cur_num_samp += downsampled_wave.size();
  }

  // end_frame is the total number of frames we can now process, including
  // previously processed ones.
  int32 end_frame = NumFramesAvailable(
                      downsampled_samples_processed_ + downsampled_wave.size(), opts_.snip_edges);
  // "start_frame" is the first frame-index we process
  int32 start_frame = (int)frame_info_.size() - 1,
        num_new_frames = end_frame - start_frame;

  if (num_new_frames == 0) {
    UpdateRemainder(downsampled_wave);
    return;
    // continuing to the rest of the code would generate
    // an error when sizing matrices with zero rows, and
    // anyway is a waste of time.
  }

  int32 num_measured_lags = nccf_last_lag_ + 1 - nccf_first_lag_,
        num_resampled_lags = (int)lags_.size(),
        frame_shift = opts_.NccfWindowShift(),
        basic_frame_length = opts_.NccfWindowSize(),
        full_frame_length = basic_frame_length + nccf_last_lag_;

  std::vector<float> window(full_frame_length),
      inner_prod(num_measured_lags),
      norm_prod(num_measured_lags);
  xnnFloatRuntimeMatrix nccf_pitch(num_measured_lags, num_new_frames),
                        nccf_pov(num_measured_lags, num_new_frames);
  //std::vector<float> nccf_pitch(num_measured_lags), nccf_pov(num_measured_lags);
  std::vector<float> cur_forward_cost(num_resampled_lags);


  // Because the resampling of the NCCF is more efficient when grouped together,
  // we first compute the NCCF for all frames, then resample as a matrix, then
  // do the Viterbi [that happens inside the constructor of PitchFrameInfo].

  for (int32 frame = start_frame; frame < end_frame; frame++) {
    // start_sample is index into the whole wave, not just this part.
    int64 start_sample = static_cast<int64>(frame)* frame_shift;
    ExtractFrame(downsampled_wave, start_sample, &window);
    if (opts_.nccf_ballast_online) {
      // use only up to end of current frame to compute root-mean-square value.
      // end_sample will be the sample-index into "downsampled_wave", so
      // not really comparable to start_sample.
      int64 end_sample = start_sample + full_frame_length -
                         downsampled_samples_processed_;
      IDEC_ASSERT(end_sample > 0);  // or should have processed this frame last
      // time.  Note: end_sample is one past last
      // sample.
      if (end_sample > (int64)downsampled_wave.size()) {
        IDEC_ASSERT(input_finished_);
        end_sample = (int64)downsampled_wave.size();
      }
      //std::vector<float> new_part(downsampled_wave, prev_frame_end_sample,
      //    end_sample - prev_frame_end_sample);
      std::vector<float> new_part;
      for (int64 t = prev_frame_end_sample; t < end_sample; t++) {
        new_part.push_back(downsampled_wave[t]);
      }
      cur_num_samp += new_part.size();
      cur_sumsq += vec_dot(new_part, new_part);
      //cur_sum += new_part.Sum();
      for (size_t t = 0; t < new_part.size(); t++) {
        cur_sum += new_part[t];
      }

      prev_frame_end_sample = end_sample;
    }
    double mean_square = cur_sumsq / cur_num_samp -
                         pow(cur_sum / cur_num_samp, 2.0f);

    ComputeCorrelation(window, nccf_first_lag_, nccf_last_lag_,
                       basic_frame_length, &inner_prod, &norm_prod);
    double nccf_ballast_pov = 0.0f,
           nccf_ballast_pitch = pow(mean_square * basic_frame_length, 2) *
                                opts_.nccf_ballast;
    //avg_norm_prod = norm_prod.Sum() / norm_prod.Dim();
    double norm_prod_sum = 0.0f;
    for (size_t t = 0; t < norm_prod.size(); t++) {
      norm_prod_sum += norm_prod[t];
    }
    double avg_norm_prod = norm_prod_sum / norm_prod.size();

    //std::vector<float> nccf_pitch_row(nccf_pitch, frame - start_frame);

    float *nccf_pitch_row = nccf_pitch.Col(frame - start_frame);
    ComputeNccf(inner_prod, norm_prod, (float)nccf_ballast_pitch,
                nccf_pitch_row);

    //ComputeNccf(inner_prod, norm_prod, (float)nccf_ballast_pitch,
    //    nccf_pitch);

    //std::vector<float> nccf_pov_row(nccf_pov, frame - start_frame); // featch the needed row of matrix

    float *nccf_pov_row = nccf_pov.Col(frame - start_frame);
    ComputeNccf(inner_prod, norm_prod, (float)nccf_ballast_pov,
                nccf_pov_row);

    //ComputeNccf(inner_prod, norm_prod, (float)nccf_ballast_pov,
    //    nccf_pov);

    if (frame < opts_.recompute_frame)
      nccf_info_.push_back(new NccfInfo((float)avg_norm_prod, (float)mean_square));
  }

  //xnnFloatRuntimeMatrix nccf_pitch_resampled(num_new_frames, num_resampled_lags);
  //nccf_resampler_->Resample(nccf_pitch, &nccf_pitch_resampled);
  //nccf_pitch.Resize(0, 0);  // no longer needed.
  //xnnFloatRuntimeMatrix nccf_pov_resampled(num_new_frames, num_resampled_lags);
  //nccf_resampler_->Resample(nccf_pov, &nccf_pov_resampled);
  //nccf_pov.Resize(0, 0);  // no longer needed.

  //std::vector<float> nccf_pitch_resampled(num_resampled_lags);
  //nccf_resampler_->Resample(nccf_pitch, &nccf_pitch_resampled);
  //nccf_pitch.resize(0);  // no longer needed.
  xnnFloatRuntimeMatrix nccf_pitch_resampled(num_resampled_lags, num_new_frames);
  nccf_resampler_->Resample(nccf_pitch, &nccf_pitch_resampled);
  nccf_pitch.Resize(0, 0);  // no longer needed.

  //std::vector<float> nccf_pov_resampled(num_resampled_lags);
  //nccf_resampler_->Resample(nccf_pov, &nccf_pov_resampled);
  // nccf_pov.resize(0);  // no longer needed.
  xnnFloatRuntimeMatrix nccf_pov_resampled(num_resampled_lags, num_new_frames);
  nccf_resampler_->Resample(nccf_pov, &nccf_pov_resampled);
  nccf_pov.Resize(0, 0);  // no longer needed.

  // We've finished dealing with the waveform so we can call UpdateRemainder
  // now; we need to call it before we possibly call RecomputeBacktraces()
  // below, which is why we don't do it at the very end.
  UpdateRemainder(downsampled_wave);

  std::vector<std::pair<int, int > > index_info;

  for (int32 frame = start_frame; frame < end_frame; frame++) {
    int32 frame_idx = frame - start_frame;
    PitchFrameInfo *prev_info = frame_info_.back(),
                    *cur_info = new PitchFrameInfo(prev_info);

    //cur_info->SetNccfPov(nccf_pov_resampled.Col(frame_idx));
    //cur_info->ComputeBacktraces(opts_, nccf_pitch_resampled.Col(frame_idx),
    //    lags_, forward_cost_, &index_info,
    //    &cur_forward_cost);


    std::vector <float > nccf_pov_resampled_row;
    for (size_t k = 0; k < nccf_pov_resampled.NumRows(); k++) {
      nccf_pov_resampled_row.push_back(nccf_pov_resampled.Col(frame_idx)[k]);
    }
    cur_info->SetNccfPov(nccf_pov_resampled_row);

    std::vector <float > nccf_pitch_resampled_row;
    for (size_t k = 0; k < nccf_pov_resampled.NumRows(); k++) {
      nccf_pitch_resampled_row.push_back(nccf_pitch_resampled.Col(frame_idx)[k]);
    }
    cur_info->ComputeBacktraces(opts_, nccf_pitch_resampled_row,
                                lags_, forward_cost_, &index_info,
                                &cur_forward_cost);
#if 0
    std::vector <float > nccf_pov_resampled_row;
    nccf_pov_resampled_row.assign(nccf_pov_resampled.begin(),
                                  nccf_pov_resampled.end());
    cur_info->SetNccfPov(nccf_pov_resampled_row);

    std::vector <float > nccf_pitch_resampled_row;
    nccf_pitch_resampled_row.assign(nccf_pitch_resampled.begin(),
                                    nccf_pitch_resampled.end());
    cur_info->ComputeBacktraces(opts_, nccf_pitch_resampled_row,
                                lags_, forward_cost_, &index_info,
                                &cur_forward_cost);
#endif

    forward_cost_.swap(cur_forward_cost);
    // Renormalize forward_cost so smallest element is zero.
    //float remainder = forward_cost_.Min();
    float remainder = forward_cost_[0];
    for (size_t t = 1; t < forward_cost_.size(); t++) {
      if (forward_cost_[t] < remainder) {
        remainder = forward_cost_[t];
      }
    }

    forward_cost_remainder_ += remainder;
    //forward_cost_.Add(-remainder);
    for (size_t t = 0; t < forward_cost_.size(); t++) {
      forward_cost_[t] += (-remainder);
    }
    frame_info_.push_back(cur_info);
    if (frame < opts_.recompute_frame) {
      //nccf_info_[frame]->nccf_pitch_resampled.assign(nccf_pitch_resampled.begin(), nccf_pitch_resampled.end());
      nccf_info_[frame]->nccf_pitch_resampled.resize(nccf_pitch_resampled.NumRows());
      for (size_t k = 0; k < nccf_pitch_resampled.NumRows(); k++) {
        nccf_info_[frame]->nccf_pitch_resampled[k] = nccf_pitch_resampled.Col(
              frame_idx)[k];
      }
    }


    if (frame == opts_.recompute_frame - 1 && !opts_.nccf_ballast_online)
      RecomputeBacktraces();
  }

  // Trace back the best-path.
  int best_final_state;
  float min_forward_cost_;
  //forward_cost_.Min(&best_final_state);
  min_forward_cost_ = forward_cost_[0];
  best_final_state = 0;
  for (size_t t = 1; t < forward_cost_.size(); t++) {
    if (forward_cost_[t] < min_forward_cost_) {
      min_forward_cost_ = forward_cost_[t];
      best_final_state = (int)t;
    }
  }

  lag_nccf_.resize(frame_info_.size() - 1);  // will keep any existing data.
#if 0
  std::vector<std::pair<int, float> > lag_nccf_tmp;
  if (lag_nccf_.size() == 0) {
    lag_nccf_.resize(1);
  } else {
    lag_nccf_tmp.resize(lag_nccf_.size());
    for (size_t t = 0; t < lag_nccf_tmp.size(); t++) {
      lag_nccf_tmp[t] = lag_nccf_[t];
    }

    lag_nccf_.resize(frame_info_.size() - 1);
    for (size_t t = 0; t < lag_nccf_tmp.size(); t++) {
      lag_nccf_[t] = lag_nccf_tmp[t];
    }
  }
#endif



  std::vector<std::pair<int32, float> >::iterator last_iter =
    lag_nccf_.end() - 1;
  frame_info_.back()->SetBestState(best_final_state, last_iter);
  frames_latency_ =
    frame_info_.back()->ComputeLatency(opts_.max_frames_latency);
  //IDEC_INFO << "Latency is " << frames_latency_;



}


int FrontendComponent_Waveform2Pitch::OnlinePitchFeature::NumFramesReady()
const {
  return impl_->NumFramesReady();
}

FrontendComponent_Waveform2Pitch::OnlinePitchFeature::OnlinePitchFeature(
  const PitchExtractionOptions &opts)
  :impl_(new OnlinePitchFeatureImpl(opts)) {
}

bool FrontendComponent_Waveform2Pitch::OnlinePitchFeature::IsLastFrame(
  int frame) const {
  return impl_->IsLastFrame(frame);
}

void FrontendComponent_Waveform2Pitch::OnlinePitchFeature::GetFrame(
  int32 frame, std::vector<float> &feat) {
  impl_->GetFrame(frame, &feat);
}

void FrontendComponent_Waveform2Pitch::OnlinePitchFeature::AcceptWaveform(
  float sampling_rate, const std::vector<float> &waveform) {
  impl_->AcceptWaveform(sampling_rate, waveform);
}

void FrontendComponent_Waveform2Pitch::OnlinePitchFeature::InputFinished() {
  impl_->InputFinished();
}

FrontendComponent_Waveform2Pitch::OnlinePitchFeature::~OnlinePitchFeature() {
  delete impl_;
}


void FrontendComponent_Waveform2Pitch::ComputeAndProcessKaldiPitch(
  std::vector <float > &input_data) {


  // We request the first-pass features as soon as they are available,
  // regardless of whether opts.simulate_first_pass_online == true.  If
  // opts.simulate_first_pass_online == true this should
  // not affect the features generated, but it helps us to test the code
  // in a way that's closer to what online decoding would see.

  pitch_extractor->AcceptWaveform(pitch_opts_.samp_freq, input_data);
  // fake code here(to the end)
  //if (cur_offset == wave.Dim())
  //    pitch_extractor->InputFinished();



  //if (cur_frame_ == 0 && pitch_opts_.simulate_first_pass_online)
  //{
  //IDEC_WARNING << "No features output since wave file too short";
  //}
}


}

