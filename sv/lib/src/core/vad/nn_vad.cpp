#include <stdint.h>
#include <stdexcept>
#include <algorithm>
#include "am/xnn_net.h"
#include "fe/frontend_pipeline.h"
#include "vad/nn_vad.h"

using namespace std;
namespace idec {

NNVad::WindowDetector::WindowDetector(int window_size_time,
                                      int sil_to_speech_time,
                                      int speech_to_sil_time,
                                      int frame_size_ms) {
  frame_size_ms_ = frame_size_ms;
  win_size_frame_ = static_cast<int>(window_size_time / frame_size_ms_);
  win_sum_ = 0;
  win_state_ = reinterpret_cast<int *>(malloc(sizeof(int)*win_size_frame_));
  if (NULL == win_state_) {
    return;
  }
  memset(win_state_, 0, sizeof(int)*win_size_frame_);
  cur_win_pos_ = 0;
  pre_frame_state_ = kFrameStateSil;
  cur_frame_state_ = kFrameStateSil;
  sil_to_speech_frmcnt_thres_ = static_cast<int>(sil_to_speech_time / frame_size_ms_);
  speech_to_sil_frmcnt_thres_ = static_cast<int>(speech_to_sil_time / frame_size_ms_);
  voice_last_frame_count_ = 0;
  noise_last_frame_count_ = 0;
  hydre_frame_count_ = 0;
}

NNVad::WindowDetector::~WindowDetector() {
  if (NULL != win_state_) {
    free(win_state_);
    win_state_ = NULL;
  }
}

void NNVad::WindowDetector::Reset() {
  cur_win_pos_ = 0;
  win_sum_ = 0;
  memset(win_state_, 0, sizeof(int)* win_size_frame_);
  pre_frame_state_ = kFrameStateSil;
  cur_frame_state_ = kFrameStateSil;
  voice_last_frame_count_ = 0;
  noise_last_frame_count_ = 0;
  hydre_frame_count_ = 0;
}

int NNVad::WindowDetector::GetWinSize() {
  return win_size_frame_;
}

NNVad::AudioChangeState NNVad::WindowDetector::DetectOneFrame(FrameState frameState, int frame_count) {
  int cur_frame_state = kFrameStateSil;
  if (frameState == kFrameStateSpeech) {
    //cur_frame_state = 1;
	cur_frame_state = kFrameStateSpeech;
  } else if (frameState == kFrameStateSil) {
    //cur_frame_state = 0;
	cur_frame_state = kFrameStateSil;
  } else {
    return kChangeStateInvalid;
  }

  win_sum_ -= win_state_[cur_win_pos_];
  win_sum_ += cur_frame_state;
  win_state_[cur_win_pos_] = cur_frame_state;
  cur_win_pos_ = (cur_win_pos_ + 1) % win_size_frame_;

  if (pre_frame_state_ == kFrameStateSil && win_sum_ >= sil_to_speech_frmcnt_thres_) {
    pre_frame_state_ = kFrameStateSpeech;
    return kChangeStateSil2Speech;
  }

  if (pre_frame_state_ == kFrameStateSpeech && win_sum_ <= speech_to_sil_frmcnt_thres_) {
    pre_frame_state_ = kFrameStateSil;
    return kChangeStateSpeech2Sil;
  }

  if (pre_frame_state_ == kFrameStateSil) {
    return kChangeStateSil2Sil;
  }

  if (pre_frame_state_ == kFrameStateSpeech) {
    return kChangeStateSpeech2Speech;
  }

  return kChangeStateInvalid;  // just to rm warning,not go through here
}

AlsVad::RetCode NNVad::Init(xnnNet *xnn_net, int frame_in_ms) {
  XNNAcousticModelScorerOpt xnn_opt;
  xnn_opt.input_block_size  = vad_opts_.nn_eval_block_size;
  xnn_opt.output_block_size = vad_opts_.dcd_block_size;
  xnn_opt.ac_scale          =
    1.0f;    // deal ac scale within decoder, so do not set it.
  xnn_opt.lazy_evaluation   = false;
  if (xnn_opt.input_block_size != xnn_opt.output_block_size)
    IDEC_ERROR <<"no async decoder mode such as LC-BLSTM are supported in VAD now";

  xnn_evaluator_ = new XNNAcousticModelScorer(xnn_opt, xnn_net);
  windows_detector_ = new WindowDetector(vad_opts_.window_size_ms,
                                         vad_opts_.sil_to_speech_time_thres,
                                         vad_opts_.speech_to_sil_time_thres,
                                         frame_in_ms);
  voice_detected_callback_ = NULL;
  silence_detected_callback_ = NULL;
  voice_start_callback_ = NULL;
  voice_end_callback_ = NULL;
  noise_average_decibel_ = -100.0;
  frm_cnt_ = 0;
  ResetDetection();
  return kVadSuccess;
}
void NNVad::ResetDetection() {
  continous_silence_frame_count_ = 0;
  latest_confirmed_speech_frame_ = 0;
  lastest_confirmed_silence_frame_ = -1;
  confirmed_start_frame_ = -1;
  confirmed_end_frame_ = -1;
  vad_state_machine_ = kVadInStateStartPointNotDetected;
  windows_detector_->Reset();
}
bool NNVad::SetData(int16_t pcm[], int size_in_bytes) {
  is_new_api_enable_ = false;
  if (size_in_bytes > 0) {
    if (fe_.GetSampleRate() != vad_opts_.sample_rate) {
      IDEC_ERROR <<
                 "mismatch feature extraction and VAD sample rate setting: vad = " <<
                 vad_opts_.sample_rate << "FE of VAD is" <<
                 fe_.GetSampleRate();
    }
    if (vad_opts_.sample_rate == kVadSampleRate8K) {
      fe_.PushAudio(pcm, size_in_bytes, idec::FE_8K_16BIT_PCM);
    } else if (vad_opts_.sample_rate == kVadSampleRate16K) {
      fe_.PushAudio(pcm, size_in_bytes, idec::FE_16K_16BIT_PCM);
    } else {
      IDEC_ERROR << "un_supported data\n";
    }
    return true;
  } else {
    IDEC_ERROR << "error:input pcm data size  %d less than 0\n" <<
               size_in_bytes;
    return false;
  }
}

bool NNVad::SetParam(const char *name, const char *value) {
  return vad_opts_.SetParam(name, value);
}

int NNVad::DoDetect(bool final_frames) {
  is_new_api_enable_ = false;
  int num_frame = 0;
  if (!final_frames) {
    num_frame =  DetectCommonFrames();
  } else {
    num_frame = DetectLastFrames();
  }
  return num_frame;
}
int NNVad::DetectCommonFrames() {
  if (vad_state_machine_ == kVadInStateEndPointDetected) {
    return 0;
  }
  FrameState frame_state = kFrameStateInvalid;
  while (fe_.NumFrames() > vad_opts_.nn_eval_block_size) {
    // pop feature and push to evaluator
    fe_.PopNFrames(vad_opts_.nn_eval_block_size, feat_);
    xnn_evaluator_->PushFeatures(frm_cnt_, feat_);

    // get likelihood
    for (int t = frm_cnt_;
         t < frm_cnt_ + static_cast<int>(feat_.NumCols()); ++t) {
      double cur_decibel = fe_.decibel(t);
      double cur_snr = cur_decibel - noise_average_decibel_;
      // for each frame, calc log posterior probability of each state
      float speech_prob = xnn_evaluator_->GetFrameScore(t, 1);
      float noise_prob = vad_opts_.speech_2_noise_ratio *
                         xnn_evaluator_->GetFrameScore(t, 0);

      if (exp(speech_prob) >= exp(noise_prob) + vad_opts_.speech_noise_thres_) {
        if (cur_snr >= vad_opts_.snr_thres_ && cur_decibel >= vad_opts_.decibel_thres_) {
          frame_state = kFrameStateSpeech;
        } else {
          frame_state = kFrameStateSil;
        }
      } else {
        frame_state = kFrameStateSil;
        if (noise_average_decibel_ < -99.9) {
          noise_average_decibel_ = cur_decibel;
        } else {
          noise_average_decibel_ = (cur_decibel + noise_average_decibel_ * (vad_opts_.noise_frame_num_used_for_snr_ - 1))/ vad_opts_.noise_frame_num_used_for_snr_;
        }
      }
      DetectOneFrame(frame_state, t, false);
    }
    frm_cnt_ += static_cast<int>(feat_.NumCols());
  }
  return 0;
}

int NNVad::DetectLastFrames() {
  if (vad_state_machine_ == kVadInStateEndPointDetected) {
    return 0;
  }
  fe_.EndUtterance();
  // evaluation block by block to reduce the memory usage in
  // pure batch-mode
  while (fe_.NumFrames() > vad_opts_.nn_eval_block_size) {
    fe_.PopNFrames(fe_.NumFrames(), feat_);
    xnn_evaluator_->PushFeatures(frm_cnt_, feat_);
    FrameState frame_state_decision = kFrameStateInvalid;
    // frame-by-frame-decision
    for (int t = frm_cnt_; t < frm_cnt_ + static_cast<int>(feat_.NumCols()); ++t) {
      double cur_decibel = fe_.decibel(t);
      double cur_snr = cur_decibel - noise_average_decibel_;
      if (xnn_evaluator_->GetFrameScore(t, 1) >= xnn_evaluator_->GetFrameScore(t, 0)) {
        if (cur_snr >= vad_opts_.snr_thres_ && cur_decibel >= vad_opts_.decibel_thres_) {
          frame_state_decision = kFrameStateSpeech;
        } else {
          frame_state_decision = kFrameStateSil;
        }
      } else {
        frame_state_decision = kFrameStateSil;
        if (noise_average_decibel_ < -99.9) {
          noise_average_decibel_ = cur_decibel;
        } else {
          const int noise_frame_num_for_snr = vad_opts_.noise_frame_num_used_for_snr_;
		  // average continue n frames decibel for calculating noise average decibel
		  noise_average_decibel_ = (cur_decibel + noise_average_decibel_ * (noise_frame_num_for_snr - 1)) / noise_frame_num_for_snr;
        }
      }
      DetectOneFrame(frame_state_decision, t, (t == (frm_cnt_ + static_cast<int>(feat_.NumCols()) - 1)));
    }
    frm_cnt_ += static_cast<int>(feat_.NumCols());
  }
  return 0;
}

void NNVad::DetectOneFrame(FrameState cur_frm_state, int cur_frm_idx, bool is_final_frame) {
  NNVad::AudioChangeState state_change = windows_detector_->DetectOneFrame(cur_frm_state, cur_frm_idx);
  int frm_shift_in_ms = static_cast<int>(fe_.FrameShiftInMs());
  if (kChangeStateSil2Speech == state_change) {
    int silence_frame_count = continous_silence_frame_count_;
    continous_silence_frame_count_ = 0;
    int start_frame = 0;
    switch (vad_state_machine_) {
    case idec::kVadInStateStartPointNotDetected:
      start_frame = std::max(static_cast<int>(data_buf_start_frame_), cur_frm_idx - LatencyFrmNumAtStartPoint());
      OnVoiceStart(start_frame);
      vad_state_machine_ = kVadInStateInSpeechSegment;
      for (int t = start_frame + 1; t <= cur_frm_idx; t++) {
        OnVoiceDetected(t);
      }
      break;
    case idec::kVadInStateInSpeechSegment:
      // happen if the window detector output intermediate
      // silence decision, but longer enough
      for (int t = latest_confirmed_speech_frame_ + 1; t < cur_frm_idx; t++) {
        OnVoiceDetected(t);
      }
      // the max speech timeout here
      if (cur_frm_idx - confirmed_start_frame_ + 1 >
          vad_opts_.max_single_segment_time / frm_shift_in_ms) {
        OnVoiceEnd(cur_frm_idx, false, false);
        vad_state_machine_ = kVadInStateEndPointDetected;
      } else if (!is_final_frame) {
        // treat as regular speech frame, emit immediately
        OnVoiceDetected(cur_frm_idx);
      } else {  // hit last frame
        MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx);
      }
      break;
    case idec::kVadInStateEndPointDetected:
      break;
    default:
      break;
    }
  } else if (kChangeStateSpeech2Sil == state_change) {
    continous_silence_frame_count_ = 0;
    switch (vad_state_machine_) {
    case idec::kVadInStateStartPointNotDetected:
      break;
    case idec::kVadInStateInSpeechSegment:
      // the max speech timeout here
      if (cur_frm_idx - confirmed_start_frame_ + 1 >
          vad_opts_.max_single_segment_time / 10) {
        OnVoiceEnd(cur_frm_idx, false, false);
        vad_state_machine_ = kVadInStateEndPointDetected;
      } else if (!is_final_frame) {
        // regular speech frame, emit immediately
        OnVoiceDetected(cur_frm_idx);
      } else {  // hit last frame
        MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx);
      }
      break;
    case idec::kVadInStateEndPointDetected:
      break;
    default:
      break;
    }
  } else if (kChangeStateSpeech2Speech == state_change) {
    continous_silence_frame_count_ = 0;
    switch (vad_state_machine_) {
    case idec::kVadInStateStartPointNotDetected:
      break;
    case idec::kVadInStateInSpeechSegment:
      // the max speech timeout here
      if (cur_frm_idx - confirmed_start_frame_ + 1 >
          vad_opts_.max_single_segment_time / 10) {
        OnVoiceEnd(cur_frm_idx, false, false);
        vad_state_machine_ = kVadInStateEndPointDetected;
      } else if (!is_final_frame) {  // regular speech frame, emit immediately
        OnVoiceDetected(cur_frm_idx);
      } else {  // hit last frame
        MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx);
      }
      break;
    case idec::kVadInStateEndPointDetected:
      break;
    default:
      break;
    }
  } else if (kChangeStateSil2Sil == state_change) {
    continous_silence_frame_count_++;
    switch (vad_state_machine_) {
    case idec::kVadInStateStartPointNotDetected:
      // silence timeout, return zero length decision
      if (((vad_opts_.detect_mode == kVadSingleUtteranceDetectMode) &&
           ((continous_silence_frame_count_ * frm_shift_in_ms) >
            vad_opts_.max_start_silence_time))
          || (is_final_frame && (number_end_time_detected_ == 0))
         ) {
        // the silence decision about the silence segment
        for (int t = lastest_confirmed_silence_frame_ + 1;
             t < cur_frm_idx; t++) {
          OnSilenceDetected(t);
        }
        OnVoiceStart(0, true);
        OnVoiceEnd(0, true, false);
        vad_state_machine_ = kVadInStateEndPointDetected;
      } else {
        // pretty sure about the silence at the sentence start
        if (cur_frm_idx >= LatencyFrmNumAtStartPoint()) {
          OnSilenceDetected(cur_frm_idx - LatencyFrmNumAtStartPoint());
        }
      }
      break;
    case idec::kVadInStateInSpeechSegment:
      // note that we are not sure about this segment now
      // 1) if the total silence length exceed the threshold, then it maybe
      //    the silence
      // 2) otherwise this frame will be the silence after the end point
      // regular end time detection, silence length exceed the end sensitivity
      if ((continous_silence_frame_count_ * frm_shift_in_ms) >=
          vad_opts_.max_end_silence_time) {
        int lookback_frame =
          static_cast<int>(vad_opts_.max_end_silence_time / frm_shift_in_ms);
        if (vad_opts_.do_extend) {
          lookback_frame -=
            static_cast<int>(vad_opts_.lookahead_time_end_point /
                             frm_shift_in_ms);
          lookback_frame -= 1;
          lookback_frame = std::max(0, lookback_frame);
        }
        OnVoiceEnd(cur_frm_idx - lookback_frame, false, false);
        vad_state_machine_ = kVadInStateEndPointDetected;
      } else if (cur_frm_idx - confirmed_start_frame_ + 1 >
                 vad_opts_.max_single_segment_time / frm_shift_in_ms) {
        // the max speech timeout here
        OnVoiceEnd(cur_frm_idx, false, false);
        vad_state_machine_ = kVadInStateEndPointDetected;
      } else if (vad_opts_.do_extend && !is_final_frame) {
        // eliminate the end point latency caused by lookahead in end point
        // detection.
        if (continous_silence_frame_count_ <=
            static_cast<int>(vad_opts_.lookahead_time_end_point / frm_shift_in_ms)) {
          OnVoiceDetected(cur_frm_idx);
        }
      } else {
        MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx);
      }
      break;
    case idec::kVadInStateEndPointDetected:
      break;
    default:
      break;
    }
  }
  // automatically change back into un-detected state;
  // note that we do not reset the frontend and frame count
  if (vad_state_machine_ == kVadInStateEndPointDetected
      && vad_opts_.detect_mode == kVadMutipleUtteranceDetectMode) {
    ResetDetection();
  }
}

void NNVad::Uninit() {
  if (NULL != windows_detector_) {
    delete windows_detector_;
    windows_detector_ = NULL;
  }
  if (NULL != xnn_evaluator_) {
    delete xnn_evaluator_;
    xnn_evaluator_ = NULL;
  }
}

void NNVad::Destroy() {
  if (NULL != windows_detector_) {
    delete windows_detector_;
    windows_detector_ = NULL;
  }
  if (NULL != xnn_evaluator_) {
    delete xnn_evaluator_;
    xnn_evaluator_ = NULL;
  }
}
int NNVad::CheckParams() {
  if (vad_opts_.sample_rate != kVadSampleRate8K &&
      vad_opts_.sample_rate != kVadSampleRate16K) {
    IDEC_ERROR << "error: sample_rate = " << vad_opts_.sample_rate <<
               ", must be 8000 or 16000 !\n";
    return kVadUnsupportedSampleRate;
  }
  if (vad_opts_.max_end_silence_time <= 0) {
    IDEC_ERROR << "error: max_end_silence_time = " <<
               vad_opts_.max_end_silence_time << ", must be greater than 0~\n";
  }
  if (vad_opts_.max_start_silence_time <= 0) {
    IDEC_ERROR << "error: max_start_silence_time = " <<
               vad_opts_.max_start_silence_time << ", must be greater than 0~\n";
  }
  if (vad_opts_.detect_mode != 0 && vad_opts_.detect_mode != 1) {
    IDEC_ERROR << "error: detect_mode = " << vad_opts_.detect_mode <<
               ", must be 0 or 1\n";
  }
  if (vad_opts_.window_size_ms <= 0) {
    IDEC_ERROR << "error: window_size = " << vad_opts_.window_size_ms <<
               ", must be greater than 0~\n";
  }
  if (vad_opts_.sil_to_speech_time_thres > vad_opts_.window_size_ms) {
    IDEC_ERROR << "error: sil_to_speech_thres = " <<
               vad_opts_.sil_to_speech_time_thres << ", greater than window_size" <<
               vad_opts_.window_size_ms << "\n";
  }
  if (vad_opts_.speech_to_sil_time_thres > vad_opts_.window_size_ms) {
    IDEC_ERROR << "error: speech_to_sil_thres = " <<
               vad_opts_.speech_to_sil_time_thres << ", greater than window_size" <<
               vad_opts_.window_size_ms << "\n";
  }
  if (vad_opts_.speech_2_noise_ratio > 2
      || vad_opts_.speech_2_noise_ratio < 1 ) {
    IDEC_ERROR << "error: speech_2_noise_ratio = " <<
               vad_opts_.speech_2_noise_ratio << ", should be between 1 and 2" <<  "\n";
  }
  if (vad_opts_.do_extend != 0
      && vad_opts_.do_extend != 1) {
    IDEC_ERROR << "error: extend_time_flag = " <<
               vad_opts_.do_extend << ", should be 0 or 1" << "\n";
  }
  if (vad_opts_.lookback_time_start_point < 10) {
    IDEC_ERROR << "error: lookback_time_start_point = " <<
               vad_opts_.lookback_time_start_point <<
               ", should be greater than 10 ms \n";
  }
  if (vad_opts_.lookahead_time_end_point < 10) {
    IDEC_ERROR << "error: lookahead_time_end_point = " <<
               vad_opts_.lookahead_time_end_point <<
               ", should be greater than 10 ms\n";
  }
  if (vad_opts_.max_single_segment_time < 500
      && vad_opts_.max_single_segment_time > 120000) {
    IDEC_ERROR << "error: max_single_segment_time = " <<
               vad_opts_.max_single_segment_time <<
               ", should be between [500-120, 000]" << "\n";
  }
  return kVadSuccess;
}
bool NNVad::SetSampleRate(int sample_rate) {
  if (sample_rate == 0) {
    sample_rate = kVadSampleRate8K;
  }
  if (sample_rate == 1) {
    sample_rate = kVadSampleRate16K;
  }
  if (sample_rate != kVadSampleRate8K && sample_rate != kVadSampleRate16K) {
    IDEC_ERROR << "Invalid pcm sampleRate, only support 8000 or 16000 !\n";
    return false;
  }
  vad_opts_.sample_rate = sample_rate;
  return true;
}
int NNVad::GetSampleRate() {
  return vad_opts_.sample_rate;
}
bool NNVad::SetEndSilence(int duration) {
  if (duration <= 0) {
    IDEC_ERROR << "error: max_end_silence_time" <<
               vad_opts_.max_end_silence_time << " must be greater than 0~\n";
    return false;
  }
  vad_opts_.max_end_silence_time = duration;
  return true;
}
bool NNVad::SetStartSilence(int duration) {
  if (duration <= 0) {
    IDEC_ERROR << "error: max_start_silence_time" <<
               vad_opts_.max_start_silence_time << " must be greater than 0~\n";
    return false;
  }
  vad_opts_.max_start_silence_time = duration;
  return true;
}
bool NNVad::SetMaxSpeechTimeout(int duration) {
  vad_opts_.max_single_segment_time = duration;
  return true;
}
bool NNVad::SetWindowSize(int window_size) {
  if (window_size <= 0) {
    IDEC_ERROR << "error: window_size" << vad_opts_.window_size_ms <<
               " must be greater than 0~\n";
    return false;
  }
  vad_opts_.window_size_ms = window_size;
  return true;
}
bool NNVad::SetSil2SpeechThres(int sil_to_speech_time) {
  if (sil_to_speech_time <= 0) {
    IDEC_ERROR << "error: sil_to_speech_thres" <<
               vad_opts_.sil_to_speech_time_thres << " must be greater than 0\n";
    return false;
  }
  vad_opts_.sil_to_speech_time_thres = sil_to_speech_time;
  return true;
}
bool NNVad::SetSpeech2SilThres(int speech_to_sil_time) {
  if (speech_to_sil_time <= 0) {
    IDEC_ERROR << "error: speech_to_sil_time" <<
               vad_opts_.speech_to_sil_time_thres << " must be greater than 0\n";
    return false;
  }
  vad_opts_.speech_to_sil_time_thres = speech_to_sil_time;
  return true;
}
bool NNVad::SetStartDetect(bool enable) {
  vad_opts_.do_start_point_detection = enable;
  return true;
}
bool NNVad::SetEndDetect(bool enable) {
  vad_opts_.do_end_point_detection = enable;
  return true;
}
bool NNVad::SetDetectMode(int mode) {
  if (mode < 0) {
    IDEC_ERROR << "error: detect-mode" << vad_opts_.detect_mode <<
               " must be greater than 0 or eq 0\n";
    return false;
  }
  vad_opts_.detect_mode = mode;
  return true;
}
void NNVad::EnableVoiceStartDetect() {
  vad_opts_.do_start_point_detection = true;
}
void NNVad::DisableVoiceStartDetect() {
  vad_opts_.do_start_point_detection = false;
}
void NNVad::EnableVoiceStopDetect() {
  vad_opts_.do_end_point_detection = true;
}
void NNVad::DisableVoiceStopDetect() {
  vad_opts_.do_end_point_detection = false;
}
int NNVad::GetLatency() {
  return LatencyFrmNumAtStartPoint() * static_cast<int>(fe_.FrameShiftInMs());
}
int NNVad::LatencyFrmNumAtStartPoint() {
  int vad_latency = windows_detector_->GetWinSize();
  if (vad_opts_.do_extend) {
    vad_latency +=
      vad_opts_.lookback_time_start_point /
      static_cast<int>(fe_.FrameShiftInMs());
  }
  return vad_latency;
}
bool NNVad::SetLatency(int len) {
  throw std::runtime_error("not worked now");
  return true;
}
void NNVad::BeginUtterance() {
  fe_.BeginUtterance();
  frm_cnt_ = 0;
  number_end_time_detected_ = 0;
  noise_average_decibel_ = -100.0;
  if (is_new_api_enable_) {
    // for output the data
    data_buf_start_frame_ = 0;
    data_buf_.clear();
    ResetOutputBuf();
  }
  ResetDetection();
}
void NNVad::EndUtterance() {
}
void NNVad::OnSilenceDetected(int valid_frame) {
  lastest_confirmed_silence_frame_ = valid_frame;
  // discard the useless data from the buffer
  if (is_new_api_enable_ &&
      vad_state_machine_ == kVadInStateStartPointNotDetected) {
    PopDataBufTillFrame(valid_frame);
  }
  if (silence_detected_callback_ != NULL) {
    silence_detected_callback_(silence_detected_param_, valid_frame);
  }
}
void NNVad::OnVoiceDetected(int valid_frame) {
  // latest speech frame emitted
  if (latest_confirmed_speech_frame_ != 0 &&
      latest_confirmed_speech_frame_ + 1 != valid_frame) {
    IDEC_WARNING << "something wrong with the voice emission";
  }
  latest_confirmed_speech_frame_ = valid_frame;
  if (voice_detected_callback_ != NULL) {
    voice_detected_callback_(voice_detected_param_, valid_frame);
  }
  // pop out the speech data of one frame
  if (is_new_api_enable_) {
    PopDataToOutputBuf(valid_frame, 1, false, false, false);
  }
}
void NNVad::OnVoiceStart(int start_frame, bool fake_result) {
  if (voice_start_callback_ != NULL && vad_opts_.do_start_point_detection) {
    voice_start_callback_(voice_start_param_, start_frame);
  }
  if (confirmed_start_frame_ != -1) {
    IDEC_WARNING << "not reset vad properly";
  } else {
    confirmed_start_frame_ = start_frame;
  }
  // pop out the data of start frame
  if (!fake_result && is_new_api_enable_ &&
      vad_state_machine_ == kVadInStateStartPointNotDetected) {
    PopDataToOutputBuf(confirmed_start_frame_, 1, true, false, false);
  }
}
void NNVad::OnVoiceEnd(int end_frame, bool fake_result, bool is_last_frame) {
  // emitting the speech frame information
  for (int t = latest_confirmed_speech_frame_ + 1; t < end_frame; t++) {
    OnVoiceDetected(t);
  }
  // emit the voice-end-point information
  if (voice_end_callback_ != NULL && vad_opts_.do_end_point_detection) {
    voice_end_callback_(voice_end_param_, end_frame);
  }
  if (confirmed_end_frame_ != -1) {
    IDEC_WARNING << "not reset vad properly";
  } else {
    confirmed_end_frame_ = end_frame;
  }
  // pop out the data of end frame
  if (!fake_result && is_new_api_enable_) {
    PopDataToOutputBuf(confirmed_end_frame_, 1, false, true, is_last_frame);
  }
  // used for multiple utterance mode
  number_end_time_detected_++;
}
void NNVad::MaybeOnVoiceEndIfLastFrame(bool is_final_frame, int cur_frm_idx) {
  if (is_final_frame) {
    OnVoiceEnd(cur_frm_idx, false, true);
    vad_state_machine_ = kVadInStateEndPointDetected;
  }
}
// new set of api function
bool NNVad::SetData2(int16_t *pcm, int size_in_bytes, bool final_send) {
  is_new_api_enable_ = true;
  is_final_send_ = final_send;
  if (size_in_bytes > 0) {
    if (fe_.GetSampleRate() != vad_opts_.sample_rate) {
      IDEC_ERROR << "mismatch sample rate setting: vad = " <<
                 vad_opts_.sample_rate << ", feature of VAD is" <<
                 fe_.GetSampleRate();
    }
    // get the audio format
    idec::IDEC_FE_AUDIOFORMAT audio_format = FE_NULL_AUDIOFORMAT;
    if (vad_opts_.sample_rate == kVadSampleRate8K) {
      audio_format = idec::FE_8K_16BIT_PCM;
    } else if (vad_opts_.sample_rate == kVadSampleRate16K) {
      audio_format = idec::FE_16K_16BIT_PCM;
    }
    if (audio_format == idec::FE_8K_16BIT_PCM ||
        audio_format == idec::FE_16K_16BIT_PCM) {
      fe_.PushAudio(pcm, size_in_bytes, audio_format);
      if (vad_opts_.is_new_api_enable) {
        for (int t = 0; t < static_cast<int>(size_in_bytes / sizeof(int16_t)); ++t) {
          data_buf_.push_front(pcm[t]);
        }
      }
    } else {
      IDEC_ERROR << "unsupported data\n";
    }
    return true;
  }
  return true;
}
AlsVadResult *NNVad::DoDetect2() {
  is_new_api_enable_ = true;
  ResetOutputBuf();
  if (!is_final_send_) {
    DetectCommonFrames();
  } else {
    DetectLastFrames();
  }
  return CopyOutputBufToApi();
}
void NNVad::ResetOutputBuf() {
  output_data_buf_.resize(0);
}
// discard the data till frame_idx (including)
void NNVad::PopDataBufTillFrame(int frame_idx) {
  if (!is_new_api_enable_)
    return;
  while (data_buf_start_frame_ < static_cast<uint32>(frame_idx)) {
    if (data_buf_.size() >= fe_.NumSamplePerFrameShift()) {
      for (int i = 0; i < fe_.NumSamplePerFrameShift(); i++) {
        data_buf_.pop_back();
      }
      data_buf_start_frame_++;
    }
  }
}
// pop the data from the buffer into a segment
// @first_frm_is_start_point, if the first frame is the start point?
// @last_frm_is_end_point, if the last frame is the end point?
// @end_point_is_last_frame, if the last frame is indeed end point,
//  if it is the last frame of whole utterance?
void NNVad::PopDataToOutputBuf(int start_frm, int frm_cnt,
                               bool first_frm_is_start_point,
                               bool last_frm_is_end_point,
                               bool end_point_is_sent_end) {
  if (!is_new_api_enable_) {
    return;
  }
  // pop out the useless data first
  PopDataBufTillFrame(start_frm);
  int expected_sample_number =
    frm_cnt * fe_.NumSamplePerFrameShift();
  if (last_frm_is_end_point) {
    int extra_sample = (fe_.NumSamplePerFrameWindow() -
                        fe_.NumSamplePerFrameShift());
    extra_sample = std::max(0, extra_sample);  // fix for low frame rate
    expected_sample_number += extra_sample;
  }
  if (end_point_is_sent_end) {
    expected_sample_number =
      std::max(expected_sample_number, static_cast<int>(data_buf_.size()));
  }
  if (data_buf_.size() < (size_t)expected_sample_number) {
    IDEC_ERROR << "error in calling pop_data_buf";
  }
  // check if we should start a new segment, do it when
  // 1) when the output_data_buf_ is reset at calling Detect2()
  // 2) we detect a new segment start;
  if (output_data_buf_.size() == 0 || first_frm_is_start_point) {
    output_data_buf_.resize(output_data_buf_.size()+1);
    output_data_buf_.back().Reset();
    output_data_buf_.back().start_ms
      = output_data_buf_.back().end_ms
        = start_frm * static_cast<uint32>(fe_.FrameShiftInMs());
  }
  // check if something wrong with the algorithm
  NNVadSpeechBuf &cur_seg(output_data_buf_.back());
  if (cur_seg.end_ms != start_frm * fe_.FrameShiftInMs()) {
    IDEC_ERROR << "something wrong with the VAD algorithm";
  }
  size_t out_pos = cur_seg.buffer.size();
  int sample_cpy_out = 0;
  cur_seg.buffer.resize((size_t)expected_sample_number + cur_seg.buffer.size());
  // copy & pop out the data
  int data_to_pop = end_point_is_sent_end ? expected_sample_number :
                    (frm_cnt * fe_.NumSamplePerFrameShift());
  for (sample_cpy_out = 0; sample_cpy_out < data_to_pop; sample_cpy_out++) {
    cur_seg.buffer[out_pos++] = data_buf_.back();
    data_buf_.pop_back();
  }
  // copy the tail data, but not pop them out
  for (sample_cpy_out = data_to_pop;
       sample_cpy_out < expected_sample_number; sample_cpy_out++) {
    cur_seg.buffer[out_pos++] = data_buf_.back();
  }
  if (cur_seg.end_ms != start_frm * fe_.FrameShiftInMs()) {
    IDEC_ERROR << "something wrong with the VAD algorithm";
  }
  // fix other information, end_time etc.
  data_buf_start_frame_ += static_cast<uint32>(frm_cnt);
  cur_seg.end_ms =
    (start_frm + frm_cnt) *
    static_cast<uint32>(fe_.FrameShiftInMs());
  if (first_frm_is_start_point) {
    cur_seg.contain_seg_start_point = true;
  }
  if (last_frm_is_end_point) {
    cur_seg.contain_seg_end_point = true;
  }
}
AlsVadResult *NNVad::CopyOutputBufToApi() {
  if (!is_new_api_enable_ || output_data_buf_.size() == 0)
    return NULL;
  // allocate the speech segment
  AlsVadResult *result = new AlsVadResult();
  result->num_segments = static_cast<int>(output_data_buf_.size());
  result->speech_segments = new AlsVadSpeechBuf[result->num_segments];

  // copy the data into the output segment
  for (int i = 0; i < result->num_segments; i++) {
    NNVadSpeechSegmentToAlsVadSpeechSegment(output_data_buf_[i],
                                            result->speech_segments[i]);
  }
  return result;
}
void NNVad::NNVadSpeechSegmentToAlsVadSpeechSegment(NNVadSpeechBuf &src,
    AlsVadSpeechBuf &dst) {
  dst.contain_seg_start_point = src.contain_seg_start_point;
  dst.contain_seg_end_point = src.contain_seg_end_point;
  dst.start_ms = src.start_ms;
  dst.end_ms = src.end_ms;
  dst.data = new int16_t[src.buffer.size()];
  dst.data_len = static_cast<uint32>(src.buffer.size()) * sizeof(int16_t);
  memcpy(dst.data, &(src.buffer[0]), dst.data_len);
}
void NNVad::FreeApiOutputBuf(AlsVadResult *&result) {
  if (result == NULL)
    return;
  for (int i = 0; i < result->num_segments; i++) {
    delete[](reinterpret_cast<int16_t *>(result->speech_segments[i].data));
  }
  delete []result->speech_segments;
  delete result;
  result = NULL;
}
}  // namespace idec

