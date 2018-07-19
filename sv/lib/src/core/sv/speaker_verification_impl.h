#ifndef _ALS_SV_TI_H
#define  _ALS_SV_TI_H

#include <string>
#include <iterator>
#include <algorithm>
#include "mfcc.h"
#include "plda.h"
#include "config.h"
#include "new_vector.h"
#include "speaker_model.h"
#include "util/dir_utils.h"
#include "base/time_utils.h"
#include "base/log_message.h"
#include "ivector_transform.h"
#include "fe/frontend_pipeline.h"
#include "ivector_extract_pipeline_new.h"
#include "auxiliary_detector.h"

#include <fstream>

using namespace std;

enum INNER_STATUS {
  STATU_NEW = 1,
  STATU_BEGIN = 2,
  STATU_END = 3,
  STATU_VOICE = 4
};

class  SpeakerVerificationImpl {
 public:
  SpeakerVerificationImpl(const ResourceManager &res) :sv_opt_
    (res.IvectorConfigOption()),cmvn_matrix_(res.CmvnMatrix()),
    lda_matrix_(res.LdaMatrix()), res_(res),
    ivector_pipe_(IvectorExtractPipeline(res)), plda_(res.PldaConfigOption(),
        res.PldaRes()) {
    inner_status_ = STATU_NEW;
    vad_ = NULL;
    if (sv_opt_.do_vad) {
      vad_ = AlsVad::CreateFromModel(res.VadHandler());
    }

    verbose_mode_ = sv_opt_.verbose_mode;
    frame_shift_ms_ = sv_opt_.frame_shift_ms;
    int frame_length = 200, frame_shift = 80;
    GetFrameParameter(sv_opt_, frame_length, frame_shift);
    auxiliary_detector_.Init(frame_length, frame_shift, sv_opt_.energy_threshold);

    front_end_.Init(res.ConfDir(), res.SysDir());
    InputWavTypeStr2Enum(sv_opt_.str_input_type, enum_input_type_);
    InputUpdateMethodStr2Enum(sv_opt_.update_ivector_method,
                              update_ivector_method_);
  }

  ~SpeakerVerificationImpl() {
    if (sv_opt_.do_vad) {
      AlsVad::Destroy(vad_);
    }
  }

  int GetFrameParameter(const IvectorExtractOptions &sv_opt_, int &frame_length,
                        int &frame_shift) const {
    int frame_shift_ms = sv_opt_.frame_shift_ms;
    int frame_length_ms = sv_opt_.frame_length_ms;

    int sample_rate = 8;
    string str_input_str = sv_opt_.str_input_type;
    if (str_input_str == "FE_8K_16BIT_PCM") {
      sample_rate = 8;
    } else if (str_input_str == "FE_16K_16BIT_PCM") {
      sample_rate = 16;
    } else {
      idec::IDEC_ERROR << "[ERROR] unknown input type " << str_input_str;
    }

    frame_length = frame_length_ms * sample_rate;
    frame_shift = frame_shift_ms * sample_rate;
    return 0;
  }

  void BeginRegister(const char *spk_id = NULL) {
    if ((inner_status_ != STATU_NEW) && (inner_status_ != STATU_END)) {
      idec::IDEC_WARN << "Incorrect order in calling sv engine BeginRegister()." <<
                      " inner status is " << inner_status_;
    }
    inner_status_ = STATU_BEGIN;
    if (spk_id == NULL) {
      spk_guid_ = "";
    } else {
      spk_guid_ = string(spk_id, strlen(spk_id));
    }

    if (sv_opt_.do_vad) {
      vad_->BeginUtterance();
      auxiliary_detector_.Reset();
      valid_frame_index_.clear();
    }

    front_end_.BeginUtterance();
    ivector_pipe_.Clear();
    num_frm_ = 0;
    register_wave_num_ = 1;
  }

  void InputWavTypeStr2Enum(const std::string str_input_str,
                            idec::IDEC_FE_AUDIOFORMAT &enum_wav_type) const {
    enum_wav_type = idec::FE_8K_16BIT_PCM;
    if (str_input_str == "FE_8K_16BIT_PCM") {
      enum_wav_type = idec::FE_8K_16BIT_PCM;
    } else if (str_input_str == "FE_16K_16BIT_PCM") {
      enum_wav_type = idec::FE_16K_16BIT_PCM;
    } else {
      idec::IDEC_ERROR << "[ERROR] unknown input type " << str_input_str;
    }
  }

  void InputUpdateMethodStr2Enum(const std::string str_input_str,
                                 UpdateIvectorMethod &enum_update_method) const {
    enum_update_method = UPDATE_IVECTOR_BY_STATS;
    if (str_input_str == "update_by_stats") {
      enum_update_method = UPDATE_IVECTOR_BY_STATS;
    } else if (str_input_str == "update_by_mean") {
      enum_update_method = UPDATE_IVECTOR_BY_MEAN;
    } else if (str_input_str == "update_by_weighted_mean") {
      enum_update_method = UPDATE_IVECTOR_BY_WEIGHTED_MEAN;
    } else {
      idec::IDEC_ERROR << "[ERROR] unknown input type " << str_input_str;
    }
  }

  void Register(char *wave, unsigned int len) {
    if ((inner_status_ != STATU_BEGIN) && (inner_status_ != STATU_VOICE)) {
      idec::IDEC_ERROR << "[ERROR] Incorrect order in calling sv engine Register()."
                       << " inner status is " << inner_status_;
    }
    inner_status_ = STATU_VOICE;

    // do vad, get the frame_by_frame result
    const size_t wavblocksize = 3200;
    unsigned int loop = len / wavblocksize;
    if ((len % wavblocksize) != 0) {
      loop += 1;
    }

    unsigned int base_ptr = 0, frag_len = 0;
    for (int i = 0; i < loop; ++i) {
      frag_len = (i == loop - 1) ? (len % wavblocksize) : wavblocksize;
      if (frag_len == 0) {
        frag_len = wavblocksize;
      }

      front_end_.PushAudio(wave + base_ptr, frag_len, enum_input_type_);

      if (sv_opt_.do_vad) {
        auxiliary_detector_.PushAudio(wave + base_ptr, frag_len, enum_input_type_);
        DoVad(wave + base_ptr, frag_len, false);
      }
      base_ptr += frag_len;
    }
  }

  void XnnMatrix2DoubleMatrix(const idec::xnnFloatRuntimeMatrix &xnnfeat,
                              DoubleMatrix &feats, int idx, int begin_idx, int end_idx) const {
    const int gap = end_idx - begin_idx;
    if (gap <= 0) {
      idec::IDEC_ERROR << "[ERROR] end_idx not gt begin_idx, end_idx is " << end_idx
                       << " , begin_idx is " << begin_idx;
    }

    if (feats.Rows() < idx + gap) {
      idec::IDEC_ERROR << "[ERROR] feats.Rows() < idx + gap, feats.Rows() is " <<
                       feats.Rows() << " , idx + gap is " << idx + gap;
    }

    float *pTmp = NULL;
    for (int i = begin_idx, k = idx; i < end_idx; ++i, ++k) {
      pTmp = xnnfeat.Col(i);
      for (int j = 0; j < xnnfeat.NumRows(); ++j) {
        feats(k, j) = pTmp[j];
      }
    }
  }

  int ComputeScore(const SpeakerModel &model, const SpeakerModel &trail_model,
                   float &score) const {
    DoubleVector ivector, trial_ivector, transformed_ivector,
                 transformed_trial_ivector;
    int utt_num = model.GetUttNum();
    if (utt_num <= 0) {
      idec::IDEC_ERROR << "[ERROR] utt_num is " << utt_num;
    }

    model.GetIvector(ivector);
    trail_model.GetIvector(trial_ivector);
    IvectorTransform::LengthNorm(ivector);

    ivector -= res_.IvRes().mean;
    trial_ivector -= res_.IvRes().mean;

    IvectorTransform::LdaTransform(lda_matrix_, ivector, transformed_ivector);

    IvectorTransform::LdaTransform(lda_matrix_, trial_ivector,
                                   transformed_trial_ivector);

    score = plda_.LogLikelihoodRatio(transformed_ivector, utt_num,
                                     transformed_trial_ivector);
    return 0;
  }

  SpeakerModel UpdateModel(const SpeakerModel &model,
                           const SpeakerModel &model_trial) const {
    idec::IDEC_ASSERT(model.GetSpeakerId() == model_trial.GetSpeakerId());
    idec::IDEC_ASSERT(model.GetFeatureType() == model_trial.GetFeatureType());
    idec::IDEC_ASSERT(static_cast<int>(model_trial.GetUttNum()) == 1);
    double utt_num = model.GetUttNum() + model_trial.GetUttNum();
    DoubleVector ivector_transformed;
    if (update_ivector_method_ == UPDATE_IVECTOR_BY_STATS) {
      idec::IDEC_ERROR << "[ERROR] unsupport model update method "
                       <<update_ivector_method_;
    } else if ((update_ivector_method_ == UPDATE_IVECTOR_BY_MEAN)
               || (update_ivector_method_ == UPDATE_IVECTOR_BY_WEIGHTED_MEAN)) {
      DoubleVector iv, cur_iv;
      model.GetIvector(iv);
      model_trial.GetIvector(cur_iv);
      if (model.GetUttNum() == 1) {
        IvectorTransform::LengthNorm(iv);
      }
      IvectorTransform::LengthNorm(cur_iv);
      double ratio = 0.5;
      if (update_ivector_method_ == UPDATE_IVECTOR_BY_WEIGHTED_MEAN) {
        ratio = (double) model.GetFramesNum() / (model.GetFramesNum() +
                model_trial.GetFramesNum());
        ivector_transformed = iv * ratio + cur_iv * (1 - ratio);
      } else {
        ivector_transformed = (iv * model.GetUttNum() + cur_iv *
                               model_trial.GetUttNum()) * (1 / utt_num);
      }
    } else {
      idec::IDEC_ERROR << "[ERROR] unknow ivector update method.";
    }

    if (utt_num >= 32) {
      utt_num = 16;
    }

    const string &spk_id = model.GetSpeakerId();
    const string &feat_type = model.GetFeatureType();
    SpeakerModel new_model;
    new_model.UpdateModel(spk_id, feat_type, utt_num,
                          model.GetFramesNum() + model_trial.GetFramesNum(),
                          model.FeatureDim(), model.MixtureNum(), ivector_transformed);
    return new_model;
  }

  void EndAudioInputVAD() {
    DoVad(NULL, 0, true);
    vad_->EndUtterance();
  }

  void EndAudioInput() {
    DoubleMatrix feats;
    idec::xnnFloatRuntimeMatrix xnnfeat;
    PopFeat(xnnfeat);

    if (xnnfeat.NumCols() != 0) {
      feats.Resize(xnnfeat.NumCols(), xnnfeat.NumRows());
      XnnMatrix2DoubleMatrix(xnnfeat, feats, 0, 0, xnnfeat.NumCols());
    }

    front_end_.Empty();
    if (sv_opt_.do_cmvn && feats.Rows()) {
      DoCMS(feats);
    }

    num_frm_ = feats.Rows();
    ivector_pipe_.AccumulateStats(feats);
  }

  void EndRegister(SpeakerModel &spk_mdl) {
    if (inner_status_ != STATU_VOICE) {
      idec::IDEC_ERROR <<"[ERROR] Incorrect order in EndRegister()." <<
                       " inner status is " << inner_status_;
    }
    inner_status_ = STATU_END;

    front_end_.EndUtterance();
    if (sv_opt_.do_vad) {
      EndAudioInputVAD();
    } else {
      EndAudioInput();
    }

    if (verbose_mode_) {
      idec::IDEC_INFO << "Total frame accumulate is " << num_frm_;
    }

    if (num_frm_ < sv_opt_.min_frames) {
      idec::IDEC_ERROR << "[ERROR] Too short wave length of " << spk_guid_;
    }

    DoubleVector ivector;
    ivector_pipe_.ExtractIvector(ivector);
    const UtteranceStats &utt_stat = ivector_pipe_.UttStats();

    spk_mdl.UpdateModel(spk_guid_, sv_opt_.feat_type, register_wave_num_,
                        utt_stat.NumFrames(), utt_stat.FeatDim(), utt_stat.MixtureNum(), ivector);
  }

  long GetValidSpeechLength() const {
    return num_frm_ * frame_shift_ms_;
  }

  float GetSNR() const {
	  return auxiliary_detector_.GetSNR();
  }

  void AccumulateStats(const int max_to_processed = 6) {
    idec::xnnFloatRuntimeMatrix xnnfeat;
    const int N = front_end_.NumFrames();
    if (valid_frame_index_.empty()) {
      return;
    }

    int consumed_frames = valid_frame_index_.size();
    if ((0 < max_to_processed) && (max_to_processed < consumed_frames)) {
      consumed_frames = max_to_processed;
    }

    while ((consumed_frames > 0)
           && (valid_frame_index_[consumed_frames-1] >= front_end_.NumFrames())) {
      --consumed_frames;
    }

    if (consumed_frames == 0) {
      return;
    }

    vector<int> to_be_consumed;
    to_be_consumed.assign(valid_frame_index_.begin(),
                          valid_frame_index_.begin() + consumed_frames);

    auxiliary_detector_.AccmulateVadResult(to_be_consumed);
    front_end_.PeekPartFrames(to_be_consumed, xnnfeat);
    if (xnnfeat.NumCols() != 0) {
      DoubleMatrix feats(xnnfeat.NumCols(), xnnfeat.NumRows());
      XnnMatrix2DoubleMatrix(xnnfeat, feats, 0, 0, xnnfeat.NumCols());
      if (sv_opt_.do_cmvn && feats.Rows()) {
        DoCMS(feats);
      }
      ivector_pipe_.AccumulateStats(feats);
      num_frm_ += xnnfeat.NumCols();
    }
    valid_frame_index_.erase(valid_frame_index_.begin(),
                             valid_frame_index_.begin() + consumed_frames);
  }

  void DoVad(char *wave, unsigned int len, bool utterance_end) {
    int begin_frm, end_frm;
    AlsVadResult *result = NULL;
    vad_->SetData2((short *)wave, len, utterance_end);
    result = vad_->DoDetect2();
    if (result != NULL) {
      const vector<char> &abs_sil = auxiliary_detector_.GetSilenceLabel();
      for (int i = 0; i < result->num_segments; i++) {
        AlsVadSpeechBuf &buf(result->speech_segments[i]);
        begin_frm = buf.start_ms / frame_shift_ms_;
        end_frm = buf.end_ms / frame_shift_ms_;
        for (int f = begin_frm; f < end_frm; ++f) {
          if (!abs_sil[f]) {
            valid_frame_index_.push_back(f);
          }
        }
      }
      AlsVadResult_Release(&result);
      result = NULL;
    }

    if (utterance_end) {
      AccumulateStats(-1);
    } else {
      AccumulateStats();
    }
  }

  void PopFeat(idec::xnnFloatRuntimeMatrix &feat) {
    size_t N = front_end_.NumFrames();
    if (N > 0) {
      front_end_.PopNFrames(N, feat);
    }
  }

  void PeekFeat(idec::xnnFloatRuntimeMatrix &feat) {
    size_t N = front_end_.NumFrames();
    if (N > 0) {
      front_end_.PeekNFrames(N, feat);
    }
  }

  void DoCMS(DoubleMatrix &feat) const {
    if (cmvn_matrix_.Rows() != 2) {
      idec::IDEC_ERROR << "[ERROR] cmvn matrix row_num != 2 in DoCMS.";
    }

    const int dims = feat.Cols();
    if (cmvn_matrix_.Cols() != dims + 1) {
      idec::IDEC_ERROR << "[ERROR] cmvn matrix col_num != dims + 1 in DoCMS.";
    }

    const int num_frm = feat.Rows();
    for (unsigned int i = 0; i < num_frm; ++i) {
      for (unsigned int j = 0; j < dims; ++j) {
        feat(i,j) -= cmvn_matrix_(0, j);
      }
    }
  }

 private:
  AlsVad *vad_;
  Plda plda_;
  idec::IDEC_FE_AUDIOFORMAT enum_input_type_;
  const ResourceManager &res_;
  const DoubleMatrix &lda_matrix_;
  const DoubleMatrix &cmvn_matrix_;
  IvectorExtractOptions sv_opt_;
  idec::FrontendPipeline front_end_;
  IvectorExtractPipeline ivector_pipe_;
  unsigned int register_wave_num_;
  unsigned int num_frm_;
  bool verbose_mode_;
  UpdateIvectorMethod update_ivector_method_;
  std::string spk_guid_;
  int frame_shift_ms_;
  INNER_STATUS inner_status_;
  AuxiliaryDetector auxiliary_detector_;
  vector<int> valid_frame_index_;
};

#endif
