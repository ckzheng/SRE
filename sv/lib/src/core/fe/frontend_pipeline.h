// xnn-scorer/xnn-featurepipeline.h
// Copyright 2015 Alibaba-inc  [zhijie.yzj] [kuaiyi.zhw]

#ifndef FE_PIPELINE_H_
#define FE_PIPELINE_H_
#include <vector>
#include "am/xnn_runtime.h"
#include "am/xnn_kaldi_utility.h"
#include "base/log_message.h"
#include "util/parse-options.h"
#include "fe/frontend.h"
#include "fe/frontend_component_concatenator.h"
#include "fe/frontend_component_context_expansion.h"
#include "fe/frontend_component_decimate.h"
#include "fe/frontend_component_delta.h"
#include "fe/frontend_component_feature_buffer.h"
#include "fe/frontend_component_filterbank2mfcc.h"
#include "fe/frontend_component_waveform2filterbank.h"
#include "fe/frontend_component_waveform2pitch.h"


namespace idec {


class FrontendPipeline : public FrontEnd {
 protected:
  ParseOptions po_;
  std::string str_input_type_, str_output_type_;
  int frame_length_ms_;
  int frame_shift_ms_;
  int frame_length_sample_;
  int frame_shift_sample_;
  int frame_decimate_rate_;
  bool is_last_data_; 

  std::deque<float> wave_buff_;
  std::vector<FrontendComponentInterface *> pipeline_;
  FrontendComponent_FeatureBuffer feat_buff_;
  std::vector<int > component_index_receive_raw_data_;
  int frame_count_for_decibel_;
  std::vector< std::pair<int, double> > frame_decibel_index_;

 public:
  explicit FrontendPipeline(std::string name = "") :
    po_("config parser for frontend pipeline"), feat_buff_(po_) {
    // default control parameters
    str_input_type_ = "FE_8K_16BIT_PCM";
    str_output_type_ = "FE_MFCC_0_D_A+N+P";
    frame_length_ms_ = 25;    // 25ms == 200 samples @ 8kHz, 400 samples @ 16kHz
    frame_shift_ms_ = 10; // 10ms == 80 samples @ 8kHz, 160 samples @ 16kHz
    frame_decimate_rate_ = 1;  // for low frame rate, e.g. 3

    is_last_data_ = false;
    frame_count_for_decibel_ = 0;   

    po_.Register("input-type", &str_input_type_,
                 "input type (e.g. FE_RAW, FE_8K_16BIT_PCM, FE_16K_16BIT_PCM)");
    po_.Register("output-type", &str_output_type_,
                 "output type (e.g. FE_RAW, FE_MFCC_0_D_A+N+P, FE_LOGFB_D_A+N+P)");
    /*po_.Register("sample-frequency", &samp_freq_,
    	"Waveform data sample frequency (must match the waveform file, "
    	"if specified there)");*/
    po_.Register("frame-length", &frame_length_ms_, "Frame length in milliseconds");
    po_.Register("frame-shift", &frame_shift_ms_, "Frame shift in milliseconds");
  }

  virtual ~FrontendPipeline() {
    for (size_t i = 0; i < pipeline_.size(); ++i) {
      delete pipeline_[i];
    }
    component_index_receive_raw_data_.clear();
    frame_decibel_index_.clear();
  }

  virtual int  NumSamplePerFrameShift() { return frame_shift_sample_ * frame_decimate_rate_; }
  virtual int  NumSamplePerFrameWindow() { return frame_length_sample_; }
  virtual size_t  FrameShiftInMs() { return frame_shift_ms_ * frame_decimate_rate_; }

  virtual void Init(const std::string &cfg_file, const std::string sys_dir) {
    // read non-default configs in global scope
    po_.ReadConfigFile(cfg_file);
    // set parameters according to input type
    if (str_input_type_.find("FE_RAW") != std::string::npos) {
      sample_rate_ = 0;
    } else if (str_input_type_ == "FE_8K_16BIT_PCM") {
      sample_rate_ = 8000;
    } else if (str_input_type_ == "FE_16K_16BIT_PCM") {
      sample_rate_ = 16000;
    } else {
      IDEC_ERROR << "unknown input type " << str_input_type_;
    }

    frame_length_sample_ = frame_length_ms_ * sample_rate_ / 1000;
    frame_shift_sample_ = frame_shift_ms_ * sample_rate_ / 1000;

    // create pipeline components according to output feature type
    BuildPipeline();

    // read non-default configs again for each component
    po_.ReadConfigFile(cfg_file);

    // initialize all components
    for (size_t i = 0; i < pipeline_.size(); ++i) {
      pipeline_[i]->Init();
    }

    // initialize feature buffer
    feat_buff_.ConnectToPred(pipeline_.empty() ? NULL : *pipeline_.rbegin());
    feat_buff_.Init();

    // set parent class dim
    dim_ = (int)feat_buff_.OutputDim();  
  }

  virtual bool BeginUtterance() {
    // reset all components
    is_last_data_ = false;
    wave_buff_.clear();
    frame_count_for_decibel_ = 0;
    frame_decibel_index_.clear();
    for (size_t i = 0; i < pipeline_.size(); ++i) {
      pipeline_[i]->Reset();
    }
    feat_buff_.Reset();
    return(true);
  }

  virtual bool EndUtterance() {
    is_last_data_ = true;
    Process();
    for (size_t i = 0; i < pipeline_.size(); ++i) {
      pipeline_[i]->Finalize();
    }
    return(true);
  }

  virtual void PushAudio(void *buf, int buf_len, IDEC_FE_AUDIOFORMAT format) {
    // push to the buffer
    if (format == FE_8K_16BIT_PCM || format == FE_16K_16BIT_PCM) {
      for (int t = 0; t < int(buf_len / sizeof(short)); ++t) {
        wave_buff_.push_back(((short *)buf)[t]);
      }
    } else {
      IDEC_ERROR << "unsupported wav quantization";
    }

    Process();
  }

  virtual size_t PeekNFrames(size_t N, xnnFloatRuntimeMatrix &ret) {
    return(feat_buff_.PeekNFrames(N, ret));
  }

  virtual size_t PopNFrames(size_t N, xnnFloatRuntimeMatrix &ret) {
    return(feat_buff_.PopNFrames(N, ret));
  }

  /////////////////////////////add by zhuozhu.zz //////////////
  virtual size_t PeekPartFrames(size_t begin_frm, size_t end_frm, xnnFloatRuntimeMatrix &ret) {
    idec::IDEC_ASSERT(begin_frm <= end_frm);
    return(feat_buff_.PeekPartFrames(begin_frm, end_frm, ret));
  }

  virtual size_t PeekPartFrames(const std::vector<int>& frames_index, xnnFloatRuntimeMatrix &ret) {
	if (frames_index.empty()) {
		return 0;
	}
	return(feat_buff_.PeekPartFrames(frames_index, ret));
  }
  //add by zhuozhu.zz

  virtual size_t PopNFrames(size_t N) {
    return(feat_buff_.PopNFrames(N));
  }

  virtual size_t NumFrames() {
    return(feat_buff_.NumFrames());
  }

  virtual bool Empty() {
    return (feat_buff_.Empty());
  }

  static void GenerateRawFeatCfg(int dim, const char *out_cfg) {
    FILE *fp = fopen(out_cfg, "wt");
    fprintf(fp, "--input-type=FE_RAW%d\n--output-type=FE_RAW%d\n", dim, dim);
    fclose(fp);
  }

  virtual void LoadKaldiFeatureArk(std::string feat_file) {
    using namespace xnnKaldiUtility;

    std::ifstream is;
    bool binary = true;
    feat_file =
      "D:\\se\\asr\\decoder\\testdata\\fe\\data\\8k_logfb80_nodither.ark";
    is.open(feat_file.c_str(),
            binary ? std::ios::binary | std::ios::in : std::ios::in);
    if (!is.is_open())
      IDEC_ERROR << "error opening " << feat_file;

    std::string token;
    ReadToken(is, binary, &token);

    // make sure the input file is binary
    if (is.peek() != '\0')
      IDEC_ERROR << "only support kaldi binary format";
    is.get();
    if (is.peek() != 'B')
      IDEC_ERROR << "only support kaldi binary format";
    is.get();

    ReadToken(is, binary, &token);
    if (token != "FM") {
      IDEC_ERROR << ": Expected token " << "FM" << ", got " << token;
    }

    int32 rows, cols;
    ReadBasicType(is, binary, &rows);  // throws on error.
    ReadBasicType(is, binary, &cols);  // throws on error.

    std::vector<float> frame(cols);

    for (int32 i = 0; i < rows; i++) {
      is.read(reinterpret_cast<char *>(&frame[0]), sizeof(Real)*cols);
      if (is.fail()) IDEC_ERROR << "read matrix error";

      if (!pipeline_[0]->ReceiveOneFrameFromPrecedingComponent(NULL, &frame[0],
          cols))
        IDEC_ERROR <<
                   "error loading feature, try to check feature dimension in config file";

      // process
      for (size_t i = 0; i < pipeline_.size(); ++i) {
        pipeline_[i]->Process();
      }
    }

    is.close();

    // finalize
    for (size_t i = 0; i < pipeline_.size(); ++i) {
      pipeline_[i]->Finalize();
    }
  }

  inline const double decibel(int frame_index) {
    if (frame_index <= frame_count_for_decibel_) {
      int start_index = frame_decibel_index_[0].first;
      if (frame_index > start_index) {
        frame_decibel_index_.erase(frame_decibel_index_.begin(),
                                   frame_decibel_index_.begin() + (frame_index - start_index));
      }
      return frame_decibel_index_[frame_index - frame_decibel_index_[0].first].second;
    } else {
      return -100.0;
    }
  }

 protected:

  bool Process(size_t min_num_frm = 0) {
    if (wave_buff_.empty())
      return(true);

    if (!is_last_data_) {
      // process as many frames as possible, block-by-block
      size_t frame_avaliable = (int)wave_buff_.size() >= frame_length_sample_ ?
                               (wave_buff_.size() - frame_length_sample_) / (frame_shift_sample_)+1 : 0;
      while (frame_avaliable > min_num_frm) {
        size_t frame_in_block = std::min(frame_avaliable, FRAME_BLOCK);
        std::vector<float> tmp(frame_length_sample_);
        std::vector<float> tmp_pitch(frame_shift_sample_);

        // push wave frames to the first component and pitch component
        for (size_t i = 0; i < frame_in_block; ++i) {
          tmp.assign(wave_buff_.begin(), wave_buff_.begin() + frame_length_sample_);        
          double temp_energy = 0.0;
          for (int ii = 0; ii < tmp.size(); ++ii) {
            temp_energy += tmp[ii] * tmp[ii];
          }

          if (!pipeline_[0]->ReceiveOneFrameFromPrecedingComponent(NULL, &tmp[0], frame_length_sample_))
            return(false); // return on error (most likely because the input buffers of downstream components are full)

          if (component_index_receive_raw_data_.size() > 1) { // if there is pitch component
            int index = (int)component_index_receive_raw_data_.size() - 1;  // get pitch component index
            int pitch_component_index = component_index_receive_raw_data_[index];
            tmp_pitch.assign(wave_buff_.begin(), wave_buff_.begin() + frame_shift_sample_);
            if (!pipeline_[pitch_component_index]->ReceiveOneFrameFromPrecedingComponent(
                  NULL, &tmp_pitch[0], frame_shift_sample_))
              return(false); // return on error (most likely because the input buffers of downstream components are full)
          }         

          frame_decibel_index_.push_back(std::make_pair(frame_count_for_decibel_,
                                         std::max(-100.0, 10 * log10(temp_energy))));
          frame_count_for_decibel_++;
          // erase samples from beginning and calculate remaining available frames, if there are any
          wave_buff_.erase(wave_buff_.begin(), wave_buff_.begin() + frame_shift_sample_);
        }

        // process
        for (size_t i = 0; i < pipeline_.size(); ++i) {
          pipeline_[i]->Process();
        }

        frame_avaliable = (int)wave_buff_.size() >= (frame_length_sample_) ?
                          (wave_buff_.size() - frame_length_sample_) / (frame_shift_sample_)+1 : 0;
      }
    } else {
      // process as many frames as possible, block-by-block
      size_t frame_avaliable = (int)wave_buff_.size() >= frame_length_sample_ ?
                               (wave_buff_.size() - frame_length_sample_) / (frame_shift_sample_)+1 : 0;
      while (frame_avaliable > min_num_frm) {
        size_t frame_in_block = std::min(frame_avaliable, FRAME_BLOCK);
        std::vector<float> tmp(frame_length_sample_);
        std::vector<float> tmp_pitch(frame_shift_sample_);

        // push wave frames to the first component and pitch component
        for (size_t i = 0; i < frame_in_block; ++i) {
          tmp.assign(wave_buff_.begin(), wave_buff_.begin() + frame_length_sample_);
          if (!pipeline_[0]->ReceiveOneFrameFromPrecedingComponent(NULL, &tmp[0],
              frame_length_sample_))
            return(false); // return on error (most likely because the input buffers of downstream components are full)

          if (component_index_receive_raw_data_.size() > 1) { // if there is pitch component
            int index = (int)component_index_receive_raw_data_.size() -
                        1;  // get pitch component index
            int pitch_component_index = component_index_receive_raw_data_[index];
            tmp_pitch.assign(wave_buff_.begin(), wave_buff_.begin() + frame_shift_sample_);
            if (!pipeline_[pitch_component_index]->ReceiveOneFrameFromPrecedingComponent(
                  NULL, &tmp_pitch[0], frame_shift_sample_))
              return(false); // return on error (most likely because the input buffers of downstream components are full)
          }

          double temp_energy = 0.000001;
          for (int i = 0; (i < frame_length_sample_) && (i < wave_buff_.size()); ++i) {
            temp_energy += wave_buff_[i] * wave_buff_[i];
          }
          frame_decibel_index_.push_back(std::make_pair(frame_count_for_decibel_,
                                         std::max(-100.0, 10 * log10(temp_energy))));
          frame_count_for_decibel_++;
          // erase samples from beginning and calculate remaining available frames, if there are any
          wave_buff_.erase(wave_buff_.begin(), wave_buff_.begin() + frame_shift_sample_);
        }

        frame_avaliable = (int)wave_buff_.size() >= (frame_length_sample_) ?
                          (wave_buff_.size() - frame_length_sample_) / (frame_shift_sample_)+1 : 0;
      }

      if (component_index_receive_raw_data_.size() > 1) {
        // push last part data to pitch component
        int index = (int)component_index_receive_raw_data_.size() - 1;
        int pitch_component_index = component_index_receive_raw_data_[index];

        std::vector<float> tmp_pitch(frame_shift_sample_);
        int frame_num = (int)(wave_buff_.size() / frame_shift_sample_);
        if (wave_buff_.size() % frame_shift_sample_ !=
            0) // we need to push the tail samples by adding zeros
          frame_num++;
        for (int i = 0; i < frame_num; i++) {
          tmp_pitch.assign(frame_shift_sample_, 0);
          tmp_pitch.assign(wave_buff_.begin(),
                           wave_buff_.begin() + std::min(frame_shift_sample_, (int)wave_buff_.size()));

          if (!pipeline_[pitch_component_index]->ReceiveOneFrameFromPrecedingComponent(
                NULL, &tmp_pitch[0], frame_shift_sample_))
            return(false); // return on error (most likely because the input buffers of downstream components are full)

          double temp_energy = 0.000001;
          for (int i = 0; (i < frame_length_sample_) && (i < wave_buff_.size()); ++i) {
            temp_energy += wave_buff_[i] * wave_buff_[i];
          }
          frame_decibel_index_.push_back(std::make_pair(frame_count_for_decibel_,
                                         std::max(-100.0, 10 * log10(temp_energy))));
          frame_count_for_decibel_++;
          // erase samples from beginning and calculate remaining available frames, if there are any
          wave_buff_.erase(wave_buff_.begin(),
                           wave_buff_.begin() + std::min(frame_shift_sample_, (int)wave_buff_.size()));
        }
      }

      // process
      for (size_t i = 0; i < pipeline_.size(); ++i) {
        pipeline_[i]->Process();
      }

    }

    return(true);
  }

  void BuildPipeline() {
    std::string str_base_type, str_remaining_type;
    if (str_output_type_.find('+') != std::string::npos) {
      str_base_type = str_output_type_.substr(0, str_output_type_.find('+'));
      str_remaining_type = str_output_type_.substr(str_output_type_.find('+'), -1);
    } else {
      str_base_type = str_output_type_;
    }

    if (str_base_type.find("FE_RAW") != std::string::npos) {
      // load feature from file (mainly for debug)
      int dim = atoi(str_base_type.substr(str_base_type.find("FE_RAW") +
                                          strlen("FE_RAW"), -1).c_str());

      if (dim == 0)
        IDEC_ERROR << "raw feature dimension not set, use FE_RAWxx";
      FrontendComponent_FeatureBuffer *feat_buff = new
      FrontendComponent_FeatureBuffer(po_, "FE_RAW");
      feat_buff->input_dim_ = feat_buff->output_dim_ = dim;
      feat_buff->ConnectToPred(NULL);
      pipeline_.push_back(feat_buff);
    } else if (str_base_type.find("FE_LOGFB") != std::string::npos
               || str_base_type.find("FE_MFCC") != std::string::npos) {


      // basic component
      FrontendComponent_Waveform2Filterbank *wav2fb = new
      FrontendComponent_Waveform2Filterbank(po_);

      // set configs
      wav2fb->mfcc_opts_.frame_opts.samp_freq = (float)sample_rate_;
      wav2fb->input_dim_ = frame_length_sample_;
      pipeline_.push_back(wav2fb);
      component_index_receive_raw_data_.push_back((int)(pipeline_.size() - 1));

      if (str_base_type.find("FE_MFCC") != std::string::npos) {

        FrontendComponent_Filterbank2Mfcc *fb2mfcc = new
        FrontendComponent_Filterbank2Mfcc(po_);
        fb2mfcc->input_dim_ = wav2fb->output_dim_;
        fb2mfcc->ConnectToPred(*pipeline_.rbegin());
        pipeline_.push_back(fb2mfcc);


        // mfcc type (decide whether use energe)
        // 0 not use enengy  ;1 use energy
        int mfcc_type = 1; // default
        if (str_base_type.find("FE_MFCC0") != std::string::npos) {
          mfcc_type = 0;
        }

        if (1 == mfcc_type) {
          wav2fb->use_energy_ = true;
          fb2mfcc->use_energy_ = true;
        } else if (0 == mfcc_type) {
          wav2fb->use_energy_ = false;
          fb2mfcc->use_energy_ = false;
        }
      }

      // pitch
      if (str_base_type.find("_PITCH") != std::string::npos) {
        FrontendComponent_Waveform2Pitch *wav2pitch = new
        FrontendComponent_Waveform2Pitch(po_);
        wav2pitch->input_dim_ = frame_shift_sample_;
        pipeline_.push_back(wav2pitch);
        component_index_receive_raw_data_.push_back(int(pipeline_.size()) -
            1);  // value is the component index of pipeline


        // Concatenate static feature (pitch & mfcc/fbank)
        pipeline_.push_back(new FrontendComponent_Concatenator(po_));
        (*pipeline_.rbegin())->ConnectToPred(*(pipeline_.rbegin() +
                                               2)); // connect to fbank or mfcc
        (*pipeline_.rbegin())->ConnectToPred(*(pipeline_.rbegin() +
                                               1)); // connect to pitch
      }

      // deltas
      if (str_base_type.find("_Delta") != std::string::npos) {
        // have delta
        FrontendComponentInterface *pStatic = *pipeline_.rbegin();

        int order = atoi(str_base_type.substr(str_base_type.find("_Delta") +
                                              strlen("_Delta"), -1).c_str());
        std::string name;
        for (int i = 1; i <= order; ++i) {
          name += "Delta";
          FrontendComponent_Delta *delta = new FrontendComponent_Delta(po_, i, name);
          delta->ConnectToPred(pStatic);
          //delta->connectTo(*(pipeline_.rbegin()+1));
          pipeline_.push_back(delta);
        }

        pipeline_.push_back(new FrontendComponent_Concatenator(po_));
        for (int i = -1; i < order; ++i) {
          (*pipeline_.rbegin())->ConnectToPred(*(pipeline_.rbegin() + order - i));
        }
      }
    } else {
      IDEC_ERROR << "Unknown output base type " << str_base_type;
    }

    if (str_remaining_type.find("+P") != std::string::npos) {
      FrontendComponent_ContextExpansion *ctxexp = new
      FrontendComponent_ContextExpansion(po_);
      ctxexp->ConnectToPred(*pipeline_.rbegin());
      pipeline_.push_back(ctxexp);
    }
    if (str_remaining_type.find("+Dec") != std::string::npos) {
      frame_decimate_rate_ = atoi(str_remaining_type.substr(
                                    str_remaining_type.find("+Dec") + strlen("+Dec"), -1).c_str());
      FrontendComponent_Decimate *frmdec = new FrontendComponent_Decimate(po_,
          frame_decimate_rate_);
      frmdec->ConnectToPred(*pipeline_.rbegin());
      pipeline_.push_back(frmdec);
    }
  }
};

}

#endif


