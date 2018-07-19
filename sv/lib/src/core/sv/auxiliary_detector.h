#ifndef _SPEAKER_VERIFICATION_ABSOLUTE_SILENCE_DETECTOR_H_
#define _SPEAKER_VERIFICATION_ABSOLUTE_SILENCE_DETECTOR_H_

#include <vector>
#include <numeric>
#include "fe/frontend.h"
#include "base/log_message.h"

class  AuxiliaryDetector {

 public:
  explicit AuxiliaryDetector(int frame_length = 200,
                             int frame_shift = 80, double energy_threshold = 0) {
    wave_buff_.reserve(1024);
    is_sil_.reserve(1024);
    frames_energy_.reserve(1024);
    vad_result_.reserve(1024);
    cur_pos_ = 0;
    frame_length_ = frame_length;
    frame_shift_ = frame_shift;
    energy_threshold_ = energy_threshold;
  }

  void PushAudio(void *buf, const int buf_len,
                 const idec::IDEC_FE_AUDIOFORMAT &format) {
    // push to the buffer
    if (format == idec::FE_8K_16BIT_PCM || format == idec::FE_16K_16BIT_PCM) {
      for (int t = 0; t < int(buf_len / sizeof(short)); ++t) {
        wave_buff_.push_back(((short *)buf)[t]);
      }
    } else {
      idec::IDEC_ERROR << "unsupported wav quantization";
    }

    Process();
  }

  void Process() {
    int wave_available = wave_buff_.size() - cur_pos_;
    size_t frame_avaliable = wave_available >= frame_length_ ?
                             (wave_available - frame_length_) / (frame_shift_) + 1 : 0;

    while (frame_avaliable > 0) {
      int begin_pos = cur_pos_;
      int end_pos = cur_pos_ + frame_length_;

      unsigned long long tmp_energy = 0;
      for (int ii = begin_pos; ii < end_pos; ++ii) {
        tmp_energy += wave_buff_[ii] * wave_buff_[ii];
      }

      frames_energy_.push_back(tmp_energy);
      if (tmp_energy < energy_threshold_) {
        is_sil_.push_back(1);
      } else {
        is_sil_.push_back(0);
      }

      cur_pos_ += frame_shift_;
      --frame_avaliable;
    }
  }

  void Reset() {
    cur_pos_ = 0;
    wave_buff_.clear();
    is_sil_.clear();
    frames_energy_.clear();
    vad_result_.clear();
    wave_buff_.reserve(1024);
    is_sil_.reserve(1024);
    frames_energy_.reserve(1024);
    vad_result_.reserve(1024);
  }

  void Init(int frame_length, int frame_shift, int energy_threshold) {
    frame_length_ = frame_length;
    frame_shift_ = frame_shift;
    energy_threshold_ = energy_threshold;
  }

  void SetFrameLength(int frame_length) {
    frame_length_ = frame_length;
  }

  void SetFrameShift(int frame_shift) {
    frame_shift_ = frame_shift;
  }

  void SetEnergyThreshold(int energy_threshold) {
    energy_threshold_ = energy_threshold;
  }

  const vector<char> &GetSilenceLabel() const {
    return is_sil_;
  }

  void AccmulateVadResult(std::vector<int> &r) {
    if (r.empty()) {
      return;
    }

    int N = r.size();
    for (int i = 0; i < N; ++i) {
      vad_result_.push_back(r[i]);
    }
  }

  float GetSNR() const {
    if (vad_result_.empty()) {
      idec::IDEC_WARN <<
                      "[WARN]Vad result is empty, snr calculation needs vad result. Did you enable vad in configure?";
      return 0.0;
    }

    double signal_length = 0.0, noise_length = 0.0;
    unsigned long long signal_power = 0, noise_power = 0;

    const int N = vad_result_.size();
    for (int i = 0; i < N; ++i) {
      int frame_index = vad_result_[i];
      if (frame_index > frames_energy_.size()) {
        idec::IDEC_ERROR << "[ERROR] frame_end > wave_buf_.size(), vad_result_ is " <<
                         frame_index << ", frames_energy_ size is " << frames_energy_.size();
      }

      if (!is_sil_[i]) {
        signal_power += frames_energy_[frame_index];
        signal_length += 1;
      }
    }

    noise_length = is_sil_.size() - signal_length;
    unsigned long long total_energy = 0;
    for (int i = 0; i < frames_energy_.size(); ++i) {
      total_energy += frames_energy_[i];
    }

    noise_power = total_energy - signal_power;

    float snr = 0.0;
    if ((signal_length == 0) || (signal_power == 0)) {
      snr = -1000.0;
    } else if ((noise_length == 0) || (noise_power == 0)) {
      snr = 1000.0;
    } else {
      snr = 10 * (log10(signal_power / (signal_length*frame_length_)) - log10(
                    noise_power / (noise_length*frame_length_)));
    }
    return snr;
  }

 private:
  std::vector<float> wave_buff_;
  std::vector<char> is_sil_;
  std::vector<unsigned long long> frames_energy_;
  std::vector<int> vad_result_;
  int cur_pos_;
  int frame_length_;
  int frame_shift_;
  double energy_threshold_;

};
#endif