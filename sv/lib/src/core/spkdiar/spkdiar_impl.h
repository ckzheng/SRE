#ifndef _SPEAKER_DIARIZATION_IMPL_H_
#define _SPEAKER_DIARIZATION_IMPL_H_

#include <vector>
#include "mfcc.h"
#include "ahc.h"
#include "bic.h"
#include "vad.h"
#include "spkdiar_tunning.h"
#include "seg.h"
#include "speaker_cluster.h"
#include "spkdiar_serialize.h"


namespace alspkdiar {

class  SpeakerDiarization {
 public:
  SpeakerDiarization(ResourceManager *handler) {
    mfcc_ = new Mfcc(handler->SysDir().c_str(), handler->ConfDir().c_str(), handler->VerboseMode(), handler->StrInputType());
    vad_ = new Vad(AlsVad::CreateFromModel(handler->VadHandler()));
    bic_ = new Bic(handler->BicConfigOption());
    ahc_ = new Ahc(mfcc_, bic_, handler);
    spkdiar_tunning_ = new SpkDiarTunning(handler->UbmRes(),
                                          handler->GmmUpdateOption(), handler->HmmConfigOption());
    serial_ = new Serialize(handler->StrInputType());
    verbose_mode_ = handler->VerboseMode();
    use_bic_ = handler->BicConfigOption().use_bic;
    use_ivector_ = (handler->AhcConfigOptions().class_number > 2);
    if (use_ivector_) {
      mfcc_->Init("", handler->FrontendConfigOption().fe_conf_path.c_str());
    }
  }

  ~SpeakerDiarization() {
    idec::IDEC_DELETE(vad_);
    idec::IDEC_DELETE(mfcc_);
    idec::IDEC_DELETE(bic_);
    idec::IDEC_DELETE(ahc_);
    idec::IDEC_DELETE(serial_);
    idec::IDEC_DELETE(spkdiar_tunning_);
  }

  int BeginSpkDiar() {
    return ALS_OK;
  }

  int EndSpkDiar() {
    return ALS_OK;
  }

  int AverageSegmentation(SegCluster &segs,
                          SpeakerCluster &spk_cluster, int win_size, int shift_size) {
    idec::IDEC_ASSERT(win_size > shift_size);
    if (spk_cluster.Size() != 0) {
      spk_cluster.Clear();
    }

    SegCluster cluster;
    for (int i = 0; i < segs.Size(); ++i) {
      const Seg &seg = segs.GetSeg(i);
      unsigned int N = seg.Length();
      if (N < 2 * win_size) {
        cluster.Add(seg);
        spk_cluster.Add(cluster);
        cluster.Clear();
      }

      unsigned int start = seg.begin;
      unsigned int end = seg.end;
      unsigned int start1 = start;
      unsigned int end1 = start1 + win_size - 1;

      while (end1 + win_size < end) {
        cluster.Add(Seg(start1, end1));
        spk_cluster.Add(cluster);
        cluster.Clear();
        start1 += shift_size;
        end1 += shift_size;
      }

      cluster.Add(Seg(start1, end));
      spk_cluster.Add(cluster);
      cluster.Clear();
    }
    return ALS_OK;
  }


  int TurnDetect(Bic &bic, const SegCluster &segs,
                 SpeakerCluster &spk_cluster) {
    int begin = 0, end = 0;
    //Seg *segment = NULL;
    std::vector<int> points;
    points.reserve(128);
    const int min_length = 100;
    const int min_seg_length = 50;
    for (unsigned int i = 0; i < segs.Size(); ++i) {
      const Seg &seg = segs.GetSeg(i);
      if (seg.Length() < min_length) {
        continue;
      }
      end = 0;
      begin = seg.begin;
      bic.TurnDetectThreshLess(seg, points);

      for (int j = 0; j < points.size(); ++j) {
        end = points[j];
        if (end - begin < min_seg_length) {
          continue;
        }
        SegCluster seg_cluster;
        seg_cluster.Add(Seg(begin, end));
        spk_cluster.Add(seg_cluster);
        begin = end + 1;
      }

      points.clear();
      if (begin != seg.end) {
        SegCluster seg_cluster;
        seg_cluster.Add(Seg(begin, seg.end));
        spk_cluster.Add(seg_cluster);
      }
    }
    return ALS_OK;
  }

  int LengthCheck(const SegCluster &cluster) const {
    unsigned int len = cluster.Length();
    //10s
    const int min_length = 1000;
    if (len < min_length) {
      idec::IDEC_ERROR << "speech too short. ";
      return ALSERR_SPK_DIAR_SPEECH_TOO_SHORT;
    }
    return ALS_OK;
  }

  int SpkDiar(const std::string &wave_file,
              const std::string &out_result) {
    int ret = ALS_OK;
    std::vector<char> wave;
    wave.reserve(10240);
    string name = "xxxxx";
    ret = serial_->GetFileName(wave_file, name);
    if (ret != ALS_OK) {
      name = "xxxxx";
    }

    wave_path_ = wave_file;
    ret = serial_->ReadWave(wave_file, wave);
    if (ret != ALS_OK) {
      return ret;
    }

    SpeakerCluster spk_cluster;
    ret = SpkDiarImpl(wave, name, spk_cluster);
    if (ret != ALS_OK) {
      return ret;
    }

    if (spk_cluster.Size() < 2) {
      idec::IDEC_ERROR << " spk_cluster is empty or just one speaker. ";
      return ALSERR_UNHANDLED_EXCEPTION;
    }

    if (verbose_mode_) {
      idec::IDEC_INFO << "[" + name + ".wav] Save Result...";
    }

    ret = serial_->SaveResult(wave_file, out_result, spk_cluster);
    if (ret != ALS_OK) {
      return ret;
    }
    return ALS_OK;
  }

  int SpkDiarImpl(vector<char> &wave,
                  const string &file_name, SpeakerCluster &spk_cluster) {
    int ret = ALS_OK;
    if (wave.size() < 10000) {
      idec::IDEC_ERROR << "[" + file_name + ".wav]" << "Invalid wave length. ";
      return ALSERR_SPK_DIAR_SPEECH_TOO_SHORT;
    }

    if (verbose_mode_) {
      idec::IDEC_INFO << "[" + file_name + ".wav]" << "VAD is running...";
    }

    SegCluster seg_cluster;
    ret = vad_->VadProcess(wave, seg_cluster);
    if (ret != ALS_OK) {
      return ret;
    }

    ret = LengthCheck(seg_cluster);
    if (ret != ALS_OK) {
      return ret;
    }

    if (verbose_mode_) {
      idec::IDEC_INFO << "[" + file_name + ".wav]" <<
                      "Acoustic feature of MFCC is extracting...";
    }

    // generate mfcc_ feature
    ret = mfcc_->FE(&wave[0], wave.size());
    if (use_ivector_) {
      mfcc_->FENew(&wave[0], wave.size());
    }

    if (ret != ALS_OK) {
      return ret;
    }

    if (verbose_mode_) {
      idec::IDEC_INFO << "[" + file_name + ".wav]" <<
                      "BIC change point detecting...";
    }

    ahc_->Init(wave_path_);

    if (!use_bic_) {
      const int win_size = 100;
      const int shift_size = 99;
      AverageSegmentation(seg_cluster, spk_cluster, win_size, shift_size);
      ahc_->Clustering(spk_cluster);
    } else {
      bic_->Init(mfcc_);
      TurnDetect(*bic_, seg_cluster, spk_cluster);
      ahc_->Process(spk_cluster);
    }

    if (verbose_mode_) {
      idec::IDEC_INFO << "[" + file_name + ".wav]" << "HMM tunning...";
    }

    ret = spkdiar_tunning_->ReSegmentProcess(*mfcc_, *ahc_, spk_cluster,
          seg_cluster);
    if (ret != ALS_OK) {
      return ret;
    }

    mfcc_->Clear();
    return ALS_OK;
  }

  int SpkDiar(vector<char> &wave,
              AlsSpkdiarResult *spkdiar_result) {
    int ret = ALS_OK;
    try {
      if (spkdiar_result != NULL) {
        spkdiar_result->fragment_num = 0;
        spkdiar_result->speech_fragments = NULL;
      }

      string file_name = "xxxxx";
      SpeakerCluster spk_cluster;
      ret = SpkDiarImpl(wave, file_name, spk_cluster);
      if (ret != ALS_OK) {
        return ALSERR_UNHANDLED_EXCEPTION;
      }

      if (spk_cluster.Size() < 2) {
        idec::IDEC_ERROR << "spk_cluster is empty or just one speaker.";
        return ALSERR_UNHANDLED_EXCEPTION;
      }

      if (verbose_mode_) {
        idec::IDEC_INFO << "[" + file_name + ".wav]" << "Save Result...";
      }

      ret = serial_->SaveResult(spk_cluster, spkdiar_result);
      if (ret != ALS_OK) {
        return ALSERR_UNHANDLED_EXCEPTION;
      }
    } catch (...) {
      idec::IDEC_INFO << "Internal error.";
    }

    return ALS_OK;
  }

 public:
  Mfcc *mfcc_;
  Vad *vad_;
  Bic *bic_;
  Ahc *ahc_;
  string wave_path_;
  Serialize *serial_;
  SpkDiarTunning *spkdiar_tunning_;

  bool use_bic_;
  bool use_ivector_;
  bool verbose_mode_;
};

}

#endif
