#include "spkdiar_impl.h"
#include <fstream>
#include "wav.h"
#include "vad.h"
#include "spkdiar_serialize.h"

#define THRESHOLD_BIC 0.0

namespace alspkdiar {

using namespace std;

SpeakerDiarization::SpeakerDiarization(AlsSpkdiarMdlHandler *handler) {
  mfcc_ = new Mfcc(handler->sys_dir_, handler->cfg_file_);
  vad_ = new Vad(AlsVad::CreateFromModel(handler->vad_handle_));  
  bic_ = new Bic(handler->opt_);
  ahc_ = new Ahc(bic_); 
  serial_ = new Serialize();
  spkdiar_tunning_ = new SpkDiarTunning(handler->ubm_, handler->gmm_update_opt_,
                                        handler->opt_);
  verbose_mode_ = handler->opt_.verbose_mode;
}

SpeakerDiarization::~SpeakerDiarization() {
  idec::IDEC_DELETE(vad_);
  idec::IDEC_DELETE(mfcc_);
  idec::IDEC_DELETE(bic_);
  idec::IDEC_DELETE(ahc_);
  idec::IDEC_DELETE(serial_);
  idec::IDEC_DELETE(spkdiar_tunning_);
}

ALS_RETCODE SpeakerDiarization::BeginSpkDiar() {
  return ALS_OK;
}

ALS_RETCODE SpeakerDiarization::EndSpkDiar() {
  return ALS_OK;
}

ALS_RETCODE SpeakerDiarization::TurnDetect(Bic&bic,
    SpeakerCluster&spk_cluster, const SegCluster &segs) {
  int begin = 0, end = 0;
  Seg *seg = NULL;
  Seg *segment = NULL;
  std::vector<int> points = vector <int>();
  points.reserve(128);
  const int min_length = 100;
  const int min_seg_length = 50;
  for (unsigned int i = 0; i < segs.Size(); ++i) {
    seg = segs.GetSeg(i);
    if (seg->Length() < min_length) {
      continue;
    }
    end = 0;
    begin = seg->GetBegin();
    // bic->TurnDetect(seg, points);
    bic->TurnDetectThreshLess(seg, points);

    for (int j = 0; j < points.size(); ++j) {
      end = points[j];
      if (end - begin < min_seg_length) {
        continue;
      }

      segment = new Seg(begin, end, "");
      spk_cluster->AddSpeaker(segment);
      begin = end + 1;
    }

    points.clear();
    if (begin != seg->GetEnd()) {
      segment = new Seg(begin, seg->GetEnd(), "");
      spk_cluster->AddSpeaker(segment);
    }
  }
  return ALS_OK;
}

ALS_RETCODE SpeakerDiarization::LengthCheck(SegCluster
    &cluster) {
  unsigned int len = cluster.Length();
  //10s
  const int min_length = 1000;
  if (len < min_length) {
    cluster.Clear();
    idec::IDEC_ERROR << "speech too short. ";
    return ALSERR_SPK_DIAR_SPEECH_TOO_SHORT;
  }
  return ALS_OK;
}

ALS_RETCODE SpeakerDiarization::SpkDiar(const std::string &wave_file,
                                        const std::string &out_result) {
  // double time_start = idec::TimeUtils::GetTimeMilliseconds();
  ALS_RETCODE ret = ALS_OK;
  std::vector<char> wave = std::vector<char>();
  wave.reserve(10240);
  string name = "xxxxx";
  ret = serial_->GetFileName(wave_file, name);
  if (ret != ALS_OK) {
    name = "xxxxx";
  }

  ret = serial_->ReadWave(wave_file, wave);
  if (ret != ALS_OK) {
    return ret;
  }

  SpeakerCluster *spk_cluster = new SpeakerCluster();
  ret = SpkDiarImpl(&wave[0], wave.size(), name, spk_cluster);
  if (ret != ALS_OK) {
    return ret;
  }

  if ((spk_cluster == NULL) || (spk_cluster->Size() < 2)) {
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

  spk_cluster->Clear();
  idec::IDEC_DELETE(spk_cluster);
  return ALS_OK;
}

ALS_RETCODE SpeakerDiarization::SpkDiarImpl(char *wave,
    unsigned int wave_len, const string &file_name, SpeakerCluster *&spk_cluster) {
  ALS_RETCODE ret = ALS_OK;
  if ((wave == NULL) || (wave_len < 10000)) {
    idec::IDEC_ERROR << "[" + file_name + ".wav]" << "Invalid wave length. " ;
    return ALSERR_SPK_DIAR_SPEECH_TOO_SHORT;
  }

  if (verbose_mode_) {
    idec::IDEC_INFO << "[" + file_name + ".wav]" <<
                    "Acoustic feature of MFCC is extracting...";
  }

  SegCluster seg_cluster = SegCluster();
  if (verbose_mode_) {
    idec::IDEC_INFO << "[" + file_name + ".wav]" << "VAD is running...";
  }

  ret = vad_->VadProcess(wave, wave_len, seg_cluster);
  if (ret != ALS_OK) {
    return ret;
  }

  ret = LengthCheck(seg_cluster);
  if (ret != ALS_OK) {
    return ret;
  }

  // generate mfcc_ feature
  ret = mfcc_->FE(wave, wave_len);
  if (ret != ALS_OK) {
	  return ret;
  }

  // Bic* bic = new Bic(mfcc_, ubm_, gmm_update_opt_);
  if (verbose_mode_) {
    idec::IDEC_INFO << "[" + file_name + ".wav]" <<
                    "BIC change point detecting...";
  }

  bic_->Init(mfcc_);

  // use single gaussian to find speaker turn points
  ret = TurnDetect(bic_, spk_cluster, seg_cluster);
  if (ret != ALS_OK) {
    seg_cluster.Clear();
    return ret;
  }

  if (verbose_mode_) {
    idec::IDEC_INFO << "[" + file_name + ".wav]" << "AHC clustering...";
  }

  // bottom up clustering
  ret = ahc_->Clustering(spk_cluster);
  if (ret != ALS_OK) {
    seg_cluster.Clear();
    return ret;
  }

  if (verbose_mode_) {
    idec::IDEC_INFO << "[" + file_name + ".wav]" << "HMM tunning...";
  }

  ret = spkdiar_tunning_->ReSegmentProcess(mfcc_, spk_cluster, seg_cluster);
  if (ret != ALS_OK) {
    seg_cluster.Clear();
    return ret;
  }

  seg_cluster.Clear();
  mfcc_->Clear();
  return ALS_OK;
}

ALS_RETCODE SpeakerDiarization::SpkDiar(char *wave, unsigned int wave_len,
                                        AlsSpkdiarResult *spkdiar_result) {
  ALS_RETCODE ret = ALS_OK;
  SpeakerCluster *spk_cluster = new SpeakerCluster();
  try {
    if (spkdiar_result != NULL) {
      spkdiar_result->fragment_num = 0;
      spkdiar_result->speech_fragments = NULL;
    }
    
    string file_name = "xxxxx";
    ret = SpkDiarImpl(wave, wave_len, file_name, spk_cluster);
    if (ret != ALS_OK) {
      return ALSERR_UNHANDLED_EXCEPTION;
    }

    if ((spk_cluster == NULL) || (spk_cluster->Size() < 2)) {
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

  spk_cluster->Clear();
  idec::IDEC_DELETE(spk_cluster);

  return ALS_OK;
}
}
