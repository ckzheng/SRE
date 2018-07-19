#ifndef _SPKDIAR_SERIALIZE_H_
#define _SPKDIAR_SERIALIZE_H_

#include <set>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include "speaker_cluster.h"
#include "als_error.h"
#include "wav.h"
#include "seg.h"
#include "spkdiar_result.h"

using std::vector;
using std::string;
using std::ofstream;

namespace alspkdiar {

class Serialize {
 public:
  Serialize(const string &str_input_type = "FE_8K_16BIT_PCM") {
    if (str_input_type == "FE_8K_16BIT_PCM") {
      sample_rate_ = 8000;
      bytes_per_sample_ = 2;
    } else if (str_input_type == "FE_16K_16BIT_PCM") {
      sample_rate_ = 16000;
      bytes_per_sample_ = 2;
    } else {
      idec::IDEC_ERROR << "unknown input type " << str_input_type;
    }
  }

  static int DestoryResult(AlsSpkdiarResult *out_result) {
    if (out_result != NULL) {
      SpeechFragment *speech_fragments = out_result->speech_fragments;
      if (NULL != speech_fragments) {
        idec::IDEC_DELETE_ARRAY(speech_fragments);
      }
      idec::IDEC_DELETE(out_result);
    }
    return ALS_OK;
  }

  int SaveResult(SpeakerCluster &spk_cluster,
                 AlsSpkdiarResult *spkdiar_result) const {
    unsigned int begin, end;
    std::vector<SpeechFragment> labels;
    for (int i = 0; i < spk_cluster.Size(); ++i) {
      const SegCluster &cluster = spk_cluster.Get(i);
      for (int j = 0; j < cluster.Size(); ++j) {
        const Seg &segment = cluster.GetSeg(j);
        begin = segment.begin * 10;
        end = segment.end * 10;
        SpeechFragment speech_fragment = { begin, end, i };
        labels.push_back(speech_fragment);
      }
    }

    SpeechFragment *speech_fragments = new SpeechFragment[labels.size()];
    for (int i = 0; i < labels.size(); ++i) {
      speech_fragments[i].begin_time = labels[i].begin_time;
      speech_fragments[i].end_time = labels[i].end_time;
      speech_fragments[i].speaker_id = labels[i].speaker_id;
    }
    spkdiar_result->speech_fragments = speech_fragments;
    spkdiar_result->fragment_num = labels.size();
    return ALS_OK;
  }

  int WriteHeader(char *chn_wav, unsigned int wave_len,
                  Wave_header &header) const {
    header.data->cb_size = ((wave_len + 1) / 2) * 2;
    header.riff->cb_size = 4 + 4 + header.fmt->cb_size + 4 + 4 +
                           header.data->cb_size + 4;
    unsigned int bytes = 0;
    // Write RIFF
    char *chunk = (char *)header.riff;
    memcpy(chn_wav + bytes, chunk, sizeof(BaseChunk));
    bytes += sizeof(BaseChunk);

    // Write WAVE fourcc
    memcpy(chn_wav + bytes, (char *)&(header.wave_fcc), 4);
    bytes += 4;

    // Write fmt
    chunk = (char *)header.fmt;
    memcpy(chn_wav + bytes, chunk, sizeof(BaseChunk));
    bytes += sizeof(BaseChunk);

    // Write fmt_data
    chunk = (char *)header.fmt_data;
    memcpy(chn_wav + bytes, chunk, header.fmt->cb_size);
    bytes += header.fmt->cb_size;

    // Write data
    chunk = (char *)header.data;
    memcpy(chn_wav + bytes, chunk, sizeof(BaseChunk));
    bytes += sizeof(BaseChunk);
    return ALS_OK;
  }

  int SaveLabel(std::string file_path, std::vector<Seg> &spk_cluster) const {
    std::ofstream fout(file_path.c_str());
    if (!fout) {
      fout.close();
      idec::IDEC_ERROR << "Open file " << file_path.c_str();
      return ALSERR_UNHANDLED_EXCEPTION;
    }

    for (unsigned int i = 0; i < spk_cluster.size(); ++i) {
      const Seg &segment = spk_cluster[i];
      fout << segment.begin << " , " << segment.end << " , " <<
           segment.label << std::endl;
    }

    fout.close();
    return ALS_OK;
  }

  int SaveWaveAndLabel(const char *raw_wav, const string wav_path,
                       const string label_path, SegCluster &cluster) const {
    std::vector<Seg> segs;
    for (int i = 0; i < cluster.Size(); ++i) {
      segs.push_back(cluster.GetSeg(i));
    }

    Seg segment;
    for (int i = 0; i < segs.size(); i++) {
      for (int j = i + 1; j < segs.size(); j++) {
        if (segs[i].begin > segs[j].begin) {
          segment = segs[i];
          segs[i] = segs[j];
          segs[j] = segment;
        }
      }
    }

    // copy wave from raw wav
    std::vector<char> wav;
    for (int i = 0; i < segs.size(); ++i) {
      segment = segs[i];
      //unsigned int begin = 44 + segment.begin * 160;
      //unsigned int end = 44 + segment.end * 160;
      unsigned int begin = 44 + segment.begin * (sample_rate_ * bytes_per_sample_ * 0.01);
      unsigned int end = 44 + segment.end * (sample_rate_ * bytes_per_sample_ * 0.01);
      for (unsigned int pos = begin; pos < end; ++pos) {
        wav.push_back(raw_wav[pos]);
      }
    }

    //Wave_header header(1, 8000, 16);
    Wave_header header(1, sample_rate_, bytes_per_sample_ * 8);
    WaveFile wave_file;
    wave_file.Write(wav_path, header, &wav[0], wav.size());
    SaveLabel(label_path, segs);
    return ALS_OK;
  }

  int CopyFromRawWave(const char *raw_wav, const SegCluster &cluster, vector<char> &wav) const {
    vector<Seg> segs;
    for (int i = 0; i < cluster.Size(); ++i) {
      segs.push_back(cluster.GetSeg(i));
    }

    Seg segment;
    for (int i = 0; i < segs.size(); i++) {
      for (int j = i + 1; j < segs.size(); j++) {
        if (segs[i].begin > segs[j].begin) {
          segment = segs[i];
          segs[i] = segs[j];
          segs[j] = segment;
        }
      }
    }

    unsigned int begin, end, pos;
    for (int i = 0; i < segs.size(); ++i) {
      const Seg &segment = segs[i];
      //begin = 44 + segment.begin * 160;
      //end = 44 + segment.end * 160;
      begin = 44 + segment.begin * (sample_rate_ * bytes_per_sample_ * 0.01);
      end = 44 + segment.end * (sample_rate_ * bytes_per_sample_ * 0.01);
      for (pos = begin; pos < end; ++pos) {
        wav.push_back(raw_wav[pos]);
      }
    }
    return ALS_OK;
  }

  int CopyFromRawWave(const char *raw_wav, SpeakerCluster &spk_cluster, std::vector<char> &wav) const {
    vector<Seg> segs;
    const SegCluster &cluster0 = spk_cluster.Get(0);
    for (int i = 0; i < cluster0.Size(); ++i) {
      segs.push_back(cluster0.GetSeg(i));
    }

    const SegCluster &cluster1 = spk_cluster.Get(1);
    for (int i = 0; i < cluster1.Size(); ++i) {
      segs.push_back(cluster1.GetSeg(i));
    }

    for (int i = 0; i < segs.size(); i++) {
      for (int j = i + 1; j < segs.size(); j++) {
        if (segs[i].begin > segs[j].begin) {
          Seg temp = segs[i];
          segs[i] = segs[j];
          segs[j] = temp;
        }
      }
    }

    unsigned int begin, end, pos;
    const int label = segs[0].label;
    for (int i = 0; i < segs.size(); ++i) {
      const Seg &segment = segs[i];
      //begin = 44 + segment.begin * 160;
      //end = 44 + segment.end * 160;
      begin = 44 + segment.begin *( sample_rate_ * bytes_per_sample_ * 0.01);
      end = 44 + segment.end * (sample_rate_ * bytes_per_sample_ * 0.01);
      if (segment.label == label) {
        for (pos = begin; pos < end; pos += 2) {
          wav.push_back(raw_wav[pos]);
          wav.push_back(raw_wav[pos + 1]);
          wav.push_back(0);
          wav.push_back(0);
        }
      } else {
        for (pos = begin; pos < end; pos += 2) {
          wav.push_back(0);
          wav.push_back(0);
          wav.push_back(raw_wav[pos]);
          wav.push_back(raw_wav[pos + 1]);
        }
      }
    }
    return ALS_OK;
  }

  int GetFileName(const string &wav_path, string &name) const {
    int first, last, len;
    last = wav_path.find_last_of('.');
    if (-1 == last) {
      return ALSERR_UNHANDLED_EXCEPTION;
    }

    first = last - 1;
    while (first > 0) {
      if (wav_path[first] == '\\' || wav_path[first] == '/') {
        break;
      }
      --first;
    }

    if (first != 0) {
      first += 1;
    }

    len = last - first;
    name = wav_path.substr(first, len);
    return ALS_OK;
  }

  static int ReadWave(const string &fname, std::vector<char> &wave) {
    ifstream ifs(fname.c_str(), ios::binary);
    if (!ifs) {
      ifs.close();
      idec::IDEC_ERROR << "Cannot WAVE file: " << fname;
      return ALSERR_UNHANDLED_EXCEPTION;
    }

    ifs.seekg(0, ios::end);
    int len = ifs.tellg();
    len -= 44;
    wave.resize(len);
    ifs.seekg(44, ios::beg);
    ifs.read(&wave[0], len);
    ifs.close();
    return ALS_OK;
  }

  int SaveResult(const string &in_wav_path, const string &out_path, SpeakerCluster &spk_cluster) const {
    if (spk_cluster.Size() < 2) {
      idec::IDEC_ERROR << "spk_cluster is empty or just one speaker ";
      return ALSERR_UNHANDLED_EXCEPTION;
    }

    string name, out_wav_path, labelPath;
    GetFileName(in_wav_path, name);
    vector<char> raw_wave;
    ReadWave(in_wav_path, raw_wave);

    char *pWav = &raw_wave[0];
    unsigned int len = raw_wave.size();

    out_wav_path = out_path + name + "_1.wav";
    labelPath = out_path + name + "_1.lbl";
    SaveWaveAndLabel(pWav, out_wav_path, labelPath, spk_cluster.Get(0));

    out_wav_path = out_path + name + "_2.wav";
    labelPath = out_path + name + "_2.lbl";
    SaveWaveAndLabel(pWav, out_wav_path, labelPath, spk_cluster.Get(1));

    //Wave_header header(2, 8000, 16);
    Wave_header header(2, sample_rate_, bytes_per_sample_ * 8);
    unsigned int new_len = (len - 44) * 2;
    vector<char> wav;
    wav.reserve(new_len);
    CopyFromRawWave(pWav, spk_cluster, wav);
    WaveFile wave_file = WaveFile();
    wave_file.Write(out_path + name + "-res.wav", header, &wav[0], wav.size());
    wav.empty();
    return ALS_OK;
  }
 private:
  int sample_rate_;
  int bytes_per_sample_;
};
}
#endif

