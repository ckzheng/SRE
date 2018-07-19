#ifndef _VAD_H_
#define _VAD_H_

#include "als_vad.h"
#include "seg.h"

namespace alspkdiar {

class Vad {
 public:
  Vad(AlsVad *vad_imp) :vad_imp_(NULL) { vad_imp_ = vad_imp; }
  ~Vad() { AlsVad::Destroy(vad_imp_); }
  int VadProcess(const vector<char> &waves, SegCluster &seg_cluster) {
    vad_imp_->SetVoiceStartCallback(NULL, NULL);
    vad_imp_->SetVoiceDetectedCallback(NULL, NULL);
    vad_imp_->SetVoiceEndCallback(NULL, NULL);
    vad_imp_->BeginUtterance();

    const size_t block_size = 400; // 25ms
    AlsVadResult *result = NULL;
    alspkdiar::Seg *seg = NULL;
    const int wave_len = waves.size();
    unsigned int loop = waves.size() / block_size;
    loop = (loop == 0) ? 1 : loop;
    bool start_flag = false, end_flag = false, flag = false;
    flag = (waves.size() % block_size != 0) ? true : false;
    unsigned int base = 0, begin = 0, end = wave_len;
    for (int i = 0; i < loop; ++i) {
      if (i == loop - 1) {
        unsigned int len = flag ? (wave_len % block_size + block_size) :
                           block_size;
        vad_imp_->SetData2((short *)(&waves[0] + base), len, true);
      } else {
        vad_imp_->SetData2((short *)(&waves[0] + base), block_size, false);
      }

      base += block_size;
      result = vad_imp_->DoDetect2();
      if (result != NULL) {
        for (int i = 0; i < result->num_segments; i++) {
          AlsVadSpeechBuf &buf(result->speech_segments[i]);
          if (buf.contain_seg_start_point) {
            begin = buf.start_ms / 10;
            start_flag = true;
          }

          if (buf.contain_seg_end_point) {
            // convert to frames
            end = buf.end_ms / 10;
            end_flag = true;
          }

          if (start_flag && end_flag) {
            seg_cluster.Add(Seg(begin, end));
            start_flag = false;
            end_flag = false;
          }
        }
        AlsVadResult_Release(&result);
        result = NULL;
      }
    }
    vad_imp_->EndUtterance();
    return 0;
  }

 private:
  AlsVad *vad_imp_;
};

}

#endif
