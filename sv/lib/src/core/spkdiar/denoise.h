#ifndef _DENOISE_H_
#define _DENOISE_H_
#include <vector>
#include "new_matrix.h"
#include "new_vector.h"
#include "diag_gmm.h"
#include "full_gmm.h"
#include "full_gaussian.h"
#include "fe/frontend.h"
#include "fe/frontend_pipeline.h"

using namespace alspkdiar;
using std::vector;

class Denoise {
 public:
  Denoise(const char *sys_dir, const char *cfg_file) {
    front_end_ = new idec::FrontendPipeline();
    front_end_->Init(cfg_file, sys_dir);
  }

  Denoise() {
    front_end_ = NULL;
  }

  ~Denoise() {
    idec::IDEC_DELETE(front_end_);
  }

  void FE(char *wave, unsigned int len) {
    // do vad, get the frame_by_frame result
    front_end_->BeginUtterance();
    const size_t wavblocksize = 400;
    unsigned int loop = len / wavblocksize;
    if ((len % wavblocksize) != 0) {
      loop += 1;
    }

    unsigned int base_ptr = 0, begin_pos = 0, frag_len = 0;
    for (int i = 0; i < loop; ++i) {
      frag_len = (i == loop - 1) ? (len % wavblocksize) : wavblocksize;
      if (frag_len == 0) {
        frag_len = wavblocksize;
      }

      front_end_->PushAudio(wave + base_ptr, frag_len, idec::FE_8K_16BIT_PCM);

      base_ptr += frag_len;
    }
    front_end_->EndUtterance();
  }

  unsigned int GetFrames() const {
    return front_end_->NumFrames();
  }

  unsigned int GetDims() const {
    return front_end_->GetFeatureDim();
  }

  void XnnMatrix2DoubleMatrix(const idec::xnnFloatRuntimeMatrix &xnnfeat,
                              DoubleMatrix &feats, int idx, int begin_idx, int end_idx) const {
    idec::IDEC_ASSERT(end_idx - begin_idx > 0);
    idec::IDEC_ASSERT(feats.Rows() >= idx + end_idx - begin_idx);
    //feats.Resize(end_idx - begin_idx, xnnfeat.NumRows());
    float *pTmp = NULL;
    for (int i = begin_idx, k = idx; i < end_idx; ++i, ++k) {
      pTmp = xnnfeat.Col(i);
      for (int j = 0; j < xnnfeat.NumRows(); ++j) {
        feats(k, j) = pTmp[j];
      }
    }
  }

  int MergeFeature(const SegCluster &cluster, DoubleMatrix &feats) const {
    int num_frm = 0;
    int total_len = cluster.Length();
    feats.Resize(total_len, front_end_->GetFeatureDim());
    for (int i = 0; i < cluster.Size(); ++i) {
      const Seg &segment = cluster.GetSeg(i);
      idec::xnnFloatRuntimeMatrix xnnfeat;
      front_end_->PeekPartFrames(segment.begin, segment.end, xnnfeat);
      XnnMatrix2DoubleMatrix(xnnfeat, feats, num_frm, 0, xnnfeat.NumCols());
      num_frm += xnnfeat.NumCols();
    }
    idec::IDEC_ASSERT(total_len == num_frm);
    return 0;
  }

  int DiagGmmgSelect(const DoubleMatrix &feats, const DiagGmm &gmm,
                     int num_gselect,
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

    return 0;
  }

  int LogLikelihood(const DoubleMatrix &feats, const FullGmm &fgmm,
                    DoubleVector &llk) const {
    int num_frames = feats.Rows();
    llk.Resize(num_frames);

    int dims = feats.Cols();

    DiagGmm dgmm = DiagGmm();
    dgmm.CopyFromFullGmm(fgmm);

    const int num_gselect = 10;
    vector<vector<int> > gselect;
    DiagGmmgSelect(feats, dgmm, num_gselect, gselect);

    double this_tot_loglike = 0;
    DoubleVector frame, loglikes;
    for (int t = 0; t < num_frames; t++) {
      frame = feats.Rowv(t);
      fgmm.LogLikelihoodsPreselect(frame, gselect[t], loglikes);
      this_tot_loglike = loglikes.LogSumExp();
      llk(t) = this_tot_loglike;
    }

    double mean = (double) llk.Sum() / llk.Size();
    for (int i = 0; i < llk.Size(); ++i) {
      llk(i) -= mean;
    }

    //ofstream ofs("C:\\xxxx.txt");
    //for (int i = 0; i < llk.Size(); ++i) {
    //  ofs << llk(i) << endl;
    //}
    //ofs.close();
    return 0;
  }

  void DetectMusic(const DoubleVector &llk, vector<int> &tag) const {
    double threshhold_score = -140;
    if (llk.Size() != tag.size()) {
      tag.resize(llk.Size());
    }

    for (int ipos = 0; ipos < llk.Size(); ++ipos) {
      tag[ipos] = 0;
      if (llk(ipos) > threshhold_score) {
        tag[ipos] = 1;
      }
    }

    const int win_size = 50;
    int acc, flag_begin = 0;
    for (int ipos = 0; ipos < tag.size(); ++ipos) {
      if (tag[ipos]) {
        int end_point = ipos + win_size;
        if (end_point > tag.size()) {
          end_point = tag.size();
        }
        acc = 0;
        for (int d = ipos; d < end_point; ++d) {
          acc += tag[d];
        }

        if (acc > win_size * 0.75) {
          while (ipos < end_point) {
            tag[ipos++] = 1;
          }

          while ((ipos < tag.size()) && tag[ipos]) {
            ipos++;
          }

          while (ipos + 5 < tag.size()) {
            if (tag[ipos + 1] || tag[ipos + 2] || tag[ipos + 3] || tag[ipos + 4]
                || tag[ipos + 5]) {
              tag[ipos++] = 1;
              continue;
            }
            break;
          }
        } else {
          tag[ipos] = 0;
        }
      }
    }
  }

  void DetectDtmfNoise(const DoubleVector &llk, vector<int> &tag) const {
    double threshhold_score = -3;
    if (llk.Size() != tag.size()) {
      tag.resize(llk.Size());
    }

    for (int ipos = 0; ipos < llk.Size(); ++ipos) {
      tag[ipos] = 0;
      if (llk(ipos) > threshhold_score) {
        tag[ipos] = 1;
      }
    }

    const int win_size = 30;
    int  acc, flag_begin = 0;
    for (int ipos = 0; ipos < tag.size(); ++ipos) {
      if (tag[ipos]) {
        acc = 0;
        int end_point = ipos + win_size;
        if (end_point > tag.size()) {
          end_point = tag.size();
        }

        for (int d = ipos; d < end_point; ++d) {
          acc += tag[d];
        }

        if (acc > win_size * 0.75) {
          while (ipos < end_point) {
            tag[ipos++] = 1;
          }

          while (tag[ipos] && (ipos < tag.size())) {
            ipos++;
          }

          while (ipos + 5 < tag.size()) {
            if (tag[ipos + 1] || tag[ipos + 2] || tag[ipos + 3] || tag[ipos + 4]
                || tag[ipos + 5]) {
              tag[ipos++] = 1;
              continue;
            }
            break;
          }
        } else {
          tag[ipos] = 0;
        }
      }
    }
  }

  void GetResult(const vector<int> &tag, SegCluster &cluster) {
    int begin = 0, end = 0;
    for (int i = 0; i < tag.size(); ++i) {
      if (!begin && tag[i]) {
        begin = i;
      }

      if (begin && !tag[i]) {
        end = i;
      }

      if (begin && end) {
        const int min_len_dtmf = 45;
        const int min_len_music = 100;
        if (end - begin >= min_len_music) {
          cluster.Add(Seg(begin, end));
        }
        begin = 0;
        end = 0;
      }
    }
  }

  //void Process(SegCluster &cluster) {
  // //string dtmf_path = "D:\\sv\\data\\spkdiar\\mdl\\final.ubm";
  // string music_path = "D:\\sv\\data\\spkdiar\\mdl\\final-music.ubm.32";
  // ResourceLoader res_loader;
  // res_loader.loadUbmResource(music_path);
  // FullGmm full_gmm(res_loader.GetUbmResource());
  // cluster.Add(Seg(0, GetFrames()));
  // DoubleMatrix feats;
  // MergeFeature(cluster, feats);
  // cluster.Clear();
  // DoubleVector llk;
  // LogLikelihood(feats, full_gmm, llk);
  // vector<int> tag;
  // DetectDtmfNoise(llk, tag);
  // GetResult(tag, cluster);
  //}

  void Process(SegCluster &cluster) {
    //string dtmf_path = "D:\\sv\\data\\spkdiar\\mdl\\final.ubm";
    string speech_path = "D:\\sv\\data\\spkdiar\\mdl\\final-speech.ubm.32";
    ResourceLoader res_loader;
    res_loader.loadUbmResource(speech_path);
    FullGmm full_gmm_speech(res_loader.GetUbmResource());
    cluster.Add(Seg (0, GetFrames()));
    DoubleMatrix feats;
    MergeFeature(cluster, feats);
    cluster.Clear();
    DoubleVector llk_speech;
    LogLikelihood(feats, full_gmm_speech, llk_speech);

    string music_path = "D:\\sv\\data\\spkdiar\\mdl\\final-music.ubm.32";
    res_loader.loadUbmResource(music_path);
    FullGmm full_gmm_music(res_loader.GetUbmResource());
    DoubleVector llk_music;
    LogLikelihood(feats, full_gmm_music, llk_music);

    DoubleVector llk(llk_speech.Size());
    for (int i = 0; i < llk_speech.Size(); ++i) {
      float lratio = llk_music(i) - llk_speech(i);
      llk(i) = 1 / (1 + exp(-lratio));
    }

    //ifstream ifs("C:\\xxxx.txt");
    //float val;
    //vector<float> vec;
    //while (!ifs.eof()) {
    //	ifs >> val;
    //	vec.push_back(val);
    //}
    //
    //llk.Resize(vec.size() - 1);
    //for (int i = 0; i < vec.size() - 1; ++i) {
    //	llk(i) = vec[i];
    //}

    vector<int> tag;
    //DetectDtmfNoise(llk, tag);
    DetectMusic(llk, tag);
    GetResult(tag, cluster);
  }

 private:
  idec::FrontendPipeline *front_end_;
};

#endif