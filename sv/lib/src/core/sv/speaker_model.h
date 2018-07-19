#ifndef _SPEAKER_MODEL_H_
#define _SPEAKER_MODEL_H_
#include <string>
#include <vector>
#include "new_matrix.h"
#include "new_vector.h"

enum ElementLength {
  HEADER_SIZE = 1280,
  SPK_ID_LEN = 1024,
  FEAT_TYPE_LEN = 15,
  CREATE_TIME_LEN = 18,
  UPDATE_TIME_LEN = 18,
  MD5_LEN = 16
};

enum UpdateIvectorMethod {
  UPDATE_IVECTOR_BY_STATS = 1,
  UPDATE_IVECTOR_BY_MEAN = 2,
  UPDATE_IVECTOR_BY_WEIGHTED_MEAN = 3
};

class SpeakerModel {
 public:
  SpeakerModel();
  void Rand();
  void Searialize(string &spk_mdl);
  void Deserialize(const string &spk_mdl);
  string GetCurrentTime() const;
  string GetMd5(string bodyRequest) const;
  //void UpdateStats(const DoubleVector &gamma, const DoubleMatrix &x);
  //void GetStats(DoubleVector &gamma, DoubleMatrix &x) const;
  void GetIvector(DoubleVector &ivector) const;
  /* void UpdateModel(const string &spk_id, const string &feat_type, int utt_num,
    const DoubleVector &gamma, const DoubleMatrix &x, const DoubleVector &iv, UpdateIvectorMethod update_ivector_method);*/
  void UpdateModel(const string &spk_id, const string &feat_type, int utt_num,
                   int frame_num, int feat_dim, int mix_num, const DoubleVector &iv);
  bool IsValid(const string &spk_mdl) const;
  int GetUttNum() const {return utt_num_;}
  int GetFramesNum() const { return frame_num_; }
  unsigned int FeatureDim() const { return feat_dim_; }
  unsigned int MixtureNum() const { return mix_num_; }
  bool IsEmpty() const {return (feat_dim_ == 0);}
  string GetSpeakerId() const {return string(spk_id_, SPK_ID_LEN); }
  string GetFeatureType() const {return string(feat_type_, FEAT_TYPE_LEN);}
 public:
  char utt_num_;
  short mix_num_;
  int frame_num_;
  char feat_dim_;
  short ivector_dim_;
  char spk_id_[SPK_ID_LEN];
  char feat_type_[FEAT_TYPE_LEN];
  char create_time_[CREATE_TIME_LEN];
  char update_time_[UPDATE_TIME_LEN];
  char md5_[MD5_LEN];
  //std::vector<float> stat_n_;
  //std::vector<float>stat_f_;
  std::vector<float>ivector_;
};

#endif // !_SPEAKER_MODEL_H_
