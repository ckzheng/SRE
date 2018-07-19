#include "speaker_model.h"
#include <ctime>
#include <vector>
#include <sstream>
#include "md5.h"
#include "base/idec_common.h"
#include "base/log_message.h"

using namespace std;

SpeakerModel::SpeakerModel() : utt_num_(0),mix_num_ (0),frame_num_ (0),
  feat_dim_ (0),ivector_dim_ (0) {
  memset(md5_, 0, sizeof(char)* MD5_LEN);
  memset(spk_id_, 0, sizeof(char)*SPK_ID_LEN);
  memset(feat_type_, 0, sizeof(char) * FEAT_TYPE_LEN);
  memset(create_time_, 0, sizeof(char)* CREATE_TIME_LEN);
  memset(update_time_, 0, sizeof(char) * UPDATE_TIME_LEN);
}

void SpeakerModel::Searialize(string &spk_mdl) {
  stringstream stream;
  stream.write(spk_id_, SPK_ID_LEN);
  stream.write(feat_type_, FEAT_TYPE_LEN);
  stream.write(&utt_num_, sizeof(char));
  stream.write((char *)&frame_num_, sizeof(int));
  stream.write((char *)&mix_num_, sizeof(short));
  stream.write(&feat_dim_, sizeof(char));
  stream.write((char *)&ivector_dim_, sizeof(short));
  stream.write(create_time_, CREATE_TIME_LEN);
  stream.write(update_time_, UPDATE_TIME_LEN);
  int actual_header_size = stream.str().size();
  if (actual_header_size != HEADER_SIZE) {
    idec::IDEC_ASSERT(actual_header_size < HEADER_SIZE);
    string zero(HEADER_SIZE - actual_header_size, 0);
    stream.write(zero.c_str(), zero.size());
  }
  //stream.write((char *)&stat_n_[0], sizeof(float)* mix_num_);
  //stream.write((char *)&stat_f_[0], sizeof(float)* mix_num_* feat_dim_);
  stream.write((char *)&ivector_[0], sizeof(float)* ivector_dim_);
  string content = stream.str();
  const string sMd5 = GetMd5(content);
  stream.write(sMd5.c_str(), sMd5.size());
  if (sMd5.size() != MD5_LEN) {
    idec::IDEC_ASSERT(MD5_LEN > sMd5.size());
    const string zero(MD5_LEN - sMd5.size(), 0);
    stream.write(zero.c_str(), zero.size());
  }
  spk_mdl = stream.str();
}

void SpeakerModel::Deserialize(const string &spk_mdl) {
  stringstream stream;
  stream << spk_mdl;
  stream.seekg(0);
  stream.read(spk_id_, SPK_ID_LEN);
  stream.read(feat_type_, FEAT_TYPE_LEN);
  stream.read(&utt_num_, sizeof(char));
  stream.read((char *)&frame_num_, sizeof(int));
  stream.read((char *)&mix_num_, sizeof(short));
  stream.read(&feat_dim_, sizeof(char));
  stream.read((char *)&ivector_dim_, sizeof(short));
  stream.read(create_time_, CREATE_TIME_LEN);
  stream.read(update_time_, UPDATE_TIME_LEN);
  stream.seekg(HEADER_SIZE);
  //if (stat_n_.empty()) {
  //  stat_n_.resize(mix_num_);
  //}

  //if (stat_f_.empty()) {
  //  stat_f_.resize(mix_num_*feat_dim_);
  //}

  if (ivector_.empty()) {
    ivector_.resize(ivector_dim_);
  }
  //stream.read((char *)&stat_n_[0], sizeof(float)* mix_num_);
  //stream.read((char *)&stat_f_[0], sizeof(float)* mix_num_* feat_dim_);
  stream.read((char *)&ivector_[0], sizeof(float)* ivector_dim_);
  stream.read(md5_, sizeof(float)* MD5_LEN);
}

string SpeakerModel::GetCurrentTime() const {
  string time;
  time_t tt =  std::time(NULL);
  struct tm *ptm = gmtime(&tt);
  vector<char> foo(240, 0);
  if (0 >= strftime(&foo[0], sizeof(char)*foo.size(), "%Y%m%d%H%M%S", ptm))
    throw  runtime_error("get current time fail");
  time = string(&foo[0], &foo[0] + strlen(&foo[0]));
  return time;
}

string SpeakerModel::GetMd5(string bodyRequest) const {
  MyMD5 md5;
  md5.update(bodyRequest);
  const byte *bodymd5 = md5.digest();
  return string(reinterpret_cast<const char *>(bodymd5));
}

//void SpeakerModel::UpdateStats(const DoubleVector &gamma,
//                               const DoubleMatrix &x) {
//  idec::IDEC_ASSERT(mix_num_ == gamma.Size());
//  idec::IDEC_ASSERT(feat_dim_ == x.Cols());
//  for (int i = 0; i < mix_num_; ++i) {
//    stat_n_[i] = gamma(i);
//    for (int j = 0; j < feat_dim_; ++j) {
//      stat_f_[i*feat_dim_ + j] = x(i, j);
//    }
//  }
//}
//
//void SpeakerModel::GetStats(DoubleVector &gamma, DoubleMatrix &x) const {
//  if (gamma.Size() != mix_num_) {
//    gamma.Resize(mix_num_);
//  }
//
//  if ((x.Cols() != feat_dim_) || (x.Rows() != mix_num_)) {
//    x.Resize(mix_num_, feat_dim_);
//  }
//
//  for (int i = 0; i < mix_num_; ++i) {
//    gamma(i) = stat_n_[i];
//    for (int j = 0; j < feat_dim_; ++j) {
//      x(i, j) = stat_f_[i*feat_dim_ + j];
//    }
//  }
//}

void SpeakerModel::GetIvector(DoubleVector &ivector) const {
  idec::IDEC_ASSERT(ivector_dim_ > 0);
  if (ivector.Size() != ivector_dim_) {
    ivector.Resize(ivector_dim_);
  }

  for (int i = 0; i < ivector_dim_; ++i) {
    ivector(i) = ivector_[i];
  }
}

void SpeakerModel::UpdateModel(const string &spk_id, const string &feat_type,
                               int utt_num, int frame_num, int feat_dim, int mix_num,
                               const DoubleVector &iv) {
  if (!ivector_.empty()) {
    idec::IDEC_ERROR << "Only support update empty model.";
  }

  int spk_id_copy_len = spk_id.size();
  if (spk_id.size() > SPK_ID_LEN) {
    spk_id_copy_len = SPK_ID_LEN;
    idec::IDEC_WARN << "speaker id truncation happen.";
  }

  int feat_type_copy_len = feat_type.size();
  if (feat_type.size() > FEAT_TYPE_LEN) {
    feat_type_copy_len = FEAT_TYPE_LEN;
    idec::IDEC_WARN << "feat type truncation happen.";
  }

  memcpy(feat_type_, feat_type.c_str(), feat_type_copy_len);
  memcpy(spk_id_, spk_id.c_str(), spk_id_copy_len);

  frame_num_ = frame_num;
  mix_num_ = mix_num;
  feat_dim_ = feat_dim;

  idec::IDEC_ASSERT(frame_num_ > 0);
  idec::IDEC_ASSERT(mix_num_ > 0);
  idec::IDEC_ASSERT(feat_dim_ > 0);

  //if (stat_n_.size() != mix_num_) {
  //  stat_n_.resize(mix_num_);
  //}

  // const int sv_dim = mix_num_*feat_dim_;
  //if (stat_f_.size() != sv_dim) {
  //  stat_f_.resize(sv_dim);
  //}

  //memset(&stat_n_[0], 0, sizeof(float)* mix_num_);
  //memset(&stat_f_[0], 0, sizeof(float)* sv_dim);

  utt_num_ = utt_num;
  //UpdateStats(gamma, x);

  ivector_dim_ = iv.Size();
  idec::IDEC_ASSERT(ivector_dim_ > 0);
  ivector_.resize(ivector_dim_);
  for (int i = 0; i < ivector_dim_; ++i) {
    ivector_[i] = iv(i);
  }

  string date = GetCurrentTime();
  memcpy(create_time_, date.c_str(), date.length());
}

bool SpeakerModel::IsValid(const string &spk_mdl) const {
  unsigned int pos = spk_mdl.size() - 16;
  string content = spk_mdl.substr(0, pos);
  string sMd5 = GetMd5(content);
  string md5 = spk_mdl.substr(pos);
  return strcmp(md5.c_str() , sMd5.c_str()) == 0;
}
