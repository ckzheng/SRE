#ifndef _AHC_H_
#define _AHC_H_

#include "seg.h"
#include "bic.h"
#include "speaker_cluster.h"
#include "new_vector.h"
#include "new_matrix.h"
#include "ivector_transform.h"
#include "plda.h"
#include "ivector_extract_pipeline_new.h"
#include "spkdiar_serialize.h"
#include "denoise.h"

namespace alspkdiar {

class UpperTriangularMatrix {
 public:
  UpperTriangularMatrix(unsigned int size) {
    unsigned int alloc;
    current_size_ = size;
    capacity_size_ = size;
    alloc = capacity_size_ * capacity_size_;
    data_.resize(alloc);
    buf_.resize(alloc* 0.5 + 1);
  }

  void Set(unsigned int row, unsigned int col, float value) {
    idec::IDEC_ASSERT(row < current_size_);
    idec::IDEC_ASSERT(col < current_size_);
    idec::IDEC_ASSERT(row < col);
    data_[row * capacity_size_ + col] = value;
  }

  float Get(unsigned int row, unsigned int col) const {
    idec::IDEC_ASSERT(row < current_size_);
    idec::IDEC_ASSERT(col < current_size_);
    idec::IDEC_ASSERT(row < col);
    return data_[row * capacity_size_ + col];
  }

  float &operator()(unsigned int row, unsigned int col) {
    idec::IDEC_ASSERT(row < current_size_);
    idec::IDEC_ASSERT(col < current_size_);
    idec::IDEC_ASSERT(row < col);
    return data_[row * capacity_size_ + col];
  }

  void Inc() {
    ++current_size_;
  }

  void RemoveTwoRows(unsigned int row , unsigned int col) {
    idec::IDEC_ASSERT(row < current_size_);
    idec::IDEC_ASSERT(col < current_size_);
    idec::IDEC_ASSERT(current_size_ > 2);
    int cnt = 0;
    for (int i = 0; i < current_size_; ++i) {
      for (int j = i + 1; j < current_size_; ++j) {
        if ((i == row) || (j == col) || (i == col) || (j == row)) {
          continue;
        }
        buf_[cnt++] = data_[i*capacity_size_ + j];
      }
    }

    cnt = 0;
    for (int i = 0; i < current_size_ - 2; i++) {
      for (int j = i + 1; j < current_size_ - 2; j++) {
        data_[i*capacity_size_ + j] = buf_[cnt++];
      }
    }
    current_size_ -= 2;
  }

  void RemoveOneRows(unsigned int row) {
    idec::IDEC_ASSERT(row < current_size_);
    idec::IDEC_ASSERT(current_size_ > 1);
    int cnt = 0;
    for (int i = 0; i < current_size_; ++i) {
      for (int j = i + 1; j < current_size_; ++j) {
        if ((i == row) || (j == row)) {
          continue;
        }
        buf_[cnt++] = data_[i*capacity_size_ + j];
      }
    }

    cnt = 0;
    for (int i = 0; i < current_size_ - 1; i++) {
      for (int j = i + 1; j < current_size_ - 1; j++) {
        data_[i*capacity_size_ + j] = buf_[cnt++];
      }
    }
    current_size_ -= 1;
  }

  void Max(unsigned int &row, unsigned int &col, float &max_score) const {
    float score;
    row = 0, col = 1;
    idec::IDEC_ASSERT(current_size_ > 0);
    max_score = data_[1];
    for (int i = 0; i < current_size_; ++i) {
      for (int j = i + 1; j < current_size_; ++j) {
        score = data_[i*capacity_size_ + j];
        if (score > max_score) {
          row = i;
          col = j;
          max_score = score;
        }
      }
    }
  }

  void Min(unsigned int &row, unsigned int &col, float &min_score) const {
    float score;
    row = 0, col = 1;
    idec::IDEC_ASSERT(current_size_ > 0);
    min_score = data_[1];
    for (int i = 0; i < current_size_; ++i) {
      for (int j = i + 1; j < current_size_; ++j) {
        score = data_[i*capacity_size_ + j];
        if (score < min_score) {
          row = i;
          col = j;
          min_score = score;
        }
      }
    }
  }
 private:
  vector<float> data_;
  vector<float> buf_;
  unsigned int current_size_;
  unsigned int capacity_size_;
};

class Ahc {
 public:
  Ahc(Mfcc *mfcc, Bic *bic,
      ResourceManager *res):cmvn_matrix_(res->CmvnMatrix()),
    ahc_opt_(res->AhcConfigOptions()) {
    debug_ = false;
    this->mfcc_ = mfcc;
    this->bic_ = bic;
    this->res_ = res;
    this->C_ = 2;
    this->total_frames_ = 0;
    this->plda_ = new Plda(res_->PldaConfigOption(), res_->PldaRes());
    ivector_pipe_ = new IvectorExtractPipeline(res);
  }

  ~Ahc() {
    idec::IDEC_DELETE(plda_);
    idec::IDEC_DELETE(ivector_pipe_);
  }

  void Init(string wave_path) {
    wave_path_ = wave_path;
  }

  int LengthCheck(const SegCluster &cluster, SegCluster &cluster_long,
                  SegCluster &cluster_short) {
    const unsigned int threshold = 500;
    for (unsigned int i = cluster.Size() - 1; i > 0; --i) {
      const Seg &segment = cluster.GetSeg(i);
      if (segment.Length() >= threshold) {
        cluster_long.Add(segment);
        total_frames_ += segment.Length();
      } else {
        cluster_short.Add(segment);
        total_frames_ += segment.Length();
      }
    }
    return ALS_OK;
  }

  int Distance(SegCluster &cluster1, SegCluster &cluster2, float &dist) {
    return bic_->DeltaScore(cluster1, cluster2, dist);
  }

  int ComputeDistanceByMeanPair(SegCluster &cluster1, SegCluster &cluster2,
                                float &dist) {
    ComputeIvectorByMean(cluster1);
    ComputeIvectorByMean(cluster2);

    const double total_utt = cluster1.Size() + cluster2.Size();

    float total_score = 0.0, tmp;
    for (int i = 0; i < cluster1.Size(); ++i) {
      const Seg &segment = cluster1.GetSeg(i);
      tmp = ComputePldaScore(cluster2.Ivector(), segment.ivector);
      total_score += tmp;
    }

    for (int i = 0; i < cluster2.Size(); ++i) {
      const Seg &segment = cluster2.GetSeg(i);
      tmp = ComputePldaScore(cluster1.Ivector(), segment.ivector);
      total_score += tmp;
    }

    dist = total_score * (1 / total_utt);
    return 0;
  }

  int ComputeDistanceByPair(SegCluster &cluster1, SegCluster &cluster2,
                            float &dist) {
    ComputeIvectorByMean(cluster1);
    ComputeIvectorByMean(cluster2);

    const double total_utt = cluster1.Size() * cluster2.Size();
    vector<int> c1, c2;
    for (int i = 0; i < cluster1.Size(); ++i) {
      c1.push_back(cluster1.GetSeg(i).label);
    }

    for (int i = 0; i < cluster2.Size(); ++i) {
      c2.push_back(cluster2.GetSeg(i).label);
    }

    if (total_utt == 1) {
      dist = ComputePldaScore(cluster1.GetSeg(0).ivector,
                              cluster2.GetSeg(0).ivector);
      return 0;
    }

    double score = 0.0;
    for (int i = 0; i < c1.size(); ++i) {
      for (int j= 0; j < c2.size(); j++) {
        score += matrix_[c1[i]*init_dim_ + c2[j]];
      }
    }

    dist = score * (1 / total_utt);
    return 0;
  }

  int ComputeDistanceByPair(SpeakerCluster &spk1, SpeakerCluster &spk2,
                            float &dist) {
    const double total_utt = spk1.Size() * spk2.Size();
    vector<int> c1, c2;
    for (int i = 0; i < spk1.Size(); ++i) {
      c1.push_back(spk1.Get(i).Label());
    }

    for (int i = 0; i < spk2.Size(); ++i) {
      c2.push_back(spk2.Get(i).Label());
    }

    if (total_utt == 1) {
      dist = matrix_[c1[0] * init_dim_ + c2[0]];
      return 0;
    }

    double score = 0.0;
    for (int i = 0; i < c1.size(); ++i) {
      for (int j = 0; j < c2.size(); j++) {
        score += matrix_[c1[i] * init_dim_ + c2[j]];
      }
    }

    dist = score * (1 / total_utt);
    return 0;
  }

  int ComputeCosineDistance(SegCluster &cluster1, SegCluster &cluster2,
                            float &dist) {
    ComputeIvectorByMergeFeature(cluster1);
    ComputeIvectorByMergeFeature(cluster2);
    //dist = ComputeScore_(cluster1.Ivector(), cluster2.Ivector());
    dist = ComputeCosineScore(cluster1.Ivector(), cluster2.Ivector());
    return 0;
  }

  int ComputeCosineDistance(SpeakerCluster &spk_cluster1,
                            SpeakerCluster &spk_cluster2, float &dist) {
    if (spk_cluster1.IsIvectorNeedUpdate()) {
      SegCluster cluster1;
      for (int i = 0; i < spk_cluster1.Size(); ++i) {
        cluster1.Add(spk_cluster1.Get(i));
      }
      ComputeIvectorByMergeFeature(cluster1);
      spk_cluster1.SetIvector(cluster1.Ivector());
    }

    if (spk_cluster2.IsIvectorNeedUpdate()) {
      SegCluster cluster2;
      for (int i = 0; i < spk_cluster2.Size(); ++i) {
        cluster2.Add(spk_cluster2.Get(i));
      }
      ComputeIvectorByMergeFeature(cluster2);
      spk_cluster2.SetIvector(cluster2.Ivector());
    }

    //dist = ComputeScore_(cluster1.Ivector(), cluster2.Ivector());
    dist = ComputeCosineScore(spk_cluster1.Ivector(), spk_cluster2.Ivector());
    return 0;
  }

  int ComputeDistance(SegCluster &cluster1, SegCluster &cluster2, float &dist) {
    //DoubleMatrix feats;
    //string file_name = "003221273.mkv_10.15.34.78_41330_1_0_M_0006.mfcc";
    //mfcc_->LoadMfcc(file_name.c_str(), feats);
    //DoubleVector ivector;
    //ivector_pipe_->ExtractIvector(feats, ivector);
    ComputeIvectorByMergeFeature(cluster1);
    ComputeIvectorByMergeFeature(cluster2);
    //dist = ComputeScore_(cluster1.Ivector(), cluster2.Ivector());
    dist = ComputePldaScore(cluster1.Ivector(), cluster1.utt_num_,
                            cluster2.Ivector());
    return 0;
  }

  int ComputePldaDistance_(SegCluster &cluster1, SegCluster &cluster2,
                           float &dist) {
    ComputeIvectorByMergeFeature(cluster1);
    ComputeIvectorByMergeFeature(cluster2);
    dist = ComputePldaScore(cluster1.Ivector(), cluster2.Ivector());
    //dist = ComputeCosineScore(cluster1.Ivector(), cluster2.Ivector());
    return 0;
  }

  int ComputePldaDistance(SegCluster &cluster1, SegCluster &cluster2,
                          float &dist) {
    ComputeIvectorByMean(cluster1);
    ComputeIvectorByMean(cluster2);
    dist = ComputePldaScore(cluster1.Ivector(), cluster2.Ivector());
    //dist = ComputeCosineScore(cluster1.Ivector(), cluster2.Ivector());
    return 0;
  }

  int ComputeIvectorByMean(SegCluster &cluster) {
    const int N = cluster.Size();
    DoubleVector ivector;
    if (cluster.IsIvectorNeedUpdate()) {
      for (int i = 0; i < N; ++i) {
        SegCluster tmp_cluster;
        tmp_cluster.Add(cluster.GetSeg(i));
        DoubleMatrix feats;
        mfcc_->MergeFeature(tmp_cluster, feats);
        DoubleVector ivector;
        ivector_pipe_->ExtractIvector(feats, ivector);
        IvectorTransform::LengthNorm(ivector);
        cluster.GetSeg(i).ivector = ivector;
      }

      for (int i = 0; i < N; ++i) {
        const Seg &segment = cluster.GetSeg(i);
        if (ivector.Size() != segment.ivector.Size()) {
          ivector.Resize(segment.ivector.Size());
        }
        ivector += segment.ivector;
      }
      ivector.Scale(1.0 / N);
      cluster.SetIvector(ivector);
    }
    return 0;
  }

  void DoCMS(DoubleMatrix &feat) {
    const int num_frm = feat.Rows();
    const int dims = feat.Cols();
    idec::IDEC_ASSERT(cmvn_matrix_.Rows() == 2);
    //idec::IDEC_ASSERT(cmvn_matrix_.Cols() == dims + 1);
    for (unsigned int i = 0; i < num_frm; ++i) {
      for (unsigned int j = 0; j < dims; ++j) {
        feat(i, j) -= cmvn_matrix_(0, j);
      }
    }
  }

  int ComputeIvectorByMergeFeature(SegCluster &cluster) {
    if (cluster.IsIvectorNeedUpdate()) {
      DoubleMatrix feats;
      //mfcc_->MergeFeature(cluster, feats);
      mfcc_->MergeFeatureNew(cluster, feats);
      bool do_cmvn = true;
      if (do_cmvn && feats.Rows()) {
        DoCMS(feats);
      }
      DoubleVector ivector;
      ivector_pipe_->ExtractIvector(feats, ivector);
      //IvectorTransform::LengthNorm(ivector);
      cluster.SetIvector(ivector);
    }
    return 0;
  }

  float ComputeLdaPldaScore(DoubleVector ivector, DoubleVector trial_ivector) {
    DoubleVector transformed_ivector, transformed_trial_ivector;
    IvectorTransform::LengthNorm(ivector);
    //IvectorTransform::LengthNorm(trial_ivector);
    ivector -= res_->IvRes().mean;
    trial_ivector -= res_->IvRes().mean;
    //IvectorTransform::LengthNorm(ivector);
    //IvectorTransform::LengthNorm(trial_ivector);
    IvectorTransform::LdaTransform(res_->LdaMatrix(), ivector,
                                   transformed_ivector);
    IvectorTransform::LdaTransform(res_->LdaMatrix(), trial_ivector,
                                   transformed_trial_ivector);
    const int utt_num = 1;
    return plda_->LogLikelihoodRatio(transformed_ivector, utt_num,
                                     transformed_trial_ivector);
  }

  float ComputeCosineScore(const DoubleVector &ivector,
                           const DoubleVector &trial_ivector) {
    double sum = 0.0, sum_square;
    for (int i = 0; i < ivector.Size(); ++i) {
      sum += ivector(i) * trial_ivector(i);
    }

    sum_square = ivector.Norm(2.0) * trial_ivector.Norm(2.0);
    return sum / sum_square;

  }

  float ComputePldaScore(DoubleVector ivector, int utt_num,
                         DoubleVector trial_ivector) {
    //DoubleVector transformed_ivector, transformed_trial_ivector;
    //IvectorTransform::LengthNorm(ivector);
    // diff with speaker verification.
    //IvectorTransform::LengthNorm(trial_ivector);
    ivector -= res_->IvRes().mean;
    trial_ivector -= res_->IvRes().mean;
    //IvectorTransform::LengthNorm(ivector);
    //IvectorTransform::LengthNorm(trial_ivector);
    //IvectorTransform::LdaTransform(res_->LdaMatrix(), ivector, transformed_ivector);
    //IvectorTransform::LdaTransform(res_->LdaMatrix(), trial_ivector, transformed_trial_ivector);
    //const int utt_num = ahc_opt_.utterance_number;
    //return plda_->LogLikelihoodRatio(transformed_ivector, utt_num, transformed_trial_ivector);
    return plda_->LogLikelihoodRatio(ivector, utt_num, trial_ivector);
  }

  float ComputePldaScore(DoubleVector ivector, DoubleVector trial_ivector) {
    //DoubleVector transformed_ivector, transformed_trial_ivector;
    //IvectorTransform::LengthNorm(ivector);
    // diff with speaker verification.
    //IvectorTransform::LengthNorm(trial_ivector);
    ivector -= res_->IvRes().mean;
    trial_ivector -= res_->IvRes().mean;
    //IvectorTransform::LengthNorm(ivector);
    //IvectorTransform::LengthNorm(trial_ivector);
    //IvectorTransform::LdaTransform(res_->LdaMatrix(), ivector, transformed_ivector);
    //IvectorTransform::LdaTransform(res_->LdaMatrix(), trial_ivector, transformed_trial_ivector);
    const int utt_num = ahc_opt_.utterance_number;
    //return plda_->LogLikelihoodRatio(transformed_ivector, utt_num, transformed_trial_ivector);
    return plda_->LogLikelihoodRatio(ivector, utt_num, trial_ivector);
  }

  int MergeCluster(const SegCluster &cluster1,
                   const SegCluster &cluster2, SegCluster &cluster) {
    const FullGaussian *gaussian1 = cluster1.Gaussian();
    const FullGaussian *gaussian2 = cluster2.Gaussian();
    FullGaussian gaussian = *gaussian1;
    gaussian.Merge(gaussian2);
    gaussian.GetLDet();
    cluster.SetGaussian(gaussian);
    cluster.Add(cluster1);
    cluster.Add(cluster2);
    return ALS_OK;
  }

  int MergeIvector(const SegCluster &cluster1,
                   const SegCluster &cluster2, SegCluster &cluster) {
    DoubleVector ivector1 = cluster1.Ivector();
    IvectorTransform::LengthNorm(ivector1);
    DoubleVector ivector2 = cluster2.Ivector();
    IvectorTransform::LengthNorm(ivector2);
    double ratio = (double)cluster1.Length() / (cluster1.Length() +
                   cluster2.Length());
    DoubleVector ivector = ivector1 * ratio + ivector2 * (1-ratio);
    //ivector.Scale(0.5);
    IvectorTransform::LengthNorm(ivector);
    cluster.SetIvector(ivector);
    ++cluster.utt_num_;
    return ALS_OK;
  }

  void PrintInfo(string wave_path, SpeakerCluster spk_cluster, int step) {
    Serialize ser;
    vector<char> wave;
    ser.ReadWave(wave_path, wave);
    string file_name, out_wav_path, label_path;
    ser.GetFileName(wave_path, file_name);
    for (int i = 0; i < spk_cluster.Size(); ++i) {
      stringstream stream;
      stream << file_name;
      stream << "_step";
      stream << step;
      stream << "_";
      stream << i + 1;
      out_wav_path = stream.str() + ".wav";
      label_path = stream.str() + ".lbl";
      ser.SaveWaveAndLabel(&wave[0], out_wav_path, label_path, spk_cluster.Get(i));
      cout << "speaker cluster " << i << " length is " << spk_cluster.Get(
             i).Length() << endl;
    }
  }

  int Clustering(SpeakerCluster &spk_cluster) {
    float max_score, score;
    //ReClustering(spk_cluster);
    if (spk_cluster.Size() == 2) {
      return 0;
    }

    for (int i = spk_cluster.Size() - 1; i >= 0; --i) {
      if (spk_cluster.Get(i).Length() < 100) {
        spk_cluster.Remove(i);
      }
    }

    debug_ = false;
    if (debug_) {
      const int step = 3;
      PrintInfo(wave_path_, spk_cluster, step);
    }
    debug_ = false;

    int spk_nums = spk_cluster.Size();
    UpperTriangularMatrix dist_matrix(spk_nums);
    for (int i = 0; i < spk_nums; ++i) {
      for (int j = i + 1; j < spk_nums; ++j) {
        ComputePldaDistance_(spk_cluster.Get(i), spk_cluster.Get(j), score);
        //ComputeCosineDistance(spk_cluster.Get(i), spk_cluster.Get(j), score);
        dist_matrix(i,j) = score;
      }
    }

    matrix_.clear();
    matrix_.resize(spk_nums*spk_nums);
    for (int i = 0; i < spk_nums; ++i) {
      for (int j = i + 1; j < spk_nums; ++j) {
        matrix_[i*spk_nums + j] = dist_matrix(i, j);
        matrix_[j*spk_nums + i] = matrix_[i*spk_nums + j];
      }
    }

    vector<float> score_sum(spk_nums);
    vector<float> score_square(spk_nums);
    for (int i = 0; i < spk_nums; ++i) {
      score_sum[i] = 0.0;
      for (int j = 0; j < spk_nums; ++j) {
        if (i == j) {
          continue;
        }
        score_sum[i] += matrix_[i*spk_nums + j];
      }
    }

    for (int i = 0; i < spk_nums; ++i) {
      score_square[i] = 0.0;
      float mean = score_sum[i] / (spk_nums - 1);
      for (int j = 0; j < spk_nums; ++j) {
        if (i == j) {
          continue;
        }
        score_square[i] += (matrix_[i*spk_nums + j] - mean) * (matrix_[i*spk_nums + j]
                           - mean);
      }
    }

    int min_square_index = 0;
    float min_score = score_square[0];
    for (int i = 1; i < spk_nums; ++i) {
      if (score_square[i] < min_score) {
        min_square_index = i;
        min_score = score_square[i];
      }
    }

    float global_mean = 0.0, mean = 0.0;
    for (int i = 0; i < spk_nums; ++i) {
      global_mean += score_sum[i] / (spk_nums - 1);
    }

    mean = score_sum[min_square_index] / (spk_nums - 1);
    global_mean = global_mean / spk_nums;

    init_dim_ = spk_nums;
    if ((mean > global_mean) && (mean > 0.0)) {
      --spk_nums;
      init_dim_ = spk_nums;
      spk_cluster.Remove(min_square_index);
      dist_matrix.RemoveOneRows(min_square_index);
    }

    for (int i = 0; i < spk_nums; ++i) {
      spk_cluster.Get(i).SetLabel(i);
    }

    matrix_.clear();
    matrix_.resize(spk_nums*spk_nums);
    for (int i = 0; i < spk_nums; ++i) {
      for (int j = i + 1; j < spk_nums; ++j) {
        matrix_[i*spk_nums + j] = dist_matrix(i, j);
        matrix_[j*spk_nums + i] = matrix_[i*spk_nums + j];
      }
    }

    SegCluster cluster;
    unsigned int row, col;
    vector<SpeakerCluster> spks;
    for (int i = 0; i < spk_nums; ++i) {
      SpeakerCluster spk_tmp;
      spk_tmp.Add(spk_cluster.Get(i));
      spk_tmp.SetIvector(spk_cluster.Get(i).Ivector());
      spks.push_back(spk_tmp);
    }

    spk_nums = spks.size();
    while (spk_nums > this->C_) {
      dist_matrix.Max(row, col, max_score);
      dist_matrix.RemoveTwoRows(row, col);
      SpeakerCluster spk1 = spks[row];
      SpeakerCluster spk2 = spks[col];
      for (int i = 0; i < spk2.Size(); ++i) {
        spk1.Add(spk2.Get(i));
      }

      dist_matrix.Inc();
      for (int i = 0, j = 0; i < spk_nums; ++i) {
        if ((i == row) || (i == col)) {
          continue;
        }
        ComputeDistanceByPair(spk1, spks[i], score);
        //ComputeCosineDistance(spk1, spks[i], score);
        dist_matrix(j++, spk_nums - 2) = score;
      }

      spks.erase(spks.begin() + col);
      spks.erase(spks.begin() + row);
      spks.push_back(spk1);
      spk_nums = spks.size();
    }

    spk_cluster.Clear();
    for (int i = 0; i < spks.size(); ++i) {
      SegCluster tmp_cluster;
      SpeakerCluster spk = spks[i];
      for (int j = 0; j < spk.Size(); ++j) {
        tmp_cluster.Add(spk.Get(j));
      }
      spk_cluster.Add(tmp_cluster);
    }

    //debug_ = true;
    if (debug_) {
      const int step = 400;
      PrintInfo(wave_path_, spk_cluster, step);
    }
    //debug_ = false;
    return ALS_OK;
  }

  int PreClustering_(SpeakerCluster &spk_cluster, int cluster_number = 2,
                     bool use_threshold = true) {
    //const int step = 0;
    //PrintInfo(wave_path_, spk_cluster, step);

    int spk_nums = spk_cluster.Size();
    UpperTriangularMatrix dist_matrix(spk_nums);

    float score;
    for (int i = 0; i < spk_nums; ++i) {
      for (int j = i + 1; j < spk_nums; ++j) {
        Distance(spk_cluster.Get(i), spk_cluster.Get(j), score);
        dist_matrix(i,j) = score;
      }
    }

    SegCluster cluster;
    unsigned int  row, col;
    float min_score;
    while (spk_nums > cluster_number) {
      dist_matrix.Min(row, col, min_score);
      dist_matrix.RemoveTwoRows(row, col);

      const SegCluster &cluster1 = spk_cluster.Get(row);
      const SegCluster &cluster2 = spk_cluster.Get(col);

      MergeCluster(cluster1, cluster2, cluster);

      dist_matrix.Inc();
      for (int i = 0, j = 0; i < spk_nums; ++i) {
        if ((i == row) || (i == col)) {
          continue;
        }
        Distance(spk_cluster.Get(i), cluster, score);
        dist_matrix(j++, spk_nums - 2) = score;
      }

      spk_cluster.Remove(col);
      spk_cluster.Remove(row);
      spk_cluster.Add(cluster);
      cluster.Clear();
      spk_nums = spk_cluster.Size();
      if (use_threshold) {
        if (debug_) {
          if (spk_cluster.Size() == ahc_opt_.class_number) {
            const int step = 100;
            PrintInfo(wave_path_, spk_cluster, step);
          }
        }

        if (spk_cluster.Size() == ahc_opt_.class_number) {
          unsigned int length = spk_cluster.Length();
          unsigned int average_length = length / spk_cluster.Size();
          const int threshold = ahc_opt_.average_length; // 10s
          if (average_length > threshold) {
            break;
          }
        }
      }
    }
    return ALS_OK;
  }

  int ReClustering(SpeakerCluster &spk_cluster) {
    int spk_nums = spk_cluster.Size();
    SpeakerCluster new_spk_cluster;
    for (int i = 0; i < spk_nums; ++i) {
      const SegCluster &cluster = spk_cluster.Get(i);
      const int length = cluster.Length();
      const int time_length = 1000;
      const int cluster_number = length / time_length;
      if (cluster_number < 2) {
        new_spk_cluster.Add(cluster);
        continue;
      }

      SpeakerCluster tmp_spk_cluster;
      for (int j = 0; j < cluster.Size(); ++j) {
        SegCluster new_cluster;
        new_cluster.Add(cluster.GetSeg(j));
        tmp_spk_cluster.Add(new_cluster);
      }

      PreClustering_(tmp_spk_cluster, cluster_number, false);
      for (int k = 0; k < tmp_spk_cluster.Size(); ++k) {
        //if (tmp_spk_cluster.Get(k).Length() < 300) {
        //	continue;
        //}
        new_spk_cluster.Add(tmp_spk_cluster.Get(k));
      }

      if (debug_) {
        int step = 200;
        PrintInfo(wave_path_, tmp_spk_cluster, step++);
      }
    }

    spk_cluster = new_spk_cluster;
    return 0;
  }

  int PostClustering_(SpeakerCluster &spk_cluster) {
    //ReClustering(spk_cluster);
    int spk_nums = spk_cluster.Size();
    //const int average_len = spk_cluster.Length() / spk_nums;
    // for (int i = spk_nums - 1; i >= 0; --i) {
    //   const SegCluster &cluster = spk_cluster.Get(i);
    //if ((cluster.Length() < 0.5 * average_len) && (cluster.Length() < 500)) {
    //     spk_cluster.Remove(i);
    //   }
    // }

    //spk_nums = spk_cluster.Size();
    if (spk_nums == 2) {
      return 0;
    }

    float score, max_score;
    UpperTriangularMatrix dist_matrix(spk_nums);
    for (int i = 0; i < spk_nums; ++i) {
      for (int j = i + 1; j < spk_nums; ++j) {
        ComputeDistance(spk_cluster.Get(i), spk_cluster.Get(j), score);
        dist_matrix(i,j) = score;
      }
    }

    SegCluster cluster;
    unsigned int row, col;
    while (spk_nums > this->C_) {
      dist_matrix.Max(row, col, max_score);
      dist_matrix.RemoveTwoRows(row, col);
      const SegCluster &cluster1 = spk_cluster.Get(row);
      const SegCluster &cluster2 = spk_cluster.Get(col);
      cluster.Add(cluster1);
      cluster.Add(cluster2);
      ComputeIvectorByMergeFeature(cluster);
      //MergeIvector(cluster1, cluster2, cluster);
      //MergeCluster(cluster1, cluster2, cluster);

      dist_matrix.Inc();
      for (int i = 0, j = 0; i < spk_nums; ++i) {
        if ((i == row) || (i == col)) {
          continue;
        }
        ComputeDistance(cluster, spk_cluster.Get(i), score);
        dist_matrix(j++, spk_nums-2) = score;
      }

      spk_cluster.Remove(col);
      spk_cluster.Remove(row);
      spk_cluster.Add(cluster);
      cluster.Clear();
      spk_nums = spk_cluster.Size();
    }
    return ALS_OK;
  }

  int Process(SpeakerCluster &spk_cluster) {
    PreClustering_(spk_cluster);
    Clustering(spk_cluster);
    return ALS_OK;
  }

 private:
  // number of total classes
  unsigned int C_;
  bool debug_;
  Mfcc *mfcc_;
  Bic *bic_;
  Plda *plda_;
  string wave_path_;
  unsigned int init_dim_;
  unsigned int total_frames_;
  vector<float> matrix_;
  DoubleVector dtmf_noise_;
  const AhcOptions &ahc_opt_;
  const DoubleMatrix &cmvn_matrix_;
  ResourceManager *res_;
  IvectorExtractPipeline *ivector_pipe_;
};

}

#endif
