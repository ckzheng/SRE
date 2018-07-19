#include "ahc.h"

namespace alspkdiar {

Ahc::Ahc(Bic& bic) {
  this->bic_ = bic;
  this->total_frames_ = 0;
  this->C_ = 2;
}

ALS_RETCODE Ahc::LengthCheck(SpeakerCluster *spk, SpeakerCluster *spk_long,
                             SpeakerCluster *spk_short) {
  Seg *segment = NULL;
  const unsigned int threshold = 500;
  for (unsigned int i = spk->Size() - 1; i > 0; --i) {
    segment = static_cast<Seg *>(spk->GetSeg(i));
    if (segment->Length() >= threshold) {
      spk_long->AddSpeaker(segment);
      total_frames_ += segment->Length();
    } else {
      spk_short->AddSpeaker(segment);
      total_frames_ += segment->Length();
    }
  }
  return ALS_OK;
}

Ahc::~Ahc() {
}

ALS_RETCODE Ahc::Distance(AbstractSeg *abstract_seg1,
                          AbstractSeg *abstract_seg2, float &dist) {
  ALS_RETCODE ret = ALS_OK;
  ret = bic_->DeltaScore(abstract_seg1, abstract_seg2, dist);
  return ret;
}

ALS_RETCODE Ahc::MergeCluster(AbstractSeg *abstract_seg1,
                              AbstractSeg *abstract_seg2, SegCluster *&cluster) {
  FullGaussian *gaussian, *gaussian1, *gaussian2;
  cluster = new SegCluster();
  cluster->SetLength(abstract_seg1->Length() + abstract_seg2->Length());
  gaussian1 = abstract_seg1->GetGaussian();
  gaussian2 = abstract_seg2->GetGaussian();
  gaussian = gaussian1->Clone();
  gaussian->Merge(gaussian2);
  cluster->SetGaussian(gaussian);
  cluster->AddSegWithoutMerge(abstract_seg1);
  cluster->AddSegWithoutMerge(abstract_seg2);
  return ALS_OK;
}

ALS_RETCODE Ahc::Clustering(SpeakerCluster *spk_cluster) {
  int i,j, row, col, spk_nums, orig_spk_nums, buf_size;
  float min_score, score;
  AbstractSeg *abstract_seg1 = NULL, *abstract_seg2 = NULL;
  SegCluster *seg_cluster = NULL;
  orig_spk_nums = spk_cluster->Size();
  buf_size = orig_spk_nums * orig_spk_nums / 2;
  std::vector<float> buf = std::vector<float>();
  buf.reserve(buf_size);
  // calculate bic distance matrix
  float **dist_matrix = new float*[orig_spk_nums];
  for (i = 0; i < orig_spk_nums; ++i) {
    dist_matrix[i] = new float[orig_spk_nums];
    memset(dist_matrix[i], 0, sizeof(float) * orig_spk_nums);
    for (j = i + 1; j < orig_spk_nums; ++j) {
      Distance(spk_cluster->GetSeg(i), spk_cluster->GetSeg(j), score);
      dist_matrix[i][j] = score;
    }
  }

  bool flag;
  spk_nums = spk_cluster->Size();
  while (spk_nums > this->C_) {
    row = -1;
    col = -1;
    min_score = 10e10;
    flag = false;
    for ( i = 0; i < spk_nums; ++i) {
      for (j = i + 1; j < spk_nums; ++j) {
        score = dist_matrix[i][j];
        if (score < min_score) {
          row = i;
          col = j;
          min_score = score;
        }
      }
    }

    // remove two columns of bic distance matrix
    for (i = 0; i < spk_nums; ++i) {
      for (j = i + 1; j < spk_nums; ++j) {
        if ((i == row) || (j == col) || (i == col) || (j == row)) {
          continue;
        }
        buf.push_back(dist_matrix[i][j]);
      }
    }

    // remove two columns of bic distance matrix
    int cnt = 0;
    for (i = 0; i < spk_nums - 2; i++) {
      for (j = i + 1; j < spk_nums - 2; j++) {
        dist_matrix[i][j] = buf[cnt++];
      }
    }

    abstract_seg1 = spk_cluster->GetSeg(row);
    abstract_seg2 = spk_cluster->GetSeg(col);
    MergeCluster(abstract_seg1, abstract_seg2, seg_cluster);

    //add one column of bic distance matrix
    for (i = 0, j = 0; i < spk_nums; ++i) {
      if ((i == row) || (i == col)) {
        continue;
      }
      Distance(spk_cluster->GetSeg(i), seg_cluster, score);
      dist_matrix[j++][spk_nums - 2] = score;
    }

    spk_cluster->RemoveSpeaker(col);
    spk_cluster->RemoveSpeaker(row);
    spk_cluster->AddSpeaker(seg_cluster);

    buf.clear();
    spk_nums = spk_cluster->Size();
  }

  if (dist_matrix != NULL) {
    for (i = 0; i < orig_spk_nums; ++i) {
      idec::IDEC_DELETE_ARRAY(dist_matrix[i]);
    }
    idec::IDEC_DELETE_ARRAY(dist_matrix);
  }
  return ALS_OK;
}

ALS_RETCODE Ahc::PreClustering(SpeakerCluster *spk) {
  return ALS_OK;
}

ALS_RETCODE Ahc::PostClustering(SpeakerCluster *spk) {
  return ALS_OK;
}

ALS_RETCODE Ahc::Process(SpeakerCluster *new_spk) {

  PreClustering(new_spk);
  PostClustering(new_spk);
  return ALS_OK;
}
}


