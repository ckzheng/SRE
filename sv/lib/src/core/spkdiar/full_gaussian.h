#ifndef _FULL_GAUSSIAN_H_
#define _FULL_GAUSSIAN_H_

#include <vector>
#include "als_error.h"
#include "base/log_message.h"

namespace alspkdiar {

using std::vector;

class  FullGaussian {
 public:
  FullGaussian() {
    is_det_ = false;
    dim_ = 0;
    frames_ = 0;
    det_ = 0;
  }

  ~FullGaussian() {
    Clear();
  }

  unsigned int GetDim() const {
    return this->dim_;
  }

  void SetDim(unsigned int dim) {
    this->dim_ = dim;
  }

  void SetFrames(unsigned int frames) {
    this->frames_ = frames;
  }

  unsigned int GetFrames() const {
    return this->frames_;
  }

  const double *GetMean() const {
    return &mean_[0];
  }

  void SetMean(const vector<double> &mean) {
    this->mean_ = mean;
  }

  const double *GetVar() const {
    return &var_[0];
  }

  void SetVar(const vector<double> &var) {
    this->var_ = var;
  }

  void SetDet(float det) {
    this->det_ = det;
    is_det_ = true;
  }

  int Train(float **feat, unsigned int dim,
            unsigned int begin,
            unsigned int end) {
    this->dim_ = dim;
    this->frames_ = end - begin;
    mean_.resize(dim);
    var_.reserve(dim * dim);
    for (unsigned int i = 0; i < dim * dim; ++i) {
      var_[i] = 0;
    }

    for (unsigned int j = 0; j < dim_; ++j) {
      mean_[j] = 0.0;
      for (unsigned int i = begin; i < end; ++i) {
        mean_[j] += feat[i][j];
      }

      mean_[j] /= frames_;
    }

    for (unsigned int i = 0; i < dim_; ++i) {
      for (unsigned int j = i; j < dim_; j++) {
        for (unsigned int k = begin; k < end; ++k) {
          var_[i* dim_ + j] += (feat[k][i] - mean_[i]) * (feat[k][j] - mean_[j]);
        }

        var_[i* dim_ + j] /= frames_ - 1;
        if (i != j) {
          var_[j * dim + i] = var_[i* dim_ + j];
        }
      }
    }

    return ALS_OK;
  }


  int Train(vector<float *> &feat, unsigned int dim,
            unsigned int begin,
            unsigned int end) {
    this->dim_ = dim;
    this->frames_ = end - begin;
    mean_.resize(dim);
    var_.resize(dim * dim);
    for (unsigned int i = 0; i < dim * dim; ++i) {
      var_[i] = 0;
    }

    for (unsigned int j = 0; j < dim_; ++j) {
      mean_[j] = 0.0;
      for (unsigned int i = begin; i < end; ++i) {
        mean_[j] += feat[i][j];
      }

      mean_[j] /= frames_;
    }

    for (unsigned int i = 0; i < dim_; ++i) {
      for (unsigned int j = i; j < dim_; j++) {
        for (unsigned int k = begin; k < end; ++k) {
          var_[i* dim_ + j] += (feat[k][i] - mean_[i]) * (feat[k][j] - mean_[j]);
        }

        var_[i* dim_ + j] /= frames_ - 1;
        if (i != j) {
          var_[j * dim + i] = var_[i* dim_ + j];
        }
      }
    }

    return ALS_OK;
  }

  int Merge(const FullGaussian *full_gaussian) {
    double ratio = (float) this->frames_ / (this->frames_ +
                                            full_gaussian->GetFrames());
    const double *m = full_gaussian->GetMean();
    const double *v = full_gaussian->GetVar();
    float r = 1 - ratio;
    float Exy, ExEy;

    //covar = E[xy] - ExEy
    for (unsigned int i = 0; i < dim_; ++i) {
      for (unsigned int j = i; j < dim_; j++) {
        Exy = (var_[i*dim_ + j] + mean_[i] * mean_[j]) * ratio +
              (v[i*dim_ + j] + m[i] * m[j]) * r;
        ExEy = (ratio * mean_[i] + r * m[i]) * (ratio * mean_[j] + r * m[j]);
        var_[i*dim_ + j] = Exy - ExEy;
        if (i != j) {
          var_[j*dim_ + i] = Exy - ExEy;
        }
      }
      mean_[i] = mean_[i] * ratio + m[i] * r;
    }

    this->frames_ += full_gaussian->frames_;
    // need recalculate |¡Æ|
    is_det_ = false;

    return ALS_OK;
  }

  int CalLogDet(const double *mat, float *log_det,
                const int dim) {
    int i = 0, j = 0, k = 0, u = 0, l = 0;
    double d = 0.0f;
    std::vector<double> vec = std::vector<double>(dim*dim);
    for (int i = 0; i < dim*dim; ++i) {
      vec[i] = mat[i];
    }

    if (vec[0] < 1.0e-6) {
      idec::IDEC_ERROR << "Matrix is not positive definite.";
      return 0;
    }

    vec[0] = sqrt(vec[0]);
    d = vec[0];
    for (i = 1; i < dim; i++) {
      u = dim *i;
      vec[u] /= vec[0];
    }

    for (j = 1; j < dim; j++) {
      l = j* dim + j;
      for (k = 0; k < j; k++) {
        u = j * dim + k;
        vec[l] -= vec[u] * vec[u];
      }

      if (vec[l] < 1.0e-6) {
        idec::IDEC_ERROR << "Matrix is not positive definite.";
        return 0;
      }

      vec[l] = sqrt(vec[l]);
      d *= vec[l];
      for (i = j + 1; i < dim; ++i) {
        u = i * dim + j;
        for (k = 0; k < j; k++) {
          vec[u] -= vec[i* dim + k] * vec[j*dim + k];
        }
        vec[u] /= vec[l];
      }
    }

    *log_det = (float)log(d);
    return ALS_OK;
  }

  float GetLDet() {
    if (!is_det_) {
      is_det_ = true;
      //det_ = UpperCholesky();
      CalLogDet(&var_[0], &det_, dim_);
    }
    return det_;
  }

  FullGaussian *Clone() const {
    FullGaussian *full_gaussian = new FullGaussian();
    vector<double> mean(dim_);
    for (unsigned int i = 0; i < dim_; ++i) {
      mean[i] = mean_[i];
    }

    vector<double> var (dim_ * dim_);
    for (unsigned int i = 0; i < dim_ * dim_; ++i) {
      var[i] = var_[i];
    }

    full_gaussian->SetDim(dim_);
    full_gaussian->SetDet(det_);
    full_gaussian->SetFrames(frames_);
    full_gaussian->SetMean(mean);
    full_gaussian->SetVar(var);
    return full_gaussian;
  }

  bool IsEmpty() const {
    return (mean_.empty() && (dim_ == 0));
  }

  void Clear() {
    mean_.clear();
    var_.clear();
    dim_ = 0;
    is_det_ = false;
  }

 private:
  vector<double> mean_;
  vector<double> var_;
  unsigned int dim_;
  unsigned int frames_;
  bool is_det_;
  float det_;
};
}

#endif
