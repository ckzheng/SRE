#ifndef _SPKVER_RESOURCE_LOADER_H_
#define _SPKVER_RESOURCE_LOADER_H_

#include <fstream>
#include <string>
#include <vector>
#include "new_matrix.h"
#include "new_vector.h"
#include "base/log_message.h"
#include "als_error.h"
#include "am/xnn_runtime.h"
#include "matrix_transform.h"

using namespace std;

struct UbmResource {
  DoubleVector g_const;
  DoubleVector weight;
  DoubleMatrix means_invcovars;
  vector<DoubleMatrix> inv_covars;
  idec::xnnFloatRuntimeMatrix g_const_xnn;
  idec::xnnFloatRuntimeMatrix weight_xnn;
  idec::xnnFloatRuntimeMatrix means_invcovars_xnn;
  vector<idec::xnnFloatRuntimeMatrix> inv_covars_xnn;
};

struct IvectorResource {
  double prior_offset;
  DoubleVector w_vec;
  DoubleVector mean;
  vector<DoubleMatrix> t_matrix;
  vector<DoubleMatrix> sigma_inv;
  vector<DoubleMatrix> Sigma_inv_M;
  vector<DoubleMatrix> Sigma_inv_M_trans;
  vector<DoubleMatrix> U;
  DoubleVector gconsts;
  idec::xnnFloatRuntimeMatrix w_vec_xx;
  idec::xnnFloatRuntimeMatrix mean_xx;
  vector<idec::xnnFloatRuntimeMatrix> t_matrix_xx;
  vector<idec::xnnFloatRuntimeMatrix> sigma_inv_xx;
  vector<idec::xnnFloatRuntimeMatrix> Sigma_inv_M_xx;
  vector<idec::xnnFloatRuntimeMatrix> Sigma_inv_M_trans_xx;
  vector<idec::xnnFloatRuntimeMatrix> U_xx;
  idec::xnnFloatRuntimeMatrix gconsts_xx;
  IvectorResource() {
    prior_offset = 0.0;
  }
};

struct PldaResource {
  DoubleVector mean;
  DoubleVector psi;
  DoubleVector offset;
  DoubleMatrix transform;
};

class ResourceLoader {
 public:
  ResourceLoader() {
    fdims_ = 0;
    vdims_ = 0;
    mixtures_ = 0;
  }

  ALS_RETCODE LoadVadArk(const string &file_name) {
    ifstream ifs(file_name.c_str(), ios::binary);
    if (!ifs) {
      idec::IDEC_ERROR << "load " << file_name << " error." << endl;
      ifs.close();
      return ALSERR_FILE_NOT_FOUND;
    }

    string label = "";
    ifs >> label;
    if (ifs.peek() != ' ') {
      idec::IDEC_ERROR << ": Expected token blank space, but not appear.";
    }
    cout << ifs.get();
    cout << ifs.tellg();
    ifs >> label;
    cout << ifs.get();
    int rows;
    ReadInt(ifs, rows);
    ReadFVector(ifs, ubm_res_.weight, rows);
  }

  ALS_RETCODE LoadIvectorResource(const string &file_name) {
    ifstream ifs(file_name.c_str(), ios::binary);
    if (!ifs) {
      idec::IDEC_ERROR << "load " << file_name << " error." << endl;
      ifs.close();
      return ALSERR_FILE_NOT_FOUND;
    }

    if (ifs.peek() != '\0')
      idec::IDEC_ERROR << "only support kaldi binary format";
    ifs.get();
    if (ifs.peek() != 'B')
      idec::IDEC_ERROR << "only support kaldi binary format";
    ifs.get();

    int rows;
    ReadLabel(ifs, "<IvectorExtractor>");
    ReadLabel(ifs, "<w>");
    ReadLabel(ifs, "DM");
    ReadInt(ifs, rows);
    ReadInt(ifs, rows);
    ReadLabel(ifs, "<w_vec>");
    ReadLabel(ifs, "DV");
    ReadInt(ifs, rows);
    ReadDVector(ifs, iv_res_.w_vec, rows);
    ReadLabel(ifs, "<M>");
    ReadInt(ifs, mixtures_);
    for (int i = 0; i < mixtures_; ++i) {
      DoubleMatrix tmp = DoubleMatrix();
      ReadLabel(ifs, "DM");
      ReadInt(ifs, fdims_);
      ReadInt(ifs, vdims_);
      ReadDMatrix(ifs, tmp, fdims_, vdims_);
      iv_res_.t_matrix.push_back(tmp);
    }

    ReadLabel(ifs, "<SigmaInv>");
    for (int i = 0; i < mixtures_; ++i) {
      ReadLabel(ifs, "DP");
      ReadInt(ifs, rows);
      DoubleMatrix v = DoubleMatrix();
      ReadDHalfMatrix(ifs, v, rows);
      iv_res_.sigma_inv.push_back(v);
    }

    ReadLabel(ifs, "<IvectorOffset>");
    ReadDouble(ifs, iv_res_.prior_offset);
    ReadLabel(ifs, "</IvectorExtractor>");
    ifs.close();
    return ALS_OK;
  }

  ALS_RETCODE LoadIvectorMean(const string &mean_path) {
    ifstream ifs(mean_path.c_str(), ios::binary);
    if (!ifs) {
      idec::IDEC_ERROR << "load " << mean_path << " error." << endl;
      ifs.close();
      return ALSERR_FILE_NOT_FOUND;
    }

    int cols;
    if (ifs.peek() != '\0')
      idec::IDEC_ERROR << "only support kaldi binary format";
    ifs.get();
    if (ifs.peek() != 'B')
      idec::IDEC_ERROR << "only support kaldi binary format";
    ifs.get();

    ReadLabel(ifs, "DV");
    ReadInt(ifs, cols);
    ReadDVector(ifs, iv_res_.mean, cols);
    return ALS_OK;
  }

  ALS_RETCODE loadUbmResource(const string &ubm_path) {
    ifstream ifs(ubm_path.c_str(), ios::binary);
    if (!ifs) {
      idec::IDEC_ERROR << "load " << ubm_path << " error." << endl;
      ifs.close();
      return ALSERR_FILE_NOT_FOUND;
    }

    if (ifs.peek() != '\0')
      idec::IDEC_ERROR << "only support kaldi binary format";
    ifs.get();
    if (ifs.peek() != 'B')
      idec::IDEC_ERROR << "only support kaldi binary format";
    ifs.get();

    int size, rows, cols;
    ReadLabel(ifs, "<FullGMM>");
    ReadLabel(ifs, "<GCONSTS>");
    ReadLabel(ifs, "FV");
    ReadInt(ifs, size);
    ReadFVector(ifs, ubm_res_.g_const, size);

    ReadLabel(ifs, "<WEIGHTS>");
    ReadLabel(ifs, "FV");
    ReadInt(ifs, size);
    ReadFVector(ifs, ubm_res_.weight, size);

    ReadLabel(ifs, "<MEANS_INVCOVARS>");
    ReadLabel(ifs, "FM");
    ReadInt(ifs, rows);
    ReadInt(ifs, cols);
    ReadFMatrix(ifs, ubm_res_.means_invcovars, rows, cols);

    ReadLabel(ifs, "<INV_COVARS>");
    mixtures_ = size;
    DoubleMatrix inv_covar = DoubleMatrix();
    for (int i = 0; i < mixtures_; ++i) {
      ReadLabel(ifs, "FP");
      ReadInt(ifs, rows);
      ReadFHalfMatrix(ifs, inv_covar, rows);
      ubm_res_.inv_covars.push_back(inv_covar);
    }

    MatrixTransform::EigenVector2XnnMatrix(ubm_res_.g_const, ubm_res_.g_const_xnn);
    MatrixTransform::EigenVector2XnnMatrix(ubm_res_.weight, ubm_res_.weight_xnn);
    MatrixTransform::EigenMatrix2XnnMatrix(const_cast<const DoubleMatrix &>
                                           (ubm_res_.means_invcovars).Transpose(), ubm_res_.means_invcovars_xnn);
    ubm_res_.inv_covars_xnn.resize(mixtures_);
    for (int i = 0; i < mixtures_; ++i) {
      MatrixTransform::EigenMatrix2XnnMatrix(ubm_res_.inv_covars[i],
                                             ubm_res_.inv_covars_xnn[i]);
    }

    return ALS_OK;
  }

  ALS_RETCODE LoadFeatureMeanVars(const string &mean_path) {
    ifstream ifs(mean_path.c_str(), ios::binary);
    if (!ifs) {
      idec::IDEC_ERROR << "load " << mean_path << " error." << endl;
      ifs.close();
      return ALSERR_FILE_NOT_FOUND;
    }

    int rows, cols;
    if (ifs.peek() != '\0')
      idec::IDEC_ERROR << "only support kaldi binary format";
    ifs.get();
    if (ifs.peek() != 'B')
      idec::IDEC_ERROR << "only support kaldi binary format";
    ifs.get();

    ReadLabel(ifs, "FM");
    ReadInt(ifs, rows);
    ReadInt(ifs, cols);
    ReadFMatrix(ifs, cmvn_res_, rows, cols);
    return ALS_OK;
  }

  void LoadLdaMatrix(const string tranform_matrix_path) {
    ifstream ifs(tranform_matrix_path.c_str(), ios::binary);
    if (!ifs) {
      idec::IDEC_ERROR << "load " << tranform_matrix_path << " error." << endl;
    }

    if (ifs.peek() != '\0')
      idec::IDEC_ERROR << "only support kaldi binary format";
    ifs.get();
    if (ifs.peek() != 'B')
      idec::IDEC_ERROR << "only support kaldi binary format";
    ifs.get();

    int cols, rows;
    ReadLabel(ifs, "FM");
    ReadInt(ifs, rows);
    ReadInt(ifs, cols);
    ReadFMatrix(ifs, lda_transform_, rows, cols);
  }

  ALS_RETCODE LoadPldaResource(const string &plda_path) {
    ifstream ifs(plda_path.c_str(), ios::binary);
    if (!ifs) {
      idec::IDEC_ERROR << "load " << plda_path << " error." << endl;
      ifs.close();
      return ALSERR_FILE_NOT_FOUND;
    }

    int size, cols, rows;
    if (ifs.peek() != '\0')
      idec::IDEC_ERROR << "only support kaldi binary format";
    ifs.get();
    if (ifs.peek() != 'B')
      idec::IDEC_ERROR << "only support kaldi binary format";
    ifs.get();

    ReadLabel(ifs, "<Plda>");
    ReadLabel(ifs, "DV");
    ReadInt(ifs, size);
    ReadDVector(ifs, plda_res_.mean, size);
    ReadLabel(ifs, "DM");
    ReadInt(ifs, rows);
    ReadInt(ifs, cols);
    ReadDMatrix(ifs, plda_res_.transform, rows, cols);
    ReadLabel(ifs, "DV");
    ReadInt(ifs, size);
    ReadDVector(ifs, plda_res_.psi, size);
    ReadLabel(ifs, "</Plda>");
    return ALS_OK;
  }

  int FeatureDim() const {
    return fdims_;
  }

  int IvectorDim() const {
    return vdims_;
  }

  int UbmMixture() const {
    return mixtures_;
  }

  const UbmResource &GetUbmResource() const {
    return ubm_res_;
  }

  const DoubleMatrix &GetCmvnResource() const {
    return cmvn_res_;
  }

  const IvectorResource &GetIvectorResource() const {
    return iv_res_;
  }

  const DoubleMatrix &GetLdaMatrix() const {
    return lda_transform_;
  }

  const PldaResource &GetPldaResource() const {
    return plda_res_;
  }

  bool ReadLabel(ifstream &ifs, const string &expect_label) {
    bool ret = true;
    string label = "";
    ifs >> label;
    if (label != expect_label) {
      ret = false;
      idec::IDEC_ERROR << ": Expected token " << label << ", got " << expect_label;
    }

    if (ifs.peek() != ' ') {
      idec::IDEC_ERROR << ": Expected token blank space, but not appear." ;
    }
    ifs.get();
    return ret;
  }

  bool ReadInt(ifstream &ifs, int &value) {
    if (ifs.peek() != 4) {
      idec::IDEC_ERROR << ": Expected token blank space, but not appear.";
    }
    ifs.get();
    ifs.read((char *)&value, sizeof(value));
    return true;
  }

  bool ReadDouble(ifstream &ifs, double &value) {
    if (ifs.peek() != 8) {
      idec::IDEC_ERROR << ": Expected token blank space, but not appear.";
    }
    ifs.get();
    ifs.read((char *)&value, sizeof(value));
    return true;
  }

  bool ReadDVector(ifstream &ifs, DoubleVector &vec, int num) {
    vec.Resize(num);
    ifs.read((char *)vec.Data(), num*sizeof(double));
    return true;
  }

  bool ReadFVector(ifstream &ifs, DoubleVector &vec, int num) {
    vector<float> fvec = vector<float>(num);
    vec.Resize(num);
    ifs.read((char *)&fvec[0], num*sizeof(float));
    for (int i = 0; i < num; ++i) {
      vec(i) = fvec[i];
    }
    return true;
  }

  bool ReadDMatrix(ifstream &ifs, DoubleMatrix &matrix, int rows, int cols) {
    matrix.Resize(rows, cols);
    DoubleVector v(cols);
    for (int i = 0; i < rows; ++i) {
      ifs.read((char *)v.Data(), cols * sizeof(double));
      matrix.Row(i, v);
    }
    return true;
  }

  bool ReadFMatrix(ifstream &ifs, DoubleMatrix &matrix, int rows, int cols) {
    int num = rows*cols;
    matrix.Resize(rows, cols);
    vector<float> fvec = vector<float>(num);
    ifs.read((char *)&fvec[0], num*sizeof(float));
    for (int i = 0, irow = 0, icol = 0; i < num; ++i) {
      irow = i / cols;
      icol = i % cols;
      matrix(irow, icol) = fvec[i];
    }
    return true;
  }

  bool ReadFHalfMatrix(ifstream &ifs, DoubleMatrix &matrix, int rows) {
    const int num = rows * (rows + 1) / 2;
    matrix.Resize(rows, rows);
    vector<float> fvec = vector<float>(num);
    ifs.read((char *)&fvec[0], num*sizeof(float));
    int base = 0;
    for (int j = 0; j < rows; ++j) {
      base = j*(j + 1) / 2;
      for (int k = 0; k <= j; ++k) {
        matrix(j, k) = fvec[base + k];
        if (k != j) {
          matrix(k, j) = matrix(j, k);
        }
      }
    }
    return true;
  }

  bool ReadDHalfMatrix(ifstream &ifs, DoubleMatrix &matrix, int rows) {
    const int num = rows * (rows + 1) / 2;
    matrix.Resize(rows, rows);
    vector<double> fvec = vector<double>(num);
    ifs.read((char *)&fvec[0], num*sizeof(double));
    int base = 0;
    for (int j = 0; j < rows; ++j) {
      base = j*(j + 1) / 2; // kaldi
      for (int k = 0; k <= j; ++k) {
        matrix(j, k) = fvec[base + k];
        if (k != j) {
          matrix(k, j) = matrix(j, k);
        }
      }
    }
    return true;
  }

 private:
  int fdims_;
  int vdims_;
  int mixtures_;
  DoubleMatrix cmvn_res_;
  UbmResource ubm_res_;
  IvectorResource iv_res_;
  DoubleMatrix lda_transform_;
  PldaResource plda_res_;
};


#endif // !_RESOURCE_LOADER_
