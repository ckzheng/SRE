#ifndef _SPKVER_RESOURCE_MANAGER_H_
#define _SPKVER_RESOURCE_MANAGER_H_

#include <string>
#include "als_vad.h"
#include "config.h"
#include "full_gmm.h"
#include "diag_gmm.h"
#include "als_error.h"
#include "resource_loader.h"

class ResourceManager {
 public:
  static ResourceManager *Instance(const string &cfg_file,
                                   const string &sys_dir = "") {
    static ResourceManager res = ResourceManager(cfg_file, sys_dir);
    return &res;
  }

  const DiagGmm &DGmm() const {
    return dgmm_;
  }

  //const FullGmm &FGmm() const {
  //  return fgmm_;
  //}

  const UbmResource &UbmRes() const {
    return ubm_res_;
  }

  const IvectorResource &IvRes() const {
    return iv_res_;
  }

  const DoubleMatrix &LdaMatrix() const {
    return lda_matrix_;
  }

  const DoubleMatrix &CmvnMatrix() const {
    return cmvn_matrix_;
  }

  const PldaResource &PldaRes() const {
    return plda_res_;
  }

  const unsigned int GaussNum() const {
    return gauss_nums_;
  }

  const unsigned int FeatDim() const {
    return feat_dims_;
  }

  const unsigned int IvDim() const {
    return iv_dims_;
  }

  const string &ConfDir() const {
    return cfg_dir_;
  }

  const string &SysDir() const {
    return sys_dir_;
  }

  const IvectorExtractOptions &IvectorConfigOption() const {
    return sv_opt_;
  }

  const PldaOptions &PldaConfigOption() const {
    return plda_opt_;
  }

  const AlsVadMdlHandle &VadHandler() const {
    return vad_handle_;
  }

  ALS_RETCODE Destroy() {
    // AlsVad::UnLoadModel(vad_handle_);
    return ALS_OK;
  }

 private:
  void GenerateGmm(const ResourceLoader &res_loader) {
    if (verbose_mode_) {
      idec::IDEC_INFO << "Generate full gmm and diagonal gmm.";
    }
    ubm_res_ = res_loader.GetUbmResource();
    FullGmm fgmm = FullGmm(ubm_res_);
    dgmm_ = DiagGmm();
    dgmm_.CopyFromFullGmm(fgmm);
    fgmm.Clear();
  }

  void GenerateIvectorDerivedVars(const ResourceLoader &res_loader) {
    if (verbose_mode_) {
      idec::IDEC_INFO << "Computing derived variables for iVector extractor";
    }
    int gauss_num = res_loader.UbmMixture();
    int feat_dim = res_loader.FeatureDim();
    iv_res_ = res_loader.GetIvectorResource();
    iv_res_.gconsts.Resize(gauss_num);
    const double M_LOG_2PI = 1.8378770664093454835606594728112;
    double var_logdet;
    for (int i = 0; i < gauss_num; i++) {
      var_logdet = -iv_res_.sigma_inv[i].LogDet();
      iv_res_.gconsts(i) = -0.5 * (var_logdet + feat_dim * M_LOG_2PI);
    }

    const vector<DoubleMatrix> &M = iv_res_.t_matrix;
    const vector<DoubleMatrix> &Sigma_inv = iv_res_.sigma_inv;
    iv_res_.U.resize(gauss_num);
    iv_res_.Sigma_inv_M.resize(gauss_num);
    for (int i = 0; i < gauss_num; ++i) {
      iv_res_.U[i] = M[i].Transpose() * Sigma_inv[i] * M[i];
      iv_res_.Sigma_inv_M[i] = Sigma_inv[i] * M[i];
    }

    iv_res_.Sigma_inv_M_trans.resize(gauss_num);
    const vector<DoubleMatrix> &sigma_inv_m = iv_res_.Sigma_inv_M;
    for (int i = 0; i < gauss_num; ++i) {
      iv_res_.Sigma_inv_M_trans[i] = sigma_inv_m[i].Transpose();
    }
  }

  void GeneratePldaDerivedVars(const ResourceLoader &res_loader) {
    if (verbose_mode_) {
      idec::IDEC_INFO << "Computing derived variables for Plda scoring.";
    }

    idec::IDEC_ASSERT(res_loader.FeatureDim() > 0);
    plda_res_ = res_loader.GetPldaResource();
    plda_res_.offset.Resize(res_loader.FeatureDim());
    plda_res_.offset = plda_res_.transform * plda_res_.mean * (-1.0);
    IvectorResourceXnnMatrix();
  }

  void GenerateFeatureMeanVars(const ResourceLoader &res_loader) {
    cmvn_matrix_ = res_loader.GetCmvnResource();
    unsigned int cols = cmvn_matrix_.Cols();
    double frame_num = cmvn_matrix_(0, cols - 1);
    cmvn_matrix_.Scale(1 / frame_num);
  }

  void IvectorResourceXnnMatrix() {
    MatrixTransform::EigenVector2XnnMatrix(iv_res_.gconsts, iv_res_.gconsts_xx);
    MatrixTransform::EigenVector2XnnMatrix(iv_res_.mean, iv_res_.mean_xx);
    MatrixTransform::EigenVector2XnnMatrix(iv_res_.w_vec, iv_res_.w_vec_xx);
    iv_res_.sigma_inv_xx.resize(gauss_nums_);
    iv_res_.Sigma_inv_M_trans_xx.resize(gauss_nums_);
    iv_res_.U_xx.resize(gauss_nums_);
    iv_res_.t_matrix_xx.resize(gauss_nums_);
    for (int g = 0; g < gauss_nums_; ++g) {
      MatrixTransform::EigenMatrix2XnnMatrix(iv_res_.sigma_inv[g],
                                             iv_res_.sigma_inv_xx[g]);
      MatrixTransform::EigenMatrix2XnnMatrix(iv_res_.Sigma_inv_M[g],
                                             iv_res_.Sigma_inv_M_trans_xx[g]);
      MatrixTransform::EigenMatrix2XnnMatrix(iv_res_.U[g], iv_res_.U_xx[g]);
      MatrixTransform::EigenMatrix2XnnMatrix(iv_res_.t_matrix[g],
                                             iv_res_.t_matrix_xx[g]);
    }
  }

 private:
  ResourceManager(const string &cfg_file,
                  const string &sys_dir = ""):cfg_dir_(cfg_file), sys_dir_(sys_dir) {
    ReadConfigFile(sys_dir, cfg_file, sv_opt_, plda_opt_);
    res_loader_.LoadIvectorResource(sv_opt_.iv_mdl_path);
    res_loader_.loadUbmResource(sv_opt_.ubm_mdl_path);
    res_loader_.LoadPldaResource(sv_opt_.plda_mdl_path);
    res_loader_.LoadLdaMatrix(sv_opt_.lda_mdl_path);
    res_loader_.LoadFeatureMeanVars(sv_opt_.feat_cmvn_file_path);
    res_loader_.LoadIvectorMean(sv_opt_.ivector_mean_file_path);
    lda_matrix_ = res_loader_.GetLdaMatrix();
    feat_dims_ = res_loader_.FeatureDim();
    iv_dims_ = res_loader_.IvectorDim();
    gauss_nums_ = res_loader_.UbmMixture();

    GenerateGmm(res_loader_);
    GenerateFeatureMeanVars(res_loader_);
    GenerateIvectorDerivedVars(res_loader_);
    GeneratePldaDerivedVars(res_loader_);
  }

  ALS_RETCODE ReadConfigFile(const string &sys_dir, const string &cfg_file,
                             IvectorExtractOptions &sv_opt, PldaOptions &plda_opt) {
    ALS_RETCODE ret = ALS_OK;
    if (sys_dir == "") {
      idec::IDEC_WARNING << "The system directory is Empty.";
    }

    // init the environment
    idec::ParseOptions *po = new idec::ParseOptions("SpkVer");
    sv_opt.Register(po, "SpkVer.");
    if (!idec::File::IsExistence(cfg_file.c_str())) {
      idec::IDEC_ERROR << "configuration file " << cfg_file << " does not exist.";
      return ALSERR_FILE_NOT_FOUND;
    }

    po->ReadConfigFile(cfg_file);
    //sv_opt.FixPath(sys_dir.c_str());

    plda_opt.Register(po, "SpkVer.");
    po->ReadConfigFile(cfg_file);
    idec::IDEC_DELETE(po);
    verbose_mode_ = sv_opt.verbose_mode;
    if (sv_opt.do_vad) {
      if (!idec::File::IsExistence((sv_opt.vad_conf_path).c_str())) {
        idec::IDEC_ERROR << "vad conf " << sv_opt_.vad_conf_path <<
                         " does not exist";
        return ALSERR_FILE_NOT_FOUND;
      }

      if (!idec::File::IsExistence((sv_opt.vad_mdl_path).c_str())) {
        idec::IDEC_ERROR << "vad model " << sv_opt_.vad_mdl_path << "does not exist";
        return ALSERR_FILE_NOT_FOUND;
      }

      vad_handle_ = AlsVad::LoadModel(sv_opt.vad_conf_path.c_str(),
                                      sv_opt.vad_mdl_path.c_str());
    }
    return ALS_OK;
  }

 private:
  ResourceManager();
  ResourceManager(ResourceManager const &);
  void operator= (ResourceManager const &);

 private:
  unsigned int feat_dims_;
  unsigned int gauss_nums_;
  unsigned int iv_dims_;
  DiagGmm dgmm_;
  //FullGmm fgmm_;
  DoubleMatrix cmvn_matrix_;
  IvectorResource iv_res_;
  DoubleMatrix lda_matrix_;
  PldaResource plda_res_;
  UbmResource ubm_res_;
  AlsVadMdlHandle vad_handle_;
  IvectorExtractOptions sv_opt_;
  PldaOptions plda_opt_;
  ResourceLoader res_loader_;
  string sys_dir_;
  string cfg_dir_;
  bool verbose_mode_;
};

#endif // !_RESOURCE_MANAGER_H_
