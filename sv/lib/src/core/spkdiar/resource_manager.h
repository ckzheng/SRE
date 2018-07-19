#ifndef _RESOURCE_MANAGER_H_
#define _RESOURCE_MANAGER_H_

#include <string>
#include <memory>
#include "full_gmm.h"
#include "diag_gmm.h"
#include "als_error.h"
#include "als_vad.h"
#include "config.h"
#include "hmm/gmm.h"
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

  const FullGmm &FGmm() const {
    return fgmm_;
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

  const FrontEndOptions &FrontendConfigOption() const {
    return frontend_opt_;
  }

  const BicOptions &BicConfigOption() const {
    return bic_opt_;
  }

  const AhcOptions &AhcConfigOptions() const {
    return ahc_opt_;
  }

  const IvectorExtractOptions &IvectorConfigOption() const {
    return sv_opt_;
  }

  const PldaOptions &PldaConfigOption() const {
    return plda_opt_;
  }

  const HmmOptions &HmmConfigOption() const {
    return hmm_opt_;
  }

  const alsid::MapUpdateOption &GmmUpdateOption() const {
    return gmm_update_opt_;
  }

  const AlsVadMdlHandle &VadHandler() const {
    return vad_handle_;
  }

  const string& StrInputType() const {
	return str_input_type_;
  }

  const alsid::GMM &UbmRes() const {
    return ubm_;
  }

  bool VerboseMode() const {
    return verbose_mode_;
  }

  int Destroy() {
    //AlsVad::UnLoadModel(vad_handle_);
    return ALS_OK;
  }

 private:
  void GenerateGmm(const ResourceLoader &res_loader) {
    if (verbose_mode_) {
      idec::IDEC_INFO << "Generate full gmm and diagonal gmm.";
    }
    fgmm_ = FullGmm(res_loader.GetUbmResource());
    dgmm_ = DiagGmm();
    dgmm_.CopyFromFullGmm(fgmm_);
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
  }

  void GenerateFeatureMeanVars(const ResourceLoader &res_loader) {
    cmvn_matrix_ = res_loader.GetCmvnResource();
    unsigned int cols = cmvn_matrix_.Cols();
    double frame_num = cmvn_matrix_(0, cols - 1);
    cmvn_matrix_.Scale(1 / frame_num);
  }

  int InitUniMdl(const std::string &uni_mdl) {
    if (!idec::File::IsReadable(uni_mdl.c_str())) {
      idec::IDEC_WARNING << "cannot read si model" << uni_mdl;
      return ALSERR_FILE_NOT_FOUND;
    }

    IDEC_RETCODE ret;
    ret = ubm_.Deserialize(hmm_opt_.ubm_mdl_path.c_str());
    if (ALS_OK != ret) {
      return ALSERR_UNHANDLED_EXCEPTION;
    }
    return ALS_OK;
  }

 private:
  ResourceManager(const string &cfg_file,
                  const string &sys_dir = "") : cfg_dir_(cfg_file), sys_dir_(sys_dir) {
    ReadConfigFile(sys_dir, cfg_file);
    ResourceLoader res_loader;
    res_loader.LoadIvectorResource(sv_opt_.iv_mdl_path);
    res_loader.loadUbmResource(sv_opt_.gmm_mdl_path);
    res_loader.LoadPldaResource(sv_opt_.plda_mdl_path);
    res_loader.LoadLdaMatrix(sv_opt_.lda_mdl_path);
    res_loader.LoadFeatureMeanVars(frontend_opt_.feat_cmvn_file_path);
    res_loader.LoadIvectorMean(sv_opt_.ivector_mean_file_path);
    lda_matrix_ = res_loader.GetLdaMatrix();
    feat_dims_ = res_loader.FeatureDim();
    iv_dims_ = res_loader.IvectorDim();
    gauss_nums_ = res_loader.UbmMixture();

    InitUniMdl(hmm_opt_.ubm_mdl_path);

    GenerateGmm(res_loader);
    GenerateFeatureMeanVars(res_loader);
    GenerateIvectorDerivedVars(res_loader);
    GeneratePldaDerivedVars(res_loader);
  }

  int ReadConfigFile(const string &sys_dir, const string &cfg_file) {
    int ret = ALS_OK;
    if (sys_dir == "") {
      idec::IDEC_WARNING << "The system directory is Empty.";
    }

    if (!idec::File::IsExistence(cfg_file.c_str())) {
      idec::IDEC_ERROR << "configuration file " << cfg_file << " does not exist.";
      return ALSERR_FILE_NOT_FOUND;
    }

    idec::ParseOptions *po = new idec::ParseOptions("SpkVer");
    frontend_opt_.Register(po, "SpkDiar.");
    po->ReadConfigFile(cfg_file);

    bic_opt_.Register(po, "SpkDiar.");
    po->ReadConfigFile(cfg_file);

    ahc_opt_.Register(po, "SpkDiar.");
    po->ReadConfigFile(cfg_file);

    sv_opt_.Register(po, "SpkVer.");
    po->ReadConfigFile(cfg_file);

    plda_opt_.Register(po, "SpkVer.");
    po->ReadConfigFile(cfg_file);

    gmm_update_opt_.Register(po, "SpkDiar.");
    po->ReadConfigFile(cfg_file);

    hmm_opt_.Register(po, "SpkDiar.");
    po->ReadConfigFile(cfg_file);

    idec::IDEC_DELETE(po);
	str_input_type_ = frontend_opt_.str_input_type;
    bic_opt_.verbose_mode = frontend_opt_.verbose_mode;
    ahc_opt_.verbose_mode = frontend_opt_.verbose_mode;
    sv_opt_.verbose_mode = frontend_opt_.verbose_mode;
    plda_opt_.verbose_mode = frontend_opt_.verbose_mode;
    hmm_opt_.verbose_mode = frontend_opt_.verbose_mode;
    verbose_mode_ = sv_opt_.verbose_mode;

    if (frontend_opt_.do_vad) {
      if (!idec::File::IsExistence((frontend_opt_.vad_conf_path).c_str())) {
        idec::IDEC_ERROR << "vad conf " << frontend_opt_.vad_conf_path <<
                         " does not exist";
        return ALSERR_FILE_NOT_FOUND;
      }

      if (!idec::File::IsExistence((frontend_opt_.vad_mdl_path).c_str())) {
        idec::IDEC_ERROR << "vad model " << frontend_opt_.vad_mdl_path <<
                         "does not exist";
        return ALSERR_FILE_NOT_FOUND;
      }

      vad_handle_ = AlsVad::LoadModel(frontend_opt_.vad_conf_path.c_str(),
                                      frontend_opt_.vad_mdl_path.c_str());
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
  alsid::GMM ubm_;
  DiagGmm dgmm_;
  FullGmm fgmm_;
  DoubleMatrix cmvn_matrix_;
  IvectorResource iv_res_;
  DoubleMatrix lda_matrix_;
  PldaResource plda_res_;
  AlsVadMdlHandle vad_handle_;
  FrontEndOptions frontend_opt_;
  BicOptions bic_opt_;
  AhcOptions ahc_opt_;
  IvectorExtractOptions sv_opt_;
  PldaOptions plda_opt_;
  alsid::MapUpdateOption gmm_update_opt_;
  HmmOptions hmm_opt_;
  string sys_dir_;
  string cfg_dir_;
  string str_input_type_;
  bool verbose_mode_;
};

#endif // !_RESOURCE_MANAGER_H_
