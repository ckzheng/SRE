#include "am/xnn_net.h"
#include "vad/nn_vad.h"
#include "util/thread.h"
#include "base/log_message.h"
#include <memory>
#include <list>

// the system-wide management of vad resources
class AlsVadImpl {
 public:
  inline static AlsVadImpl *Instance() {
    idec::thread::lock_guard<idec::thread::recursive_mutex> lock(
      AlsVadImpl::mutex_);
    if (0 == instance_.get()) {
      instance_.reset(new AlsVadImpl);
    }
    return instance_.get();
  }

  AlsVad *Create(const char *cfg) {
    idec::thread::lock_guard<idec::thread::recursive_mutex> lock(
      AlsVadImpl::mutex_);
    if (default_xnn_net_ == NULL) {
      default_xnn_net_ = LoadNet(cfg, NULL);
    }
    idec::NNVad *new_vad = new idec::NNVad(cfg, default_xnn_net_);
    allocated_vad_.push_back(new_vad);
    return new_vad;
  }

  AlsVad *Create(const char *cfg, const char *model_path) {
    idec::thread::lock_guard<idec::thread::recursive_mutex> lock(
      AlsVadImpl::mutex_);
    if (default_xnn_net_ == NULL) {
      default_xnn_net_ = LoadNet(cfg, model_path);
    }
    idec::NNVad *new_vad = new idec::NNVad(cfg, default_xnn_net_);
    allocated_vad_.push_back(new_vad);
    return new_vad;
  }


  AlsVadMdlHandle LoadModel(const char *cfg, const char *model_path) {
    idec::thread::lock_guard<idec::thread::recursive_mutex> lock(
      AlsVadImpl::mutex_);
    idec::xnnNet *new_net = LoadNet(cfg, model_path);
    if (new_net != NULL) {
      alternative_xnn_nets_.push_back(VadModelInfo(new_net,std::string(cfg)));
    }
    return AlsVadMdlHandle(new_net);
  }

  AlsVadMdlHandle LoadModel(const char *cfg) {
    idec::thread::lock_guard<idec::thread::recursive_mutex> lock(
      AlsVadImpl::mutex_);
    idec::xnnNet *new_net = LoadNet(cfg, NULL);
    if (new_net != NULL) {
      alternative_xnn_nets_.push_back(VadModelInfo(new_net, std::string(cfg)));
    }
    return AlsVadMdlHandle(new_net);
  }


  void UnLoadModel(AlsVadMdlHandle h) {
    idec::thread::lock_guard<idec::thread::recursive_mutex> lock(
      AlsVadImpl::mutex_);
    std::list<VadModelInfo>::iterator it;
    for (it = alternative_xnn_nets_.begin(); it != alternative_xnn_nets_.end();
         ++it) {
      if ((AlsVadMdlHandle)it->xnn_net == h) {
        break;
      }
    }
    if (it != alternative_xnn_nets_.end()) {
      if (it->num_attached_vad != 0) {
        idec::IDEC_WARNING << "unload the vad model before delete its attached vad";
      }
      delete it->xnn_net;
      alternative_xnn_nets_.erase(it);
    }
  }



  AlsVad  *CreateFromModel(AlsVadMdlHandle h) {
    idec::thread::lock_guard<idec::thread::recursive_mutex> lock(
      AlsVadImpl::mutex_);
    std::list<VadModelInfo>::iterator it;
    for (it = alternative_xnn_nets_.begin(); it != alternative_xnn_nets_.end();
         ++it) {
      if ((AlsVadMdlHandle)it->xnn_net == h) {
        break;
      }
    }

    if (it == alternative_xnn_nets_.end()) {
      return NULL;
    }

    idec::NNVad *new_vad = new idec::NNVad(it->vad_cfg_file.c_str(), it->xnn_net);
    allocated_vad_.push_back(new_vad);
    it->num_attached_vad++;
    return new_vad;
  }

  void Delete(AlsVad *vad) {
    idec::thread::lock_guard<idec::thread::recursive_mutex> lock(
      AlsVadImpl::mutex_);
    if (vad == NULL) return;
    std::list<idec::NNVad *>::iterator it = std::find(allocated_vad_.begin(),
                                            allocated_vad_.end(), vad);
    if (it == allocated_vad_.end()) {
      return;
    }

    // change the reference counting, in the mdl info
    std::list<VadModelInfo>::iterator it_mdl;
    for (it_mdl = alternative_xnn_nets_.begin();
         it_mdl != alternative_xnn_nets_.end(); ++it_mdl) {
      if (it_mdl->xnn_net == (*it)->GetModel()) {
        it_mdl->num_attached_vad--;
        break;
      }
    }

    allocated_vad_.erase(it);
    delete vad;
  }

  void ReleaseResult(AlsVadResult *&result) {
    idec::NNVad::FreeApiOutputBuf(result);
  }

  AlsVadImpl() {
    default_xnn_net_ = NULL;
  }


  virtual ~AlsVadImpl() {
    for (std::list<idec::NNVad *>::iterator i = allocated_vad_.begin();
         i != allocated_vad_.end(); ++i) {
      delete *i;
    }
    allocated_vad_.resize(0);

    // delete the xnn net
    if (default_xnn_net_ != NULL) {
      delete default_xnn_net_;
    }
    default_xnn_net_ = NULL;

    // delete the vad
    for (std::list<VadModelInfo>::iterator i = alternative_xnn_nets_.begin();
         i != alternative_xnn_nets_.end(); ++i) {
      delete i->xnn_net;
    }
    alternative_xnn_nets_.resize(0);

  }

 private:
  idec::xnnNet  *LoadNet(const char *cfg, const char *model_path) {
    using namespace idec;
    if (NULL == cfg || strlen(cfg) == 0) {
      IDEC_ERROR << "Invalid cfg file! " << cfg << "not exit!\n";
      return NULL;
    }

    // load vad params
    VADXOptions vad_opt;
    ParseOptions po("vad params initialize");
    vad_opt.Register(&po);
    po.ReadConfigFile(cfg);

    // load the dnn
    if (model_path == NULL || *model_path == 0) {
      model_path = vad_opt.vad_model_path.c_str();
    } else {
      vad_opt.vad_model_path = model_path;
    }
    if (strlen(model_path) == 0 || *model_path == 0) {
      return NULL;
    }

    idec::xnnNet *new_xnn_net = NULL;

    if (vad_opt.vad_model_format == kKaldiNNet1String) {
      new_xnn_net = idec::xnnNet::LoadKaldiNnet1AndQuant(vad_opt.vad_model_path,
                    kQuant16bitString);
    } else {
      new_xnn_net = idec::xnnNet::LoadKaldiAndQuant(vad_opt.vad_model_path,
                    kQuant16bitString);
    }

    // change DNN behavior only for testing purpose, do not use those in real decoding
    typedef
    idec::xnnLogSoftmaxLayer<idec::xnnFloat16RuntimeMatrix, idec::xnnFloatRuntimeMatrix, idec::xnnFloat16RuntimeMatrix, idec::xnnFloatRuntimeMatrix>
    Q16XnnLogSoftMaxLayer;

    Q16XnnLogSoftMaxLayer *pLayer
      = reinterpret_cast<Q16XnnLogSoftMaxLayer *>(new_xnn_net->Layer(
            new_xnn_net->NumLayers() - 1));

    if (pLayer == NULL) {
      return NULL;
    }
    pLayer->useRealProb(true);

    if (!vad_opt.vad_model_use_prior) {
      pLayer->usePrior(false);
    }
    return new_xnn_net;
  }

  struct VadModelInfo {
    idec::xnnNet *xnn_net;
    std::string   vad_cfg_file;
    idec::uint32  num_attached_vad;
    VadModelInfo(idec::xnnNet *n = NULL, std::string c = "") :xnn_net(n),
      vad_cfg_file(c), num_attached_vad(0) {
    }
  };

 private:
  idec::xnnNet
  *default_xnn_net_;     // the default xnn model
  std::list <VadModelInfo>
  alternative_xnn_nets_; // other model loaded
  std::list<idec::NNVad *>              allocated_vad_;
  static std::auto_ptr<AlsVadImpl>     instance_;
  static idec::thread::recursive_mutex mutex_;
};
