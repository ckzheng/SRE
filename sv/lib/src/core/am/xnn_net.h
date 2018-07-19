#ifndef XNN_NET_H_
#define XNN_NET_H_

#include "am/xnn_net/layer_base.h"
#include "am/xnn_net/add_shift_layer.h"
#include "am/xnn_net/block_softmax_layer.h"
#include "am/xnn_net/blstm_layer.h"
#include "am/xnn_net/convolutional_layer.h"
#include "am/xnn_net/embedding_layer.h"
#include "am/xnn_net/linear_layer.h"
#include "am/xnn_net/log_softmax_layer.h"
#include "am/xnn_net/lstm_layer.h"
#include "am/xnn_net/norm_layer.h"
#include "am/xnn_net/normalization_layer.h"
#include "am/xnn_net/max_pooling_layer.h"
#include "am/xnn_net/relu_layer.h"
#include "am/xnn_net/rescale_layer.h"
#include "am/xnn_net/sigmoid_layer.h"
#include "am/xnn_net/softmax_layer.h"
#include "am/xnn_net/project_blstm_layer.h"
#include "am/xnn_net/pnorm_layer.h"
#include "am/xnn_net/pure_relu.h"
#include "am/xnn_net/multi_convolutional_1d_layer.h"

namespace idec {
const std::string kKaldiNNet1String = "kaldi_nnet1";
const std::string kKaldiNNet2String = "kaldi_nnet2";
const std::string kQuant16bitString = "16bit";
class xnnNet {
 protected:
  std::vector<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *> layers_;

 public:
  ~xnnNet() {
    for (size_t i = 0; i<layers_.size(); ++i) {
      if (layers_[i]) delete layers_[i];
    }
  }

  // dimensions
  size_t vDim() const { return (!layers_.empty() ? (*layers_.begin())->vDim() : 0); }
  size_t uDim() const { return (!layers_.empty() ? (*layers_.rbegin())->uDim() : 0); }
  size_t NumLayers() const { return layers_.size(); }


  // access
  xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *Layer(size_t l) const {
    return(l<layers_.size() ? layers_[l] : NULL);
  }

  // load different model format
  static  xnnNet *LoadKaldiAndQuant(const std::string &fn,
                                    const std::string &quantization);
  static  xnnNet *LoadKaldiNnet1AndQuant(const std::string &fn,
                                         const std::string &quantization);
  void loadKaldi(const std::string &fn, int *quantize_bit);
  void loadKaldiNnet1(const std::string &fn, const std::string &fn_pu,
                      const std::string &fn_trans, int *quantize_bit, size_t nThread);
  void loadKaldiNnet1FromBuffer(std::stringstream &is);
  void loadNet(std::istream &is);
  void loadQuantNet(std::istream &is, int *quantize_bit);
  int  LoadCaffe(const char *fn, int *quantize_bit);
  void loadNetNnet1(std::istream &is, std::istream &is_pu,
                    std::istream &is_trans, size_t nThread);
  void loadNetNnet1Sim(std::istream &is, size_t nThread);
  void quantizeFloat16(const xnnNet &net);
  void quantizeFloat8(const xnnNet &net);

  // check
  void checkNetwork() {
    for (size_t i = 0; i + 1<layers_.size(); ++i) {
      layers_[i]->uDim() == layers_[i + 1]->vDim()
      || IDEC_ERROR << "dimension between layers mismatch [" << layers_[i]->uDim() <<
                    " vs. " << layers_[i + 1]->vDim();
      if (layers_[i]->getLayerType() != blstmLayer
          && layers_[i+1]->getLayerType() != blstmLayer) {
        if (layers_[i]->isBlockEvalSupported()
            && !layers_[i + 1]->isBlockEvalSupported()) {
          IDEC_ERROR << "block evaluation settings incorrect";
        }
      }
    }
  }

  void Serialize(SerializeHelper &helper) {
    uint32 size = static_cast<uint32>(layers_.size());
    helper.Serialize(size);
    for (size_t i = 0; i < layers_.size(); ++i) {
      XnnLayerType layerType = layers_[i]->getLayerType();
      XnnMatrixType matrixType = layers_[i]->getMatrixType();

      helper.Serialize(&layerType, sizeof(layerType));
      helper.Serialize(&matrixType, sizeof(matrixType));
      layers_[i]->Serialize(helper);
    }
  }

  void Deserialize(SerializeHelper &helper) {
    layers_.clear();
    uint32 L;
    helper.Deserialize(L);
    for (uint32 l = 0; l < L; ++l) {
      XnnLayerType layerType;
      XnnMatrixType matrixType;
      helper.Deserialize(&layerType, sizeof(layerType));
      helper.Deserialize(&matrixType, sizeof(matrixType));

      switch (layerType) {
      case linearLayer:
        if (matrixType == FloatFloat) {
          XnnLinearLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer = new
          XnnLinearLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();
          pLayer->Deserialize(helper);
          layers_.push_back(
            reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>
            (pLayer));
          break;
        } else if (matrixType == ShortFloat) {
          XnnLinearLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer = new
          XnnLinearLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>();
          pLayer->Deserialize(helper);
          layers_.push_back(
            reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>
            (pLayer));
          break;
        } else if (matrixType == CharFloat) {
          XnnLinearLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer = new
          XnnLinearLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix>();
          pLayer->Deserialize(helper);
          layers_.push_back(
            reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>
            (pLayer));
          break;
        }

      case reluLayer:
        if (matrixType == FloatFloat) {
          xnnReLULayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer = new
          xnnReLULayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();
          pLayer->Deserialize(helper);
          layers_.push_back(
            reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>
            (pLayer));
          break;
        } else if (matrixType == ShortFloat) {
          xnnReLULayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer = new
          xnnReLULayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>();
          pLayer->Deserialize(helper);
          layers_.push_back(
            reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>
            (pLayer));
          break;
        } else if (matrixType == CharFloat) {
          xnnReLULayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer = new
          xnnReLULayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix>();
          pLayer->Deserialize(helper);
          layers_.push_back(
            reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>
            (pLayer));
          break;
        }


      case sigmoidLayer:
        if (matrixType == FloatFloat) {
          xnnSigmoidLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer = new
          xnnSigmoidLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();
          pLayer->Deserialize(helper);
          layers_.push_back(
            reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>
            (pLayer));
          break;
        } else if (matrixType == ShortFloat) {
          xnnSigmoidLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer = new
          xnnSigmoidLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>();
          pLayer->Deserialize(helper);
          layers_.push_back(
            reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>
            (pLayer));
          break;
        } else if (matrixType == CharFloat) {
          xnnSigmoidLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer = new
          xnnSigmoidLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix>();
          pLayer->Deserialize(helper);
          layers_.push_back(
            reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>
            (pLayer));
          break;
        }
      case embeddingLayer:
        if (matrixType == FloatFloat) {
          xnnEmbeddingLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer = new
          xnnEmbeddingLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();
          pLayer->Deserialize(helper);
          layers_.push_back(
            reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>
            (pLayer));
          break;
        } else if (matrixType == ShortFloat) {
          IDEC_ERROR << "layer type" << layerType << "matrix type" << matrixType <<
                     "unsupported yet";
          break;
        } else if (matrixType == CharFloat) {
          IDEC_ERROR << "layer type" << layerType << "matrix type" << matrixType <<
                     "unsupported yet";
          break;
        }
      case lstmLayer:
        if (matrixType == FloatFloat) {
          xnnLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer = new
          xnnLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix,xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();
          pLayer->Deserialize(helper);
          layers_.push_back(
            reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>
            (pLayer));
          break;
        } else if (matrixType == ShortFloat) {
          IDEC_ERROR << "layer type" << layerType << "matrix type" << matrixType <<
                     "unsupported yet";
          break;
        } else if (matrixType == CharFloat) {
          IDEC_ERROR << "layer type" << layerType << "matrix type" << matrixType <<
                     "unsupported yet";
          break;
        }
      case softmaxLayer:
        if (matrixType == FloatFloat) {
          XnnSoftmaxLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer = new
          XnnSoftmaxLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix,xnnFloatRuntimeMatrix>();
          pLayer->Deserialize(helper);
          layers_.push_back(
            reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>
            (pLayer));
          break;
        } else if (matrixType == ShortFloat) {
          IDEC_ERROR << "layer type" << layerType << "matrix type" << matrixType <<
                     "unsupported yet";
          break;
        } else if (matrixType == CharFloat) {
          IDEC_ERROR << "layer type" << layerType << "matrix type" << matrixType <<
                     "unsupported yet";
          break;
        }

      case logsoftmaxLayer:
        if (matrixType == FloatFloat) {
          xnnLogSoftmaxLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer = new
          xnnLogSoftmaxLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();
          pLayer->Deserialize(helper);
          layers_.push_back(
            reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>
            (pLayer));
          break;
        } else if (matrixType == ShortFloat) {
          xnnLogSoftmaxLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer = new
          xnnLogSoftmaxLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>();
          pLayer->Deserialize(helper);
          layers_.push_back(
            reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>
            (pLayer));
          break;
        } else if (matrixType == CharFloat) {
          xnnLogSoftmaxLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer = new
          xnnLogSoftmaxLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix>();
          pLayer->Deserialize(helper);
          layers_.push_back(
            reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>
            (pLayer));
          break;
        }
      case normalizationLayer:
        if (matrixType == FloatFloat) {
          xnnNormalizationLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
            new xnnNormalizationLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();
          pLayer->Deserialize(helper);
          layers_.push_back(
            reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>
            (pLayer));
          break;
        } else if (matrixType == ShortFloat) {
          IDEC_ERROR << "layer type"<< layerType <<"matrix type" << matrixType <<
                     "unsupported yet";
          break;
        } else if (matrixType == CharFloat) {
          IDEC_ERROR << "layer type" << layerType << "matrix type" << matrixType <<
                     "unsupported yet";
          break;
        }
      case blstmLayer:
        if (matrixType == FloatFloat) {
          xnnBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer = new
          xnnBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix,xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();
          pLayer->Deserialize(helper);
          layers_.push_back(
            reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>
            (pLayer));
          break;
        } else if (matrixType == ShortFloat) {
          xnnBLSTMLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer = new
          xnnBLSTMLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>();
          pLayer->Deserialize(helper);
          layers_.push_back(
            reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>
            (pLayer));
          break;
        } else if (matrixType == CharFloat) {
          xnnBLSTMLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer = new
          xnnBLSTMLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix>();
          pLayer->Deserialize(helper);
          layers_.push_back(
            reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>
            (pLayer));
          break;
        }
      case projectedblstmLayer:
        if (matrixType == FloatFloat) {
          xnnProjectedBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer = new
          xnnProjectedBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();
          pLayer->Deserialize(helper);
          layers_.push_back(
            reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>
            (pLayer));
          break;
        } else if (matrixType == ShortFloat) {
          xnnProjectedBLSTMLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer = new
          xnnProjectedBLSTMLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>();
          pLayer->Deserialize(helper);
          layers_.push_back(
            reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>
            (pLayer));
          break;
        } else if (matrixType == CharFloat) {
          xnnProjectedBLSTMLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer = new
          xnnProjectedBLSTMLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix>();
          pLayer->Deserialize(helper);
          layers_.push_back(
            reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>
            (pLayer));
          break;
        }
      case purereluLayer:
        if (matrixType == FloatFloat) {
          xnnPureReLULayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer = new
          xnnPureReLULayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();
          pLayer->Deserialize(helper);
          layers_.push_back(
            reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>
            (pLayer));
          break;
        } else if (matrixType == ShortFloat) {
          IDEC_ERROR << "layer type" << layerType << "matrix type" << matrixType <<
                     "unsupported yet";
          break;
        } else if (matrixType == CharFloat) {
          IDEC_ERROR << "layer type" << layerType << "matrix type" << matrixType <<
                     "unsupported yet";
          break;
        }
      case convolutionalLayer:
        if (matrixType == FloatFloat) {
          xnnConvolutionalLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer = new
          xnnConvolutionalLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();
          pLayer->Deserialize(helper);
          layers_.push_back(
            reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>
            (pLayer));
          break;
        } else if (matrixType == ShortFloat) {
          xnnConvolutionalLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer = new
          xnnConvolutionalLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();
          pLayer->Deserialize(helper);
          layers_.push_back(
            reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>
            (pLayer));
          break;
          //IDEC_ERROR << "layer type" << layerType << "matrix type" << matrixType << "unsupported yet";
          //break;
        } else if (matrixType == CharFloat) {
          xnnConvolutionalLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer = new
          xnnConvolutionalLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();
          pLayer->Deserialize(helper);
          layers_.push_back(
            reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>
            (pLayer));
          break;
        }
      case maxpoolingLayer:
        if (matrixType == FloatFloat) {
          xnnMaxpoolingLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer = new xnnMaxpoolingLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();
          pLayer->Deserialize(helper);
          layers_.push_back(
            reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>
            (pLayer));
          break;
        } else if (matrixType == ShortFloat) {
          IDEC_ERROR << "layer type" << layerType << "matrix type" << matrixType <<
                     "unsupported yet";
          break;
        } else if (matrixType == CharFloat) {
          IDEC_ERROR << "layer type" << layerType << "matrix type" << matrixType <<
                     "unsupported yet";
          break;
        }
      case rescaleLayer:
        if (matrixType == FloatFloat) {
          xnnRescaleLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer = new
          xnnRescaleLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();
          pLayer->Deserialize(helper);
          layers_.push_back(
            reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>
            (pLayer));
          break;
        } else if (matrixType == ShortFloat) {
          IDEC_ERROR << "layer type" << layerType << "matrix type" << matrixType <<
                     "unsupported yet";
          break;
        } else if (matrixType == CharFloat) {
          IDEC_ERROR << "layer type" << layerType << "matrix type" << matrixType <<
                     "unsupported yet";
          break;
        }
      case addshiftLayer:
        if (matrixType == FloatFloat) {
          xnnAddShiftLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer = new
          xnnAddShiftLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();
          pLayer->Deserialize(helper);
          layers_.push_back(
            reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>
            (pLayer));
          break;
        } else if (matrixType == ShortFloat) {
          IDEC_ERROR << "layer type" << layerType << "matrix type" << matrixType <<
                     "unsupported yet";
          break;
        }
      case multiconvolution1dLayer:
        if (matrixType == FloatFloat) {
          xnnMultiConvolutional1DLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer = new
          xnnMultiConvolutional1DLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();
          pLayer->Deserialize(helper);
          layers_.push_back(
            reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>
            (pLayer));
          break;
        } else if (matrixType == ShortFloat) {
          IDEC_ERROR << "layer type" << layerType << "matrix type" << matrixType <<
                     "unsupported yet";
          break;
        } else if (matrixType == CharFloat) {
          IDEC_ERROR << "layer type" << layerType << "matrix type" << matrixType <<
                     "unsupported yet";
          break;
        }
      default:
        IDEC_ERROR << "unknown layer type " << layerType << " or matrix type " <<
                   matrixType;
      }
    }
  }


};

class xnnAmEvaluator {
 protected:
  const xnnNet                       &net_;
  xnnFloatRuntimeMatrix              feat_;
  // the activation of each layer
  std::vector<xnnFloatRuntimeMatrix> activations_;
  // the intermediate states used by each layer (for fully-connected DNN this should be empty, for LSTM this could be used to store memory states
  std::vector< std::vector< void * > > intermediate_states_;
  std::vector< std::vector< void * > > intermediate_states_store_;
  size_t defaultBlocksize_;
  int pos_;
  float acscale_;
  size_t threadId_;
  size_t window_size_;
  size_t window_shift_;

  int startFrame_;
  const float LZERO;// = -1.0e10f;

  xnnFloat16RuntimeMatrix activations_float16_;
  xnnFloat8RuntimeMatrix activations_float8_;

 public:

  xnnAmEvaluator(const xnnNet &net, float acscale = 1.0f, size_t block_size = 0,
                 size_t threadId = 0, size_t window_size = 0, size_t window_shift = 0)
    : net_(net), pos_(-1), acscale_(acscale), threadId_(threadId), startFrame_(0),
      LZERO(-1.0e10f), window_size_(window_size), window_shift_(window_shift) {

    defaultBlocksize_ = (block_size == 0) ? kAmEvaluatorDefaultBlockSize :
                        block_size;

    // prepare activations
    activations_.resize(net_.NumLayers());
    size_t max_width = 0;

    for (size_t i = 0; i < activations_.size(); ++i) {
      max_width = std::max(max_width, net_.Layer(i)->uDim());
      activations_[i].Resize(net_.Layer(i)->uDim(),
                             net_.Layer(i)->isBlockEvalSupported() ? defaultBlocksize_ : window_size_);
    }

    activations_float16_.Resize(max_width, defaultBlocksize_);

    // prepare intermediate states
    intermediate_states_.resize(net_.NumLayers());
    intermediate_states_store_.resize(net_.NumLayers());
    for (size_t i = 0; i < net_.NumLayers(); i++) {
      net_.Layer(i)->InitIntermediateStates(intermediate_states_[i]);
      net_.Layer(i)->InitIntermediateStates(intermediate_states_store_[i]);
    }

    // only influence the lstm
    setWindowSizeForBlstm(window_size_, window_shift_);
  }

  ~xnnAmEvaluator() {
    for (size_t i = 0; i < intermediate_states_.size(); i++) {
      net_.Layer(i)->DeleteIntermediateStates(intermediate_states_[i]);
      net_.Layer(i)->DeleteIntermediateStates(intermediate_states_store_[i]);
    }
  }

  std::vector<xnnFloatRuntimeMatrix> getActivations() {
    return activations_;
  }

  void useThread(size_t threadId) { threadId_ = threadId; };

  float  logLikelihood(int fr, int st);
  float  logLikelihood_lazy(int fr, int st);
  float  *logLikelihood(int fr);


  void pushFeatures(int startFrame,
                    const xnnFloatRuntimeMatrix
                    &feat/*, int contextExpLeft = 0, int contextExpRight = 0*/) {
    startFrame_ = startFrame;
    feat_ = feat;
    pos_ = -1;
  }

  void reset() {
    startFrame_ = 0;
    feat_.Clear();
    pos_ = -1;
    resetLstmState();
  }

  void resetLstmState() {
    for (size_t i = 0; i < net_.NumLayers(); ++i) {
      if (net_.Layer(i)->getLayerType() == blstmLayer) {
        if (net_.Layer(i)->getMatrixType() == FloatFloat) {
          xnnBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer
            = reinterpret_cast<xnnBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *>
              (net_.Layer(i));
          pLayer->ResetStateBuffer(intermediate_states_[i]);
        } else if (net_.Layer(i)->getMatrixType() == ShortFloat) {
          xnnBLSTMLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer
            = reinterpret_cast<xnnBLSTMLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix> *>
              (net_.Layer(i));
          pLayer->ResetStateBuffer(intermediate_states_[i]);
        } else if (net_.Layer(i)->getMatrixType() == CharFloat) {
          xnnBLSTMLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer
            = reinterpret_cast<xnnBLSTMLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix> *>
              (net_.Layer(i));
          pLayer->ResetStateBuffer(intermediate_states_[i]);
        }
      }
      if (net_.Layer(i)->getLayerType() == projectedblstmLayer) {
        if (net_.Layer(i)->getMatrixType() == FloatFloat) {
          xnnProjectedBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer
            = reinterpret_cast<xnnProjectedBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *>
              (net_.Layer(i));
          pLayer->ResetStateBuffer(intermediate_states_[i]);
        } else if (net_.Layer(i)->getMatrixType() == ShortFloat) {
          xnnProjectedBLSTMLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer
            = reinterpret_cast<xnnProjectedBLSTMLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix> *>
              (net_.Layer(i));
          pLayer->ResetStateBuffer(intermediate_states_[i]);
        } else if (net_.Layer(i)->getMatrixType() == CharFloat) {
          xnnProjectedBLSTMLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer
            = reinterpret_cast<xnnProjectedBLSTMLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix> *>
              (net_.Layer(i));
          pLayer->ResetStateBuffer(intermediate_states_[i]);
        }
      }
    }
  }

  void storeLstmState() {
    for (size_t i = 0; i < net_.NumLayers(); ++i) {
      if (net_.Layer(i)->getLayerType() == blstmLayer) {
        if (net_.Layer(i)->getMatrixType() == FloatFloat) {
          xnnBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer
            = reinterpret_cast<xnnBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *>
              (net_.Layer(i));
          pLayer->CopyStateBuffer(intermediate_states_[i],
                                  intermediate_states_store_[i]);
        } else if (net_.Layer(i)->getMatrixType() == ShortFloat) {
          xnnBLSTMLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer
            = reinterpret_cast<xnnBLSTMLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix> *>
              (net_.Layer(i));
          pLayer->CopyStateBuffer(intermediate_states_[i],
                                  intermediate_states_store_[i]);
        }
      }
      if (net_.Layer(i)->getLayerType() == projectedblstmLayer) {
        if (net_.Layer(i)->getMatrixType() == FloatFloat) {
          xnnProjectedBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer
            = reinterpret_cast<xnnProjectedBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *>
              (net_.Layer(i));
          pLayer->CopyStateBuffer(intermediate_states_[i],
                                  intermediate_states_store_[i]);
        } else if (net_.Layer(i)->getMatrixType() == ShortFloat) {
          xnnProjectedBLSTMLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer
            = reinterpret_cast<xnnProjectedBLSTMLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix> *>
              (net_.Layer(i));
          pLayer->CopyStateBuffer(intermediate_states_[i],
                                  intermediate_states_store_[i]);
        }
      }
    }
  }

  void restoreLstmState() {
    for (size_t i = 0; i < net_.NumLayers(); ++i) {
      if (net_.Layer(i)->getLayerType() == blstmLayer) {
        if (net_.Layer(i)->getMatrixType() == FloatFloat) {
          xnnBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer
            = reinterpret_cast<xnnBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *>
              (net_.Layer(i));
          pLayer->CopyStateBuffer(intermediate_states_store_[i],
                                  intermediate_states_[i]);
        } else if (net_.Layer(i)->getMatrixType() == ShortFloat) {
          xnnBLSTMLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer
            = reinterpret_cast<xnnBLSTMLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix> *>
              (net_.Layer(i));
          pLayer->CopyStateBuffer(intermediate_states_store_[i],
                                  intermediate_states_[i]);
        }
      }
      if (net_.Layer(i)->getLayerType() == projectedblstmLayer) {
        if (net_.Layer(i)->getMatrixType() == FloatFloat) {
          xnnProjectedBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer
            = reinterpret_cast<xnnProjectedBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *>
              (net_.Layer(i));
          pLayer->CopyStateBuffer(intermediate_states_store_[i],
                                  intermediate_states_[i]);
        } else if (net_.Layer(i)->getMatrixType() == ShortFloat) {
          xnnProjectedBLSTMLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer
            = reinterpret_cast<xnnProjectedBLSTMLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix> *>
              (net_.Layer(i));
          pLayer->CopyStateBuffer(intermediate_states_store_[i],
                                  intermediate_states_[i]);
        }
      }
    }
  }

  void setWindowSizeForBlstm(size_t window_size, size_t window_shift) {
    window_size_ = window_size;
    window_shift_ = window_shift;
    for (size_t i = 0; i < net_.NumLayers(); ++i) {
      if (net_.Layer(i)->getLayerType() == blstmLayer) {
        if (net_.Layer(i)->getMatrixType() == FloatFloat) {
          xnnBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer
            = reinterpret_cast<xnnBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *>
              (net_.Layer(i));
          pLayer->SetWindowSize(window_size_);
          pLayer->SetWindowShift(window_shift_);
        } else if (net_.Layer(i)->getMatrixType() == ShortFloat) {
          xnnBLSTMLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer
            = reinterpret_cast<xnnBLSTMLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix> *>
              (net_.Layer(i));
          pLayer->SetWindowSize(window_size_);
          pLayer->SetWindowShift(window_shift_);
        } else if (net_.Layer(i)->getMatrixType() == CharFloat) {
          xnnBLSTMLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer
            = reinterpret_cast<xnnBLSTMLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix> *>
              (net_.Layer(i));
          pLayer->SetWindowSize(window_size_);
          pLayer->SetWindowShift(window_shift_);
        }
      } else if (net_.Layer(i)->getLayerType() == projectedblstmLayer) {
        if (net_.Layer(i)->getMatrixType() == FloatFloat) {
          xnnProjectedBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer
            = reinterpret_cast<xnnProjectedBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *>
              (net_.Layer(i));
          pLayer->SetWindowSize(window_size_);
          pLayer->SetWindowShift(window_shift_);
        } else if (net_.Layer(i)->getMatrixType() == ShortFloat) {
          xnnProjectedBLSTMLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer
            = reinterpret_cast<xnnProjectedBLSTMLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix> *>
              (net_.Layer(i));
          pLayer->SetWindowSize(window_size_);
          pLayer->SetWindowShift(window_shift_);
        } else if (net_.Layer(i)->getMatrixType() == CharFloat) {
          xnnProjectedBLSTMLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix>
          *pLayer
            = reinterpret_cast<xnnProjectedBLSTMLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix> *>
              (net_.Layer(i));
          pLayer->SetWindowSize(window_size_);
          pLayer->SetWindowShift(window_shift_);
        }
      }
    }
  }
};

};



#endif
