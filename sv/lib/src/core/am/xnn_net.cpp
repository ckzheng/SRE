#include "base/log_message.h"
#include "base/time_utils.h"
#include "am/xnn_net.h"

//#define  TEST_C_STYLE_BLSTM_FPROP 1
#ifdef TEST_C_STYLE_BLSTM_FPROP
#include "am/blstm_forward.h"
#endif

namespace idec {
  xnnNet*xnnNet::LoadKaldiAndQuant(const std::string &fn, const std::string &quantization) {
    int32 quant_bit = 0;
    xnnNet  *net = new xnnNet();
    net->loadKaldi(fn.c_str(), &quant_bit);

    // online quantization
    if (quantization == "16bit" &&  quant_bit == 32) {
      xnnNet * q16_net = new xnnNet();
      q16_net->quantizeFloat16(*net);
      delete net;
      net = q16_net;
    }
    else if (quantization == "8bit" &&  quant_bit == 32) {
      xnnNet * q8_net = new xnnNet();
      q8_net->quantizeFloat8(*net);
      delete net;
      net = q8_net;
    }
    return net;
  }

  xnnNet* xnnNet::LoadKaldiNnet1AndQuant(const std::string &mdl_fn_prefix, const std::string &quantization) {
    xnnNet  *net = new xnnNet();
    std::string netf = mdl_fn_prefix + ".net";
    std::string priorf = mdl_fn_prefix + ".prior";
    std::string mvnf = mdl_fn_prefix + ".mvn";
    net->loadKaldiNnet1(netf, priorf, mvnf, NULL, 1);

    // online quantization
    if (quantization == "16bit") {
      xnnNet * q16_net = new xnnNet();
      q16_net->quantizeFloat16(*net);
      delete net;
      net = q16_net;
    }
    else if (quantization == "8bit") {
      xnnNet * q8_net = new xnnNet();
      q8_net->quantizeFloat8(*net);
      delete net;
      net = q8_net;
    }
    return net;
  }


  void xnnNet::loadKaldi(const std::string& fn, int *quan_bit) {
    using namespace xnnKaldiUtility;

    std::ifstream is;
    bool binary = true;

    is.open(fn.c_str(), binary ? std::ios::binary | std::ios::in : std::ios::in);
    if (!is.is_open())
      IDEC_ERROR << "error opening " << fn;

    // make sure the input file is binary
    if (is.peek() != '\0')
      IDEC_ERROR << "only support kaldi binary format";
    is.get();
    if (is.peek() != 'B')
      IDEC_ERROR << "only support kaldi binary format";
    is.get();

    // skip transitional models
    std::string tok;
    do {
      ReadToken(is, binary, &tok);
    } while (tok != "<Nnet>" && tok != "<QuantNnet>");

    if (tok == "<Nnet>") {
      if (quan_bit != NULL) {
        *quan_bit = 32;
      }
      loadNet(is);
    }
    else if (tok == "<QuantNnet>") {
      loadQuantNet(is, quan_bit);
    }
    else {
      IDEC_ERROR << "invalid dnn model";
    }

    is.close();
    checkNetwork();
  }

  void xnnNet::loadKaldiNnet1(const std::string& fn_nnet, const std::string& fn_prior, const std::string& fn_xform, int *quan_bit, size_t nThread) {
    using namespace xnnKaldiUtility;

    std::ifstream is_nnet;
    std::ifstream is_pu;
    std::ifstream is_xform;
    bool binary = true;

    is_xform.open(fn_xform.c_str(), binary ? std::ios::binary | std::ios::in : std::ios::in);
    if (!is_xform.is_open())
      IDEC_ERROR << "error opening transform file " << fn_xform;

    std::string tok_trans;
    ReadToken(is_xform, binary, &tok_trans);
    if (tok_trans != "<Nnet>")
      IDEC_ERROR << "only support kaldi nnet1 transform format";

    is_nnet.open(fn_nnet.c_str(), binary ? std::ios::binary | std::ios::in : std::ios::in);
    if (!is_nnet.is_open())
      IDEC_ERROR << "error opening " << fn_nnet;

    // make sure the input file is binary
    if (is_nnet.peek() != '\0')
      IDEC_ERROR << "only support kaldi binary format";
    is_nnet.get();
    if (is_nnet.peek() != 'B')
      IDEC_ERROR << "only support kaldi binary format";
    is_nnet.get();

    is_pu.open(fn_prior.c_str(), binary ? std::ios::binary | std::ios::in : std::ios::in);
    if (!is_pu.is_open())
      IDEC_ERROR << "error opening " << fn_prior;

    if (is_pu.peek() != '[')
      IDEC_ERROR << "only support kaldi nnet1 prior format";
    is_pu.get();

    // skip transitional models
    std::string tok;
    do {
      ReadToken(is_nnet, binary, &tok);
    } while (tok != "<Nnet>" && tok != "<QuantNnet>");

    if (tok == "<Nnet>") {
      if (quan_bit != NULL) {
        *quan_bit = 32;
      }
      loadNetNnet1(is_nnet, is_pu, is_xform, nThread);
    }
    else {
      IDEC_ERROR << "invalid dnn model";
    }

    is_nnet.close();
    checkNetwork();
  }

  void xnnNet::loadKaldiNnet1FromBuffer(std::stringstream& is) {
    using namespace xnnKaldiUtility;
    if (is.peek() != '\0')
      IDEC_ERROR << "only support kaldi binary format";
    is.get();
    if (is.peek() != 'B')
      IDEC_ERROR << "only support kaldi binary format";
    is.get();
    std::string tok;
    do {
      ReadToken(is, 1, &tok);
    } while (tok != "<Nnet>" && tok != "<QuantNnet>");

    if (tok == "<Nnet>") {
      loadNetNnet1Sim(is, 1);
    }
    else {
      IDEC_ERROR << "invalid dnn model";
    }
    checkNetwork();
  }

  int xnnNet::LoadCaffe(const char* fn, int *quantize_bit) {
    if (quantize_bit != NULL) {
      *quantize_bit = 32;
    }

    std::ifstream is;
    bool binary = false;

    std::string token;
    is.open(fn, binary ? std::ios::binary | std::ios::in : std::ios::in);

    std::vector<float> v;
    v.reserve(0x100000); // 1M

    int vdim = 0, udim = 0;
    enum {
      weight_r,
      bias_r,
    } progress;

    progress = weight_r;

    XnnLinearLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *temp_layer
      = new XnnLinearLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();
    while (is.good()) {
      std::string word;
      std::string gab;
      is >> word;

      if (word == "data:") {
        float f;
        is >> f;
        v.push_back(f);
      }
      else if (word == "blobs") {
        v.clear();
      }
      else if (word == "shape") {
        std::getline(is, gab);
        if (progress == weight_r) {
          is >> word >> udim;
          is >> word >> vdim;
          assert(word == "dim:");
          progress = bias_r;
          temp_layer->Assign_Wm(v, udim);
        }
        else if (progress == bias_r) {
          int dim;
          is >> word >> dim;
          assert(word == "dim:");
          assert(dim == udim);
          progress = weight_r;
          temp_layer->Assign_bm(v);
        }
      }
      else if (word == "type:") {
        std::string type;
        is >> type;
        if (type == "\"SoftmaxWithLoss\"") {
          xnnLogSoftmaxLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer
            = new xnnLogSoftmaxLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>(
            *reinterpret_cast<XnnLinearLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *>(temp_layer));

          pLayer->usePrior(false);
          pLayer->useRealProb(true);
          delete temp_layer;
          temp_layer = new XnnLinearLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();
          layers_.push_back(NULL);
          layers_[layers_.size() - 1] = reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer);

          progress = weight_r;
        }
        else if (type == "\"ReLU\"") {
          xnnReLULayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
            new xnnReLULayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>(
            *reinterpret_cast<XnnLinearLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *>(temp_layer));

          delete temp_layer;
          temp_layer = new XnnLinearLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();

          layers_.push_back(NULL);
          layers_[layers_.size() - 1] = reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer);

          progress = weight_r;
        }
      }
      else {
        std::getline(is, gab);
      }
    }

    delete temp_layer;
    return 0;
  }

  void xnnNet::loadNet(std::istream &is) {
    using namespace xnnKaldiUtility;
    bool binary = true;
    // read net
    //ExpectToken(is, binary, "<Nnet>");
    int32 num_components;
    ExpectToken(is, binary, "<NumComponents>");
    ReadBasicType(is, binary, &num_components);
    ExpectToken(is, binary, "<Components>");

    // read each component
    std::string token;
    std::string tok;
    for (int32 c = 0; c < num_components; ++c) {
      ReadToken(is, binary, &token); // e.g. "<SigmoidComponent>".
      token.erase(0, 1); // erase "<".
      token.erase(token.length() - 1); // erase ">".

      if (token == "SpliceComponent") {
        ReadSpliceComponent(is, binary);
      }
      else if (token == "FixedAffineComponent" || token == "AffineComponentPreconditionedOnline" || token == "AffineComponent") {
        XnnLinearLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer
          = new XnnLinearLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();

        std::string startTok = "<" + token + ">";
        std::string endTok = "</" + token + ">";

        if (token == "AffineComponentPreconditionedOnline" || token == "AffineComponent") {
          SkipHead_AffineComponentPreconditionedOnline(is, startTok, binary);
        }

        ExpectOneOrTwoTokens(is, binary, startTok.c_str(), "<LinearParams>");
        pLayer->ReadKaldiLayerNnet2(is);

        if (token == "AffineComponentPreconditionedOnline") {
          SkipTail_AffineComponentPreconditionedOnline(is, binary);
        }
        else if (token == "AffineComponent") {
          SkipTail_AffineComponent(is, binary);
        }

        ExpectToken(is, binary, endTok.c_str());

        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (token == "PnormComponent") {
        xnnPnormLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer = new xnnPnormLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();
        pLayer->readKaldiLayer(is);
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (token == "NormalizeComponent") {
        xnnNormLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer = new xnnNormLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();
        pLayer->readKaldiLayer(is);
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (token == "SoftmaxComponent") {
        // only need to skip the component
        do {
          ReadToken(is, binary, &tok);
          //std::cout << tok << std::endl;
        } while (tok.find("</SoftmaxComponent>") == std::string::npos);
      }
      else if (token == "RectifiedLinearComponent") {
        xnnReLULayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new xnnReLULayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>(
          *reinterpret_cast<XnnLinearLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *>(*layers_.rbegin()));

        delete *layers_.rbegin();
        layers_[layers_.size() - 1] = reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer);

        // only need to skip the component
        do {
          ReadToken(is, binary, &tok);
          //std::cout << tok << std::endl;
        } while (tok.find("</RectifiedLinearComponent>") == std::string::npos);
      }
      else if (token == "SigmoidComponent") {
        xnnSigmoidLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new xnnSigmoidLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>(
          *reinterpret_cast<XnnLinearLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *>(*layers_.rbegin()));

        delete *layers_.rbegin();
        layers_[layers_.size() - 1] = reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer);

        // only need to skip the component
        do {
          ReadToken(is, binary, &tok);
          //std::cout << tok << std::endl;
        } while (tok.find("</SigmoidComponent>") == std::string::npos);
      }
      else {
        IDEC_ERROR << "Unknown token " << token;
      }
    }

    ExpectToken(is, binary, "</Components>");
    ExpectToken(is, binary, "</Nnet>");

    // convert last layer into a Softmax layer
    if (layers_.empty())
      IDEC_ERROR << "no layers read";
    if (token != "SoftmaxComponent" || (*layers_.rbegin())->getLayerType() != linearLayer)
      IDEC_ERROR << "last layer is not softmax, or penultimate layer is not linear";

    // create softmax layer by copying the last linear layer's W_ and b_
    xnnLogSoftmaxLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer
      = new xnnLogSoftmaxLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>(
      *reinterpret_cast<XnnLinearLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *>(*layers_.rbegin()));

    // read priors
    is.peek();
    if (!is.eof())
      pLayer->readKaldiPu(is);

    // replace last layer with the newly created softmax
    delete *layers_.rbegin();
    layers_[layers_.size() - 1] = reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer);

  }

  void xnnNet::loadNetNnet1(std::istream & is, std::istream &is_pu, std::istream &is_trans, size_t nThread) {
    using namespace xnnKaldiUtility;
    bool binary = true;

    int32 dim_out, dim_in;
    std::string token;
    std::string tok;
    std::string fore_token;


    is_trans.peek();
    if (!is_trans.eof()) {
      xnnNormalizationLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer
        = new xnnNormalizationLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();

      pLayer->readKaldiLayerNnet1(is_trans);
      pLayer->enableBlockEval(false);
      layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
    }


    int next_char = Peek(is, binary);
    if (next_char == EOF) return;


    while (EOF != next_char) {
      ReadToken(is, binary, &tok);
      if (tok == "</Nnet>") break;
      token = tok;
      token.erase(0, 1); // erase "<".
      token.erase(token.length() - 1); // erase ">".

      ReadBasicType(is, binary, &dim_out);
      ReadBasicType(is, binary, &dim_in);
      if (token == "AffineTransform") {
        XnnLinearLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer
          = new XnnLinearLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();

        pLayer->ReadKaldiLayerNnet1(is);
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));

      }
      else if (token == "Softmax") {

      }
      else if (token == "BlockSoftmax") {
        xnnBlockSoftmaxLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer
          = new xnnBlockSoftmaxLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>
          (*reinterpret_cast<XnnLinearLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *>(*layers_.rbegin()));

        pLayer->ReadData(is, binary);
        delete *layers_.rbegin();
        layers_[layers_.size() - 1] = reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer);
      }
      else if (token == "RectifiedLinear") {

        if (fore_token != "AffineTransform") {
          xnnPureReLULayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
            new xnnPureReLULayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();

          pLayer->setvDim(dim_in);
          pLayer->setuDim(dim_out);
          pLayer->enableBlockEval(false);
          layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
        }
        else {
          xnnReLULayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
            new xnnReLULayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>(
            *reinterpret_cast<XnnLinearLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *>(*layers_.rbegin()));


          delete *layers_.rbegin();
          layers_[layers_.size() - 1] = reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer);
        }
      }
      else if (token == "Sigmoid") {
        xnnSigmoidLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new xnnSigmoidLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>(
          *reinterpret_cast<XnnLinearLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *>(*layers_.rbegin()));

        delete *layers_.rbegin();
        layers_[layers_.size() - 1] = reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer);
      }
      else if (token == "LcCscBLstmStreams" || token == "CscBLstmStreams" || token == "BLstmStreams") {
        xnnBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer
          = new xnnBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();

        pLayer->readKaldiLayerNnet1(is/*, nThread*/);
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (token == "LcCscBLstmStreamsFA") {
        xnnBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer
          = new xnnBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();

        pLayer->readKaldiLayerNnet1(is/*, nThread*/);
        pLayer->setForwardAppro(true);
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (token == "LcCscBLstmProjectedStreams" || token == "CscBLstmProjectedStreams" || token == "ProjectedBLstmStreams") {
        xnnProjectedBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer
          = new xnnProjectedBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();

        pLayer->readKaldiLayerNnet1(is/*, nThread*/);
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (token == "LstmStreams") {
        xnnBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer
          = new xnnBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();

        pLayer->setBidirectional(false);
        pLayer->readKaldiLayerNnet1(is/*, nThread*/);
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (token == "ConvolutionalComponent") {
        xnnConvolutionalLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new xnnConvolutionalLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();

        pLayer->readKaldiLayerNnet1(is/*, nThread*/);
        pLayer->setvDim(dim_in);
        pLayer->setuDim(dim_out);
        pLayer->enableBlockEval(false);
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (token == "MaxPoolingComponent") {
        xnnMaxpoolingLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new xnnMaxpoolingLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();

        pLayer->readKaldiLayerNnet1(is/*, nThread*/);
        pLayer->setvDim(dim_in);
        pLayer->setuDim(dim_out);
        pLayer->enableBlockEval(false);
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (token == "Rescale") {
        xnnRescaleLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new xnnRescaleLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();

        pLayer->readKaldiLayerNnet1(is/*, nThread*/);
        pLayer->enableBlockEval(false);
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (token == "AddShift") {
        xnnAddShiftLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new xnnAddShiftLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();

        pLayer->readKaldiLayerNnet1(is/*, nThread*/);
        pLayer->enableBlockEval(false);
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else {
        IDEC_ERROR << "Unknown token " << token;
      }

      fore_token = token;
    }

    // convert last layer into a Softmax layer
    if (layers_.empty())
      IDEC_ERROR << "no layers read";
    if (token == "Softmax" && (*layers_.rbegin())->getLayerType() != linearLayer)
      IDEC_ERROR << "penultimate layer is not linear:" << token;
    if (token != "Softmax" && token != "BlockSoftmax")
      IDEC_ERROR << "last layer is not softmax or block softmax";

    if (token != "BlockSoftmax") {
      // create softmax layer by copying the last linear layer's W_ and b_
      xnnLogSoftmaxLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer
        = new xnnLogSoftmaxLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>(
        *reinterpret_cast<XnnLinearLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *>(*layers_.rbegin()));

      // read priors
      is_pu.peek();
      if (!is_pu.eof())
        pLayer->readKaldiNnet1Pu(is_pu);

      // replace last layer with the newly created softmax
      delete *layers_.rbegin();
      layers_[layers_.size() - 1] = reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer);
    }
  }

  void xnnNet::loadNetNnet1Sim(std::istream & is, size_t nThread) {
    using namespace xnnKaldiUtility;
    bool binary = true;

    int32 dim_out, dim_in;
    std::string token;
    std::string tok;

    int next_char = Peek(is, binary);
    if (next_char == EOF) return;

    while (EOF != next_char) {
      ReadToken(is, binary, &tok);
      if (tok == "</Nnet>") break;
      token = tok;
      token.erase(0, 1); // erase "<".
      token.erase(token.length() - 1); // erase ">".

      ReadBasicType(is, binary, &dim_out);
      ReadBasicType(is, binary, &dim_in);
      if (token == "AffineTransform") {
        XnnLinearLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer
          = new XnnLinearLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();

        pLayer->ReadKaldiLayerNnet1(is);
        pLayer->enableBlockEval(false);
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (token == "Softmax") {
        XnnSoftmaxLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer
          = new XnnSoftmaxLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>(
          *reinterpret_cast<XnnLinearLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *>(*layers_.rbegin()));

        // replace last layer with the newly created softmax
        delete *layers_.rbegin();
        layers_[layers_.size() - 1] = reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer);
        //pLayer->useRealProb(true);
      }
      else if (token == "BlockSoftmax") {
        xnnBlockSoftmaxLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer
          = new xnnBlockSoftmaxLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>(
          *reinterpret_cast<XnnLinearLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *>(*layers_.rbegin()));

        pLayer->ReadData(is, binary);
        // replace last layer with the newly created softmax
        delete *layers_.rbegin();
        layers_[layers_.size() - 1] = reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer);
        //pLayer->useRealProb(true);
      }
      else if (token == "RectifiedLinear") {
        xnnReLULayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new xnnReLULayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>(
          *reinterpret_cast<XnnLinearLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *>(*layers_.rbegin()));

        delete *layers_.rbegin();
        layers_[layers_.size() - 1] = reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer);
        layers_[layers_.size() - 1]->enableBlockEval(false);
      }
      else if (token == "Sigmoid") {
        xnnSigmoidLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new xnnSigmoidLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>(
          *reinterpret_cast<XnnLinearLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *>(*layers_.rbegin()));

        delete *layers_.rbegin();
        layers_[layers_.size() - 1] = reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer);
        layers_[layers_.size() - 1]->enableBlockEval(false);
      }
      else if (token == "LcCscBLstmStreams" || token == "CscBLstmStreams" || token == "BLstmStreams") {
        xnnBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer
          = new xnnBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();

        pLayer->readKaldiLayerNnet1(is/*, nThread*/);
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (token == "LcCscBLstmProjectedStreams" || token == "CscBLstmProjectedStreams" || token == "ProjectedBLstmStreams") {
        xnnProjectedBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer
          = new xnnProjectedBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();

        pLayer->readKaldiLayerNnet1(is/*, nThread*/);
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (token == "LstmStreams") {
        xnnBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer
          = new xnnBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();
        pLayer->setBidirectional(false);
        pLayer->readKaldiLayerNnet1(is/*, nThread*/);
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (token == "ConvolutionalComponent") {
        xnnConvolutionalLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new xnnConvolutionalLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();

        pLayer->readKaldiLayerNnet1(is/*, nThread*/);
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (token == "MultiConvolution1d") {
        xnnMultiConvolutional1DLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer
          = new xnnMultiConvolutional1DLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();

        pLayer->readKaldiLayerNnet1(is);
        pLayer->enableBlockEval(false);
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else {
        IDEC_ERROR << "Unknown token " << token;
      }
    }

    // convert last layer into a Softmax layer or BlockSoftmax
    if (layers_.empty())
      IDEC_ERROR << "no layers read";
    //auto it = layers_.rbegin();
    //it++;
    if ((token != "Softmax" && token != "BlockSoftmax"))
      IDEC_ERROR << "last layer is not softmax, or penultimate layer is not linear";
  }

  void xnnNet::loadQuantNet(std::istream &is, int *quantize_bit) {
    bool binary = true;

    using namespace xnnKaldiUtility;
    // read quantized net
    //ExpectToken(is, binary, "<QuantNnet>");
    int32 q_bit;
    ReadBasicType(is, binary, &q_bit);
    if (quantize_bit != NULL) {
      *quantize_bit = q_bit;
    }

    // note that the peak memory will be duplicated in
    // SerializeHelper and xnnNet
    idec::SerializeHelper helper_read(1024);
    helper_read.Read(is);
    this->Deserialize(helper_read);
    ExpectToken(is, binary, "</QuantNnet>");
  }



  template<> XnnLinearLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>::XnnLinearLayer(const XnnLinearLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> &layer) {
    supportBlockEval_ = layer.supportBlockEval_;

    W_.quantize(layer.W_);
    b_ = layer.b_;
  }

  template<> XnnLinearLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix>::XnnLinearLayer(const XnnLinearLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> &layer) {
    supportBlockEval_ = layer.supportBlockEval_;

    W_.quantize(layer.W_, 0, 0.0f);
    b_ = layer.b_;
  }

  template<> xnnLogSoftmaxLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>::xnnLogSoftmaxLayer(const xnnLogSoftmaxLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> &layer) {
    supportBlockEval_ = layer.supportBlockEval_;

    W_.quantize(layer.W_);
    b_ = layer.b_;
    prior_ = layer.prior_;

    use_prior_ = layer.use_prior_;
    use_real_prob_ = layer.use_real_prob_;
  }

  template<> xnnLogSoftmaxLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix>::xnnLogSoftmaxLayer(const xnnLogSoftmaxLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> &layer) {
    supportBlockEval_ = layer.supportBlockEval_;

    W_.quantize(layer.W_, 0, 0.0f);
    b_ = layer.b_;
    prior_ = layer.prior_;

    use_prior_ = layer.use_prior_;
    use_real_prob_ = layer.use_real_prob_;
  }

  template<> xnnBLSTMLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>::xnnBLSTMLayer(const xnnBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> &layer) {
    supportBlockEval_ = layer.supportBlockEval_;

    Wfw_.quantize(layer.Wfw_);
    Rfw_.quantize(layer.Rfw_);
    bfw_ = layer.bfw_;
    pfw_ = layer.pfw_;

    isBidirectional_ = layer.isBidirectional_;
    isForwardAppro_ = layer.isForwardAppro_;
    if (isBidirectional_) {
      Wbw_.quantize(layer.Wbw_);
      Rbw_.quantize(layer.Rbw_);
      bbw_ = layer.bbw_;
      pbw_ = layer.pbw_;
    }
    wstride_ = layer.wstride_;
    window_size_ = layer.window_size_;
    window_shift_ = layer.window_shift_;
  }

  template<> xnnBLSTMLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix>::xnnBLSTMLayer(const xnnBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> &layer) {
    supportBlockEval_ = layer.supportBlockEval_;

    Wfw_.quantize(layer.Wfw_, 0, 0.0f);
    Rfw_.quantize(layer.Rfw_, 0, 0.0f);
    bfw_ = layer.bfw_;
    pfw_ = layer.pfw_;

    isBidirectional_ = layer.isBidirectional_;
    isForwardAppro_ = layer.isForwardAppro_;
    if (isBidirectional_) {
      Wbw_.quantize(layer.Wbw_, 0, 0.0f);
      Rbw_.quantize(layer.Rbw_, 0, 0.0f);
      bbw_ = layer.bbw_;
      pbw_ = layer.pbw_;
    }
    wstride_ = layer.wstride_;
    window_size_ = layer.window_size_;
    window_shift_ = layer.window_shift_;
  }

  template<> xnnProjectedBLSTMLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>::xnnProjectedBLSTMLayer(const xnnProjectedBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> &layer) {
    supportBlockEval_ = layer.supportBlockEval_;

    Wfw_.quantize(layer.Wfw_);
    Rfw_.quantize(layer.Rfw_);
    Mfw_.quantize(layer.Mfw_);
    bfw_ = layer.bfw_;
    pfw_ = layer.pfw_;

    isBidirectional_ = layer.isBidirectional_;
    if (isBidirectional_) {
      Wbw_.quantize(layer.Wbw_);
      Rbw_.quantize(layer.Rbw_);
      Mbw_.quantize(layer.Mbw_);
      bbw_ = layer.bbw_;
      pbw_ = layer.pbw_;
    }
    wstride_ = layer.wstride_;
    pstride_ = layer.pstride_;
    window_size_ = layer.window_size_;
    window_shift_ = layer.window_shift_;
  }

  template<> xnnProjectedBLSTMLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix>::xnnProjectedBLSTMLayer(const xnnProjectedBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> &layer) {
    supportBlockEval_ = layer.supportBlockEval_;

    Wfw_.quantize(layer.Wfw_, 0, 0.0f);
    Rfw_.quantize(layer.Rfw_, 0, 0.0f);
    Mfw_.quantize(layer.Mfw_, 0, 0.0f);
    bfw_ = layer.bfw_;
    pfw_ = layer.pfw_;

    isBidirectional_ = layer.isBidirectional_;
    if (isBidirectional_) {
      Wbw_.quantize(layer.Wbw_, 0, 0.0f);
      Rbw_.quantize(layer.Rbw_, 0, 0.0f);
      Mbw_.quantize(layer.Mbw_, 0, 0.0f);
      bbw_ = layer.bbw_;
      pbw_ = layer.pbw_;
    }
    wstride_ = layer.wstride_;
    pstride_ = layer.pstride_;
    window_size_ = layer.window_size_;
    window_shift_ = layer.window_shift_;
  }

  template<> xnnConvolutionalLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>::xnnConvolutionalLayer(const xnnConvolutionalLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> &layer) {
    supportBlockEval_ = layer.supportBlockEval_;

    W_.quantize(layer.W_);
    b_ = layer.b_;
    vdim_ = layer.vdim_;
    udim_ = layer.udim_;
    patch_dim_ = layer.patch_dim_;
    patch_step_ = layer.patch_step_;
    patch_stride_ = layer.patch_stride_;

  }

  template<>
  xnnConvolutionalLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>::xnnConvolutionalLayer(const xnnConvolutionalLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> &layer) {
    supportBlockEval_ = layer.supportBlockEval_;

    W_.quantize(layer.W_, 0, 0.0f);
    b_ = layer.b_;
    vdim_ = layer.vdim_;
    udim_ = layer.udim_;
    patch_dim_ = layer.patch_dim_;
    patch_step_ = layer.patch_step_;
    patch_stride_ = layer.patch_stride_;

  }

  void xnnNet::quantizeFloat16(const xnnNet &net) {
    //xnnNet* netshort = new xnnNet();

    // clean-up
    for (size_t l = 0; l < layers_.size(); ++l) {
      delete layers_[l];
    }
    layers_.clear();

    for (size_t l = 0; l < net.NumLayers(); ++l) {
      if (net.layers_[l]->getLayerType() == idec::linearLayer) {
        XnnLinearLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new XnnLinearLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>(*reinterpret_cast<XnnLinearLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>*>(net.layers_[l]));
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (net.layers_[l]->getLayerType() == idec::reluLayer) {
        xnnReLULayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new xnnReLULayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>(*reinterpret_cast<xnnReLULayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>*>(net.layers_[l]));
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (net.layers_[l]->getLayerType() == idec::sigmoidLayer) {
        xnnSigmoidLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new xnnSigmoidLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>(*reinterpret_cast<xnnSigmoidLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>*>(net.layers_[l]));
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (net.layers_[l]->getLayerType() == idec::logsoftmaxLayer) {
        xnnLogSoftmaxLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new xnnLogSoftmaxLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>(*reinterpret_cast<xnnLogSoftmaxLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>*>(net.layers_[l]));
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (net.layers_[l]->getLayerType() == idec::blstmLayer) {
        xnnBLSTMLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new xnnBLSTMLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>(*reinterpret_cast<xnnBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>*>(net.layers_[l]));
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (net.layers_[l]->getLayerType() == idec::projectedblstmLayer) {
        xnnProjectedBLSTMLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new xnnProjectedBLSTMLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>(*reinterpret_cast<xnnProjectedBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>*>(net.layers_[l]));
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (net.layers_[l]->getLayerType() == idec::normalizationLayer) {
        xnnNormalizationLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new xnnNormalizationLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>(*reinterpret_cast<xnnNormalizationLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>*>(net.layers_[l]));
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (net.layers_[l]->getLayerType() == idec::convolutionalLayer) {
        xnnConvolutionalLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new xnnConvolutionalLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>(*reinterpret_cast<xnnConvolutionalLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>*>(net.layers_[l]));
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (net.layers_[l]->getLayerType() == idec::maxpoolingLayer) {
        xnnMaxpoolingLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new xnnMaxpoolingLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>(*reinterpret_cast<xnnMaxpoolingLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>*>(net.layers_[l]));
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (net.layers_[l]->getLayerType() == idec::addshiftLayer) {
        xnnAddShiftLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new xnnAddShiftLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>(*reinterpret_cast<xnnAddShiftLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>*>(net.layers_[l]));
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (net.layers_[l]->getLayerType() == idec::rescaleLayer) {
        xnnRescaleLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new xnnRescaleLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>(*reinterpret_cast<xnnRescaleLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>*>(net.layers_[l]));
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (net.layers_[l]->getLayerType() == idec::purereluLayer) {
        xnnPureReLULayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new xnnPureReLULayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>(*reinterpret_cast<xnnPureReLULayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>*>(net.layers_[l]));
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else {
        IDEC_ERROR << "unsupported layer type";
      }

      /*if (net.layers_[i]->getLayerType() == idec::linearLayer
      || net.layers_[i]->getLayerType() == idec::reluLayer
      || net.layers_[i]->getLayerType() == idec::sigmoidLayer)
      {
      xnnLinearLayer<xnnShortRuntimeMatrix, xnnFloatRuntimeMatrix, xnnShortRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer = NULL;
      if (net.layers_[i]->getLayerType() == idec::linearLayer)
      pLayer = new xnnLinearLayer<xnnShortRuntimeMatrix, xnnFloatRuntimeMatrix, xnnShortRuntimeMatrix, xnnFloatRuntimeMatrix>();
      else if (net.layers_[i]->getLayerType() == idec::reluLayer)
      pLayer = new xnnReLULayer<xnnShortRuntimeMatrix, xnnFloatRuntimeMatrix, xnnShortRuntimeMatrix, xnnFloatRuntimeMatrix>();
      else if (net.layers_[i]->getLayerType() == idec::sigmoidLayer)
      pLayer = new xnnSigmoidLayer<xnnShortRuntimeMatrix, xnnFloatRuntimeMatrix, xnnShortRuntimeMatrix, xnnFloatRuntimeMatrix>();

      xnnLinearLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>* temp =
      (xnnLinearLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>*)net.layers_[i];

      pLayer->getW_()->Resize(temp->getW_()->NumRows(), temp->getW_()->NumCols());
      pLayer->getb_()->Resize(temp->getb_()->NumRows(), temp->getb_()->NumCols());

      pLayer->getW_()->q.setQuantizer(*temp->getW_(), 0);

      for (size_t j = 0; j < temp->getW_()->NumCols(); j++)
      {
      pLayer->getW_()->q.quantize(
      pLayer->getW_()->Col(j),
      temp->getW_()->Col(j),
      temp->getW_()->NumRows());
      }

      for (size_t j = 0; j < temp->getb_()->NumCols(); j++)
      for (size_t k = 0; k < temp->getb_()->NumRows(); k++)
      pLayer->getb_()->Col(j)[k] = temp->getb_()->Col(j)[k];

      netshort->layers_.push_back((xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf>*)pLayer);
      temp = NULL;

      }
      else if (net.layers_[i]->getLayerType() == idec::logsoftmaxLayer)
      {
      xnnLogSoftmaxLayer<xnnShortRuntimeMatrix, xnnFloatRuntimeMatrix, xnnShortRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
      new xnnLogSoftmaxLayer<xnnShortRuntimeMatrix, xnnFloatRuntimeMatrix, xnnShortRuntimeMatrix, xnnFloatRuntimeMatrix>();

      xnnLogSoftmaxLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>* temp =
      (xnnLogSoftmaxLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>*)net.layers_[i];

      pLayer->getW_()->Resize(temp->getW_()->NumRows(), temp->getW_()->NumCols());
      pLayer->getb_()->Resize(temp->getb_()->NumRows(), temp->getb_()->NumCols());

      pLayer->getW_()->q.setQuantizer(*temp->getW_(), 0);

      for (size_t j = 0; j < temp->getW_()->NumCols(); j++)
      {
      pLayer->getW_()->q.quantize(pLayer->getW_()->Col(j),
      temp->getW_()->Col(j),
      temp->getW_()->NumRows());
      }

      for (size_t j = 0; j < temp->getb_()->NumCols(); j++)
      for (size_t k = 0; k < temp->getb_()->NumRows(); k++)
      pLayer->getb_()->Col(j)[k] = temp->getb_()->Col(j)[k];

      pLayer->setPrior(temp->getPrior_());
      pLayer->usePrior(temp->getUsePrior());
      pLayer->useRealProb(temp->getUseRealProb());

      netshort->layers_.push_back((xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf>*)pLayer);
      temp = NULL;
      }
      else if (net.layers_[i]->getLayerType() == idec::pnormLayer)
      {
      xnnPnormLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
      new xnnPnormLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();

      xnnPnormLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>* temp =
      (xnnPnormLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>*)net.layers_[i];

      pLayer->setPnormLayer(temp->uDim(), temp->vDim(), temp->getGroupsize(), temp->getP());

      netshort->layers_.push_back((xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf>*)pLayer);
      temp = NULL;
      }
      else if (net.layers_[i]->getLayerType() == idec::normLayer)
      {
      xnnNormLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
      new xnnNormLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>();

      xnnNormLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>* temp
      = (xnnNormLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>*)net.layers_[i];

      pLayer->setNormLayer(temp->uDim(), temp->getNorm());

      netshort->layers_.push_back((xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf>*)pLayer);
      temp = NULL;
      }
      else
      {
      xnnLayerBase<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>* pLayer
      = (xnnLayerBase<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>*)net.layers_[i];

      netshort->layers_.push_back((xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf>*)pLayer);
      }*/


    }

    //return netshort;
  }

  void xnnNet::quantizeFloat8(const xnnNet &net) {
    // clean-up
    for (size_t l = 0; l < layers_.size(); ++l) {
      delete layers_[l];
    }
    layers_.clear();

    for (size_t l = 0; l < net.NumLayers(); ++l) {
      if (net.layers_[l]->getLayerType() == idec::linearLayer && l == 0) {
        XnnLinearLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new XnnLinearLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix>(*reinterpret_cast<XnnLinearLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>*>(net.layers_[l]));
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (net.layers_[l]->getLayerType() == idec::linearLayer && l != 0) {
        XnnLinearLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new XnnLinearLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix>(*reinterpret_cast<XnnLinearLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>*>(net.layers_[l]));
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));

      }
      else if (net.layers_[l]->getLayerType() == idec::reluLayer) {
        xnnReLULayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new xnnReLULayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix>(*reinterpret_cast<xnnReLULayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>*>(net.layers_[l]));
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (net.layers_[l]->getLayerType() == idec::sigmoidLayer) {
        xnnSigmoidLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new xnnSigmoidLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix>(*reinterpret_cast<xnnSigmoidLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>*>(net.layers_[l]));
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (net.layers_[l]->getLayerType() == idec::logsoftmaxLayer) {
        xnnLogSoftmaxLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new xnnLogSoftmaxLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix>(*reinterpret_cast<xnnLogSoftmaxLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>*>(net.layers_[l]));
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (net.layers_[l]->getLayerType() == idec::blstmLayer) {
        xnnBLSTMLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new xnnBLSTMLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix>(*reinterpret_cast<xnnBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>*>(net.layers_[l]));
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (net.layers_[l]->getLayerType() == idec::projectedblstmLayer) {
        xnnProjectedBLSTMLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new xnnProjectedBLSTMLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix>(*reinterpret_cast<xnnProjectedBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>*>(net.layers_[l]));
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (net.layers_[l]->getLayerType() == idec::normalizationLayer) {
        xnnNormalizationLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new xnnNormalizationLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>(*reinterpret_cast<xnnNormalizationLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>*>(net.layers_[l]));
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (net.layers_[l]->getLayerType() == idec::convolutionalLayer) {
        xnnConvolutionalLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new xnnConvolutionalLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>(*reinterpret_cast<xnnConvolutionalLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>*>(net.layers_[l]));
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (net.layers_[l]->getLayerType() == idec::maxpoolingLayer) {
        xnnMaxpoolingLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new xnnMaxpoolingLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>(*reinterpret_cast<xnnMaxpoolingLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>*>(net.layers_[l]));
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (net.layers_[l]->getLayerType() == idec::addshiftLayer) {
        xnnAddShiftLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new xnnAddShiftLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>(*reinterpret_cast<xnnAddShiftLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>*>(net.layers_[l]));
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (net.layers_[l]->getLayerType() == idec::rescaleLayer) {
        xnnRescaleLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new xnnRescaleLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>(*reinterpret_cast<xnnRescaleLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>*>(net.layers_[l]));
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else if (net.layers_[l]->getLayerType() == idec::purereluLayer) {
        xnnPureReLULayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> *pLayer =
          new xnnPureReLULayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>(*reinterpret_cast<xnnPureReLULayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>*>(net.layers_[l]));
        layers_.push_back(reinterpret_cast<xnnLayerBase<xnnRuntimeMatrixItf, xnnRuntimeMatrixItf> *>(pLayer));
      }
      else {
        IDEC_ERROR << "unsupported layer type";
      }


    }

    //return netshort;
  }

  float xnnAmEvaluator::logLikelihood(int fr, int st) {
    // keep [pos, pos+block_] in activations_
    // always assume we ask frame indecies in ascending order
    //if (fr < pos_)
    //    IDEC_ERROR << "old frame index requested, not allowed";
    if (fr < startFrame_ || fr >= startFrame_ + (int)feat_.NumCols() || st >= (int)net_.uDim())
      return(LZERO);

    return logLikelihood(fr)[st] * acscale_;
  }


  float* xnnAmEvaluator::logLikelihood(int fr) {


    if (pos_ == -1 || fr >= pos_ + (int)defaultBlocksize_) {

      // need to run forwardprop
      // create feature view
      for (size_t i = 0; i < net_.NumLayers(); ++i) {
        if (net_.Layer(i)->isBlockEvalSupported() || pos_ == -1) {
          // need to evaluate this layer
          xnnFloatRuntimeMatrixView v(i == 0 ? feat_ : activations_[i - 1]); // input is feature or activations of the previous layer

          if (net_.Layer(i)->isBlockEvalSupported() && (i == 0 || !net_.Layer(i - 1)->isBlockEvalSupported())) {
            // create a sub-view of at most defaultBlocksize_ columns
            size_t cols = std::min(defaultBlocksize_, v.NumCols() - size_t(fr - startFrame_));
            v.ColView(fr - startFrame_, cols);
            //start_feat_flag = 1;
          }

          xnnFloatRuntimeMatrix &u = activations_[i]; // output is the activations of the current layer

          if (net_.Layer(i)->getMatrixType() == idec::ShortFloat) {
            activations_float16_.quantize(v, 3);
            net_.Layer(i)->forwardProp(activations_float16_, u, intermediate_states_[i]);
          }
          else if (net_.Layer(i)->getMatrixType() == idec::CharFloat) {
            activations_float8_.quantize(v, 0);
            net_.Layer(i)->forwardProp(activations_float8_, u, intermediate_states_[i]);
          }
          else if (net_.Layer(i)->getMatrixType() == idec::FloatFloat) {
            net_.Layer(i)->forwardProp(v, u, intermediate_states_[i]);
          }
          else {
            IDEC_ERROR << "unsupported matrix type";
          }
        }
      }

      pos_ = (int)fr;
    }

    size_t pos = net_.Layer(net_.NumLayers() - 1)->isBlockEvalSupported() ? fr - pos_ : fr;

    return(activations_.rbegin()->Col(pos));


  }

  float xnnAmEvaluator::logLikelihood_lazy(int fr, int st) {

    //        xnnShortRuntimeMatrix temp;
    //int start_feat_flag = 0;
    //size_t tempCols = 0;

    if (pos_ == -1 || fr >= pos_ + (int)defaultBlocksize_) {
      // need to run forwardprop until the last hidden layer

      // reset activation of last layer
      activations_.rbegin()->SetAll(LZERO);

      // create feature view
      for (size_t i = 0; i < net_.NumLayers() - 1; ++i) {
        if (net_.Layer(i)->isBlockEvalSupported() || pos_ == -1) {
          // need to evaluate this layer
          xnnFloatRuntimeMatrixView v(i == 0 ? feat_ : activations_[i - 1]); // input is feature or activations of the previous layer

          if (net_.Layer(i)->isBlockEvalSupported() && (i == 0 || !net_.Layer(i - 1)->isBlockEvalSupported())) {
            // create a sub-view of at most defaultBlocksize_ columns
            size_t cols = std::min(defaultBlocksize_, v.NumCols() - size_t(fr - startFrame_));
            v.ColView(fr - startFrame_, cols);
            //start_feat_flag = 1;
          }

          xnnFloatRuntimeMatrix &u = activations_[i]; // output is the activations of the current layer

          /*if ((net_.Layer(i)->getLayerType() == idec::linearLayer
          || net_.Layer(i)->getLayerType() == idec::logsoftmaxLayer
          || net_.Layer(i)->getLayerType() == idec::reluLayer)
          && net_.Layer(i)->getMatrixType() == idec::ShortFloat)*/
          if (net_.Layer(i)->getMatrixType() == idec::ShortFloat) {
            activations_float16_.quantize(v, 3);
            net_.Layer(i)->forwardProp(activations_float16_, u, intermediate_states_[i]);

            if (i == net_.NumLayers() - 2) {
              // last hidden layer
              activations_float16_.quantize(u, 3);
              activations_[activations_.size() - 1].Resize(activations_[activations_.size() - 1].NumRows(), activations_float16_.NumCols());
            }
          }
          else if (net_.Layer(i)->getMatrixType() == idec::CharFloat) {
            activations_float8_.quantize(v, 0);
            net_.Layer(i)->forwardProp(activations_float16_, u, intermediate_states_[i]);

            if (i == net_.NumLayers() - 2) {
              // last hidden layer
              activations_float8_.quantize(u, 0);
              activations_[activations_.size() - 1].Resize(activations_[activations_.size() - 1].NumRows(), activations_float8_.NumCols());
            }
          }
          else if (net_.Layer(i)->getMatrixType() == idec::FloatFloat) {
            net_.Layer(i)->forwardProp(v, u, intermediate_states_[i]);

            if (i == net_.NumLayers() - 2) {
              // last hidden layer
              activations_[activations_.size() - 1].Resize(activations_[activations_.size() - 1].NumRows(), activations_[activations_.size() - 2].NumCols());
            }
          }
          else {
            IDEC_ERROR << "unsupported matrix type";
          }
        }
      }

      pos_ = (int)fr;
    }

    size_t pos = net_.Layer(net_.NumLayers() - 1)->isBlockEvalSupported() ? fr - pos_ : fr;
    float ret = activations_.rbegin()->Col(pos)[st];

    if (ret == LZERO) {
      // need to do lazy evaluation
      if (net_.Layer(net_.NumLayers() - 1)->getMatrixType() == idec::ShortFloat) {
        xnnFloat16RuntimeMatrixView v(activations_float16_); // input
        v.ColView(pos, v.NumCols() - pos);

        xnnFloatRuntimeMatrixView u(activations_[activations_.size() - 1]); // output
        u.ColRowView(pos, v.NumCols(), st, 1);

        net_.Layer(net_.NumLayers() - 1)->forwardPropRange(v, u, st, 1, threadId_);
      }
      else if (net_.Layer(net_.NumLayers() - 1)->getMatrixType() == idec::CharFloat) {
        xnnFloat8RuntimeMatrixView v(activations_float8_); // input
        v.ColView(pos, v.NumCols() - pos);

        xnnFloatRuntimeMatrixView u(activations_[activations_.size() - 1]); // output
        u.ColRowView(pos, v.NumCols(), st, 1);

        net_.Layer(net_.NumLayers() - 1)->forwardPropRange(v, u, st, 1, threadId_);
      }
      else if (net_.Layer(net_.NumLayers() - 1)->getMatrixType() == idec::FloatFloat) {
        xnnFloatRuntimeMatrixView v(activations_[activations_.size() - 2]); // input
        v.ColView(pos, v.NumCols() - pos);

        xnnFloatRuntimeMatrixView u(activations_[activations_.size() - 1]); // output
        u.ColRowView(pos, v.NumCols(), st, 1);

        net_.Layer(net_.NumLayers() - 1)->forwardPropRange(v, u, st, 1, threadId_);
      }
      else {
        IDEC_ERROR << "unsupported matrix type";
      }

      ret = activations_.rbegin()->Col(pos)[st];
    }

    return(ret);
  }


  // return different matrix types
  template<> XnnMatrixType xnnLayerBase<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>::getMatrixType() const {
    return(FloatFloat);
  }

  template<> XnnMatrixType xnnLayerBase<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>::getMatrixType() const {
    return(ShortFloat);
  }

  template<> XnnMatrixType xnnLayerBase<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix>::getMatrixType() const {
    return(CharFloat);
  }

  template<> XnnMatrixType XnnLinearLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>::getMatrixType() const {
    return(FloatFloat);
  }

  template<> XnnMatrixType XnnLinearLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>::getMatrixType() const {
    return(ShortFloat);
  }

  template<> XnnMatrixType XnnLinearLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix>::getMatrixType() const {
    return(CharFloat);
  }

  template<> XnnMatrixType xnnEmbeddingLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>::getMatrixType() const {
    return(FloatFloat);
  }

  template<> void xnnLogSoftmaxLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>::forwardPropRange(const xnnFloatRuntimeMatrix &v /*input*/, xnnFloatRuntimeMatrix &u /*output*/, size_t start_row, size_t num_rows, size_t threadId) const {
    u.Resize(num_rows, v.NumCols());

    xnnFloatRuntimeMatrixView W(W_), b(b_); // weight matrix and bias
    W.ColView(start_row, num_rows);
    b.RowView(start_row, num_rows);

    u.Setv(b);
    u.PlusMatTMat(W, v);

    if (use_real_prob_) {
      IDEC_ERROR << "use_real_prob_ can only be used in full forwardProp";
    }

    if (hasPrior()) {
      xnnFloatRuntimeMatrixView prior(prior_); // prior
      prior.RowView(start_row, num_rows);
      u.Minusv(prior);
    }
  }

  template<> void xnnBlockSoftmaxLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>::forwardPropRange(const xnnFloatRuntimeMatrix &v /*input*/, xnnFloatRuntimeMatrix &u /*output*/, size_t start_row, size_t num_rows, size_t threadId) const {
    xnnFloatRuntimeMatrixView U(u);
    U.RowView(start_row, num_rows);

    xnnFloatRuntimeMatrixView W(W_), b(b_); // weight matrix and bias
    W.ColView(start_row, num_rows);
    b.RowView(start_row, num_rows);

    U.Setv(b);
    U.PlusMatTMat(W, v);
    U.Softmax();
  }

  template<> void xnnLogSoftmaxLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>::forwardPropRange(const xnnFloat16RuntimeMatrix &v /*input*/, xnnFloatRuntimeMatrix &u /*output*/, size_t start_row, size_t num_rows, size_t threadId) const {
    u.Resize(num_rows, v.NumCols());

    xnnFloat16RuntimeMatrixView W(W_);
    xnnFloatRuntimeMatrixView b(b_); // weight matrix and bias
    W.ColView(start_row, num_rows);
    b.RowView(start_row, num_rows);

    u.Setv(b);
    //u.PlusMatTMat(W, v);
    u.PlusSmallMatTSmallMat(W, v);

    if (use_real_prob_) {
      IDEC_ERROR << "use_real_prob_ can only be used in full forwardProp";
    }

    if (hasPrior()) {
      xnnFloatRuntimeMatrixView prior(prior_); // prior
      prior.RowView(start_row, num_rows);
      u.Minusv(prior);
    }
  }

  template<> void xnnLogSoftmaxLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix>::forwardPropRange(const xnnFloat8RuntimeMatrix &v /*input*/, xnnFloatRuntimeMatrix &u /*output*/, size_t start_row, size_t num_rows, size_t threadId) const {
    u.Resize(num_rows, v.NumCols());

    xnnFloat8RuntimeMatrixView W(W_);
    xnnFloatRuntimeMatrixView b(b_); // weight matrix and bias
    W.ColView(start_row, num_rows);
    b.RowView(start_row, num_rows);

    u.Setv(b);
    //u.PlusMatTMat(W, v);
    u.PlusSmallMatTSmallMat(W, v);

    if (use_real_prob_) {
      IDEC_ERROR << "use_real_prob_ can only be used in full forwardProp";
    }

    if (hasPrior()) {
      xnnFloatRuntimeMatrixView prior(prior_); // prior
      prior.RowView(start_row, num_rows);
      u.Minusv(prior);
    }
  }

  template<> void xnnBlockSoftmaxLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>::forwardPropRange(const xnnFloat16RuntimeMatrix &v /*input*/, xnnFloatRuntimeMatrix &u /*output*/, size_t start_row, size_t num_rows, size_t threadId) const {
    xnnFloatRuntimeMatrixView U(u);
    U.RowView(start_row, num_rows);

    xnnFloat16RuntimeMatrixView W(W_);
    xnnFloatRuntimeMatrixView b(b_); // weight matrix and bias
    W.ColView(start_row, num_rows);
    b.RowView(start_row, num_rows);

    U.Setv(b);
    U.PlusMatTMat(W, v);
    //u.PlusSmallMatTSmallMat(W, v);
    U.Softmax();
  }

  template<> void xnnBlockSoftmaxLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix>::forwardPropRange(const xnnFloat8RuntimeMatrix &v /*input*/, xnnFloatRuntimeMatrix &u /*output*/, size_t start_row, size_t num_rows, size_t threadId) const {
    xnnFloatRuntimeMatrixView U(u);
    U.RowView(start_row, num_rows);

    xnnFloat8RuntimeMatrixView W(W_);
    xnnFloatRuntimeMatrixView b(b_); // weight matrix and bias
    W.ColView(start_row, num_rows);
    b.RowView(start_row, num_rows);

    U.Setv(b);
    U.PlusMatTMat(W, v);
    //u.PlusSmallMatTSmallMat(W, v);
    U.Softmax();
  }


  // ref. to tech memo: Design document for runtime evaluation of BLSTM layers, Sept., zhijiey, 2014


  //template<class WMatrix, class BMatrix, class MMatrix, class InputMatrix, class OutputMatrix> void xnnBLSTMLayer<WMatrix, BMatrix, MMatrix, InputMatrix, OutputMatrix>::forwardProp(const InputMatrix &v /*input*/, OutputMatrix &u /*output*/, std::vector<void *> &intermediate_states) const


#ifdef TEST_C_STYLE_BLSTM_FPROP
  blstm_forward::FloatMatrix XFloatMatToFloatMat(const xnnFloatRuntimeMatrix &inmat) {
    blstm_forward::FloatMatrix mat;
    mat.col_stride = inmat.ColStride();
    mat.data = inmat.Col(0);
    mat.num_cols = inmat.NumCols();
    mat.num_rows = inmat.NumRows();
    return mat;
  }

  template<> void xnnBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>::forwardProp(const xnnFloatRuntimeMatrix &v /*input*/, xnnFloatRuntimeMatrix &u /*output*/, std::vector<void *> &intermediate_states) const {
    // prepare intermediate states
    xnnFloatRuntimeMatrix *Ofw, *cfw, *cfwnl, *Buf_ufw, *Buf_cfw, *Obw, *cbw, *cbwnl;
    Ofw = cfw = cfwnl = Buf_ufw = Buf_cfw = Obw = cbw = cbwnl = NULL;
    Ofw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[0]);
    cfw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[1]);
    cfwnl = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[2]);
    Buf_ufw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[3]);
    Buf_cfw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[4]);
    if (isBidirectional_)
    {
      Obw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[5]);
      cbw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[6]);
      cbwnl = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[7]);
    }
    Ofw->Resize(wstride_ * 4, v.NumCols());
    cfw->Resize(wstride_, 1);
    cfwnl->Resize(wstride_, 1);
    Buf_ufw->Resize(wstride_, 1);
    Buf_cfw->Resize(wstride_, 1);
    if (isBidirectional_)
    {
      Obw->Resize(wstride_ * 4, v.NumCols());
      cbw->Resize(wstride_, 1);
      cbwnl->Resize(wstride_, 1);
    }
    u.Resize(uDim(), v.NumCols());



    typedef blstm_forward::FloatMatrix FloatMatrix;
    FloatMatrix fm_W_fw = XFloatMatToFloatMat(Wfw_);
    FloatMatrix fm_R_fw = XFloatMatToFloatMat(Rfw_);
    FloatMatrix fm_b_fw = XFloatMatToFloatMat(bfw_);
    FloatMatrix fm_p_fw = XFloatMatToFloatMat(pfw_);

    FloatMatrix fm_W_bw = XFloatMatToFloatMat(Wbw_);
    FloatMatrix fm_R_bw = XFloatMatToFloatMat(Rbw_);
    FloatMatrix fm_b_bw = XFloatMatToFloatMat(bbw_);
    FloatMatrix fm_p_bw = XFloatMatToFloatMat(pbw_);


    // input & output
    FloatMatrix fm_v = XFloatMatToFloatMat(v);
    FloatMatrix fm_u = XFloatMatToFloatMat(u);

    // intermediate states
    FloatMatrix fm_O_fw = XFloatMatToFloatMat(*Ofw);
    FloatMatrix fm_c_fw = XFloatMatToFloatMat(*cfw);
    FloatMatrix fm_c_fw_nl = XFloatMatToFloatMat(*cfwnl);
    FloatMatrix fm_buf_ufw = XFloatMatToFloatMat(*Buf_ufw);
    FloatMatrix fm_buf_cfw = XFloatMatToFloatMat(*Buf_cfw);
    FloatMatrix fm_O_bw = XFloatMatToFloatMat(*Obw);
    FloatMatrix fm_c_bw = XFloatMatToFloatMat(*cbw);
    FloatMatrix fm_c_bw_nl = XFloatMatToFloatMat(*cbwnl);



    BlstmForwardProp(&fm_W_fw, &fm_R_fw, &fm_b_fw, &fm_p_fw,
                     &fm_W_bw, &fm_R_bw, &fm_b_bw, &fm_p_bw,
                     &fm_v, &fm_u,
                     &fm_O_fw, &fm_c_fw, &fm_c_fw_nl,
                     &fm_buf_ufw, &fm_buf_cfw,
                     &fm_O_bw, &fm_c_bw, &fm_c_bw_nl,
                     wstride_, window_shift_, isBidirectional_);
  }
#else
  template<> void xnnBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>::forwardProp(const xnnFloatRuntimeMatrix &v /*input*/, xnnFloatRuntimeMatrix &u /*output*/, std::vector<void *> &intermediate_states) const {
    // prepare intermediate states
    xnnFloatRuntimeMatrix *Ofw, *cfw, *cfwnl, *Buf_ufw, *Buf_cfw, *Obw, *cbw, *cbwnl;
    Ofw = cfw = cfwnl = Buf_ufw = Buf_cfw = Obw = cbw = cbwnl = NULL;
    Ofw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[0]);
    cfw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[1]);
    cfwnl = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[2]);
    Buf_ufw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[3]);
    Buf_cfw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[4]);
    if (isBidirectional_) {
      Obw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[5]);
      cbw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[6]);
      cbwnl = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[7]);
    }
    //xnnFloat16RuntimeMatrix o;

    size_t len = v.NumCols();
    if (isForwardAppro_ == true) {
      len = (window_shift_ < v.NumCols()) ? window_shift_ : v.NumCols();
    }


    Ofw->Resize(wstride_ * 4, v.NumCols());
    cfw->Resize(wstride_, 1);
    cfwnl->Resize(wstride_, 1);
    Buf_ufw->Resize(wstride_, 1);
    Buf_cfw->Resize(wstride_, 1);
    if (isBidirectional_) {
      Obw->Resize(wstride_ * 4, v.NumCols());
      cbw->Resize(wstride_, 1);
      cbwnl->Resize(wstride_, 1);
    }

    u.Resize(uDim(), v.NumCols());

    // a lot of views to different matrices
    xnnFloatRuntimeMatrixView ufw(u), ubw(u);
    xnnFloatRuntimeMatrixView vfw(v), ofw_fa(*Ofw);

    xnnFloatRuntimeMatrixView /*ufwCol(ufw),*/  ofwCol(*Ofw); // obwCol is not used when isBidirectional_ == false
    xnnFloatRuntimeMatrixView bufufw(*Buf_ufw), bufcfw(*Buf_cfw);


    vfw.ColView(0, len);
    ofw_fa.ColView(0, len);
    // Step 1. activation calcuated on non-recurrent input
    Ofw->SetZero();
    ofw_fa.PlusMatTMat(Wfw_, vfw);
    ofw_fa.Plusv(bfw_);




    for (size_t t = 0; t < len; ++t) {

      ofwCol.ColView(t, 1);

      if (t == 0) {
        ofwCol.PlusMatTMat(Rfw_, bufufw);
      }
      else {
        ufw.ColRowView(t - 1, 1, 0, wstride_);
        //o.quantize(ufw, 3);
        ofwCol.PlusMatTMat(Rfw_, ufw);
      }

      xnnFloatRuntimeMatrixView ni(*Ofw), gate(*Ofw), peephole(pfw_);

      gate.ColRowView(t, 1, wstride_, wstride_); // select input gate
      if (t != 0) {
        peephole.ColView(0, 1); // input gate
        gate.ScalePlusvElemProdv(1.0f, *cfw, peephole);
      }
      else {
        peephole.ColView(0, 1);
        gate.ScalePlusvElemProdv(1.0f, bufcfw, peephole);
      }
      gate.Sigmoid();

      gate.ColRowView(t, 1, wstride_ * 2, wstride_); // select forget gate
      if (t != 0) {
        peephole.ColView(1, 1); // forget gate
        gate.ScalePlusvElemProdv(1.0f, *cfw, peephole);
      }
      else {
        peephole.ColView(1, 1);
        gate.ScalePlusvElemProdv(1.0f, bufcfw, peephole);
      }
      gate.Sigmoid();

      ni.ColRowView(t, 1, 0, wstride_); // select node input
      ni.Tanh();

      // Step 2(e). cell state
      if (t != 0) {
        cfw->ScalePlusvElemProdv(0, *cfw, gate);
      }
      else {
        cfw->ScalePlusvElemProdv(0, bufcfw, gate);
      }

      gate.ColRowView(t, 1, wstride_, wstride_); // select input gate
      //cfw_[threadId].ScalePlusvElemProdv(t == 0 ? 0 : 1.0f, ni, gate);
      cfw->ScalePlusvElemProdv(1.0f, ni, gate);

      cfw->ApplyFloor(-50);
      cfw->ApplyCeiling(50);

      // Step 2(g). non-linear cell state
      *cfwnl = *cfw;
      cfwnl->Tanh();

      // Step 2(f). peephole for og
      gate.ColRowView(t, 1, wstride_ * 3, wstride_); // select input gate
      peephole.ColView(2, 1); // output gate
      gate.ScalePlusvElemProdv(1.0f, *cfw, peephole);
      gate.Sigmoid();

      // Step 2(h). final output
      //ufwCol.ColView(t, 1);
      ufw.ColRowView(t, 1, 0, wstride_);
      ufw.ScalePlusvElemProdv(0, *cfwnl, gate);


      if (t == window_shift_ - 1) {
        bufufw.Setv(ufw);
        bufcfw.Setv(*cfw);
      }
    }

    if (isBidirectional_) {

      xnnFloatRuntimeMatrixView /*ubwCol(ubw),*/  obwCol(*Obw);
      Obw->SetZero();
      Obw->PlusMatTMat(Wbw_, v);
      Obw->Plusv(bbw_);

      for (int t = (int)v.NumCols() - 1; t >= 0; --t) {
        obwCol.ColView(t, 1);

        if (t != (int)v.NumCols() - 1) {
          //ubwCol.ColView(t + 1, 1);
          ubw.ColRowView(t + 1, 1, wstride_, wstride_);
          obwCol.PlusMatTMat(Rbw_, ubw);
        }

        xnnFloatRuntimeMatrixView ni(*Obw), gate(*Obw), peephole(pbw_);

        gate.ColRowView(t, 1, wstride_, wstride_); // select input gate
        if (t != (int)v.NumCols() - 1) {
          peephole.ColView(0, 1); // input gate
          gate.ScalePlusvElemProdv(1.0f, *cbw, peephole);
        }
        gate.Sigmoid();

        gate.ColRowView(t, 1, wstride_ * 2, wstride_); // select forget gate
        if (t != (int)v.NumCols() - 1) {
          peephole.ColView(1, 1); // forget gate
          gate.ScalePlusvElemProdv(1.0f, *cbw, peephole);
        }
        gate.Sigmoid();

        ni.ColRowView(t, 1, 0, wstride_); // select node input
        ni.Tanh();

        // Step 3(e). cell state
        if (t != (int)v.NumCols() - 1) {
          cbw->ScalePlusvElemProdv(0, *cbw, gate);
        }

        gate.ColRowView(t, 1, wstride_, wstride_); // select input gate
        cbw->ScalePlusvElemProdv(t == (int)v.NumCols() - 1 ? 0 : 1.0f, ni, gate);


        cbw->ApplyFloor(-50);
        cbw->ApplyCeiling(50);

        // Step 3(g). non-linear cell state
        *cbwnl = *cbw;
        cbwnl->Tanh();

        // Step 3(f). peephole for og
        gate.ColRowView(t, 1, wstride_ * 3, wstride_); // select input gate
        peephole.ColView(2, 1); // output gate
        gate.ScalePlusvElemProdv(1.0f, *cbw, peephole);
        gate.Sigmoid();

        // Step 3(h). final output
        //ubwCol.ColView(t, 1);
        ubw.ColRowView(t, 1, wstride_, wstride_);
        ubw.ScalePlusvElemProdv(0, *cbwnl, gate);

      }
    }



  }
#endif

  template<>
  void xnnBLSTMLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>::forwardProp(const xnnFloat16RuntimeMatrix &v /*input*/, xnnFloatRuntimeMatrix &u /*output*/, std::vector<void *> &intermediate_states) const {
    // prepare intermediate states
    xnnFloatRuntimeMatrix *Ofw, *cfw, *cfwnl, *Buf_ufw, *Buf_cfw, *Obw, *cbw, *cbwnl;
    Ofw = cfw = cfwnl = Buf_ufw = Buf_cfw = Obw = cbw = cbwnl = NULL;
    Ofw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[0]);
    cfw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[1]);
    cfwnl = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[2]);
    Buf_ufw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[3]);
    Buf_cfw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[4]);
    if (isBidirectional_) {
      Obw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[5]);
      cbw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[6]);
      cbwnl = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[7]);
    }
    xnnFloat16RuntimeMatrix u_float16;


    Ofw->Resize(wstride_ * 4, v.NumCols());
    cfw->Resize(wstride_, 1);
    cfwnl->Resize(wstride_, 1);
    Buf_ufw->Resize(wstride_, 1);
    Buf_cfw->Resize(wstride_, 1);
    if (isBidirectional_) {
      Obw->Resize(wstride_ * 4, v.NumCols());
      cbw->Resize(wstride_, 1);
      cbwnl->Resize(wstride_, 1);
    }

    u.Resize(uDim(), v.NumCols());

    // a lot of views to different matrices
    xnnFloatRuntimeMatrixView ufw(u), ubw(u), ofw_fa(*Ofw);
    xnnFloat16RuntimeMatrixView vfw(v);

    xnnFloatRuntimeMatrixView /*ufwCol(ufw),*/  ofwCol(*Ofw); // obwCol is not used when isBidirectional_ == false
    xnnFloatRuntimeMatrixView bufufw(*Buf_ufw), bufcfw(*Buf_cfw);

    size_t len = v.NumCols();
    if (isForwardAppro_ == true) {
      len = (window_shift_ < v.NumCols()) ? window_shift_ : v.NumCols();
    }
    vfw.ColView(0, len);
    ofw_fa.ColView(0, len);

    // Step 1. activation calcuated on non-recurrent input
    Ofw->SetZero();
    ofw_fa.PlusMatTMat(Wfw_, vfw);
    ofw_fa.Plusv(bfw_);

    for (size_t t = 0; t < len; ++t) {

      ofwCol.ColView(t, 1);

      if (t == 0) {
        u_float16.quantize(bufufw, 3);
        ofwCol.PlusMatTMat(Rfw_, u_float16);
      }
      else {
        ufw.ColRowView(t - 1, 1, 0, wstride_);
        u_float16.quantize(ufw, 3);
        ofwCol.PlusMatTMat(Rfw_, u_float16);
      }

      xnnFloatRuntimeMatrixView ni(*Ofw), gate(*Ofw), peephole(pfw_);

      gate.ColRowView(t, 1, wstride_, wstride_); // select input gate
      if (t != 0) {
        peephole.ColView(0, 1); // input gate
        gate.ScalePlusvElemProdv(1.0f, *cfw, peephole);
      }
      else {
        peephole.ColView(0, 1);
        gate.ScalePlusvElemProdv(1.0f, bufcfw, peephole);
      }
      gate.Sigmoid();

      gate.ColRowView(t, 1, wstride_ * 2, wstride_); // select forget gate
      if (t != 0) {
        peephole.ColView(1, 1); // forget gate
        gate.ScalePlusvElemProdv(1.0f, *cfw, peephole);
      }
      else {
        peephole.ColView(1, 1);
        gate.ScalePlusvElemProdv(1.0f, bufcfw, peephole);
      }
      gate.Sigmoid();

      ni.ColRowView(t, 1, 0, wstride_); // select node input
      ni.Tanh();

      // Step 2(e). cell state
      if (t != 0) {
        cfw->ScalePlusvElemProdv(0, *cfw, gate);
      }
      else {
        cfw->ScalePlusvElemProdv(0, bufcfw, gate);
      }

      gate.ColRowView(t, 1, wstride_, wstride_); // select input gate
      //cfw_[threadId].ScalePlusvElemProdv(t == 0 ? 0 : 1.0f, ni, gate);
      cfw->ScalePlusvElemProdv(1.0f, ni, gate);

      cfw->ApplyFloor(-50);
      cfw->ApplyCeiling(50);

      // Step 2(g). non-linear cell state
      *cfwnl = *cfw;
      cfwnl->Tanh();

      // Step 2(f). peephole for og
      gate.ColRowView(t, 1, wstride_ * 3, wstride_); // select input gate
      peephole.ColView(2, 1); // output gate
      gate.ScalePlusvElemProdv(1.0f, *cfw, peephole);
      gate.Sigmoid();

      // Step 2(h). final output
      //ufwCol.ColView(t, 1);
      ufw.ColRowView(t, 1, 0, wstride_);
      ufw.ScalePlusvElemProdv(0, *cfwnl, gate);


      if (t == window_shift_ - 1) {
        bufufw.Setv(ufw);
        bufcfw.Setv(*cfw);
      }
    }

    if (isBidirectional_) {

      xnnFloatRuntimeMatrixView /*ubwCol(ubw),*/  obwCol(*Obw);
      Obw->SetZero();
      Obw->PlusMatTMat(Wbw_, v);
      Obw->Plusv(bbw_);

      for (int t = (int)v.NumCols() - 1; t >= 0; --t) {
        obwCol.ColView(t, 1);

        if (t != (int)v.NumCols() - 1) {
          //ubwCol.ColView(t + 1, 1);
          ubw.ColRowView(t + 1, 1, wstride_, wstride_);
          u_float16.quantize(ubw, 3);
          obwCol.PlusMatTMat(Rbw_, u_float16);
        }

        xnnFloatRuntimeMatrixView ni(*Obw), gate(*Obw), peephole(pbw_);

        gate.ColRowView(t, 1, wstride_, wstride_); // select input gate
        if (t != (int)v.NumCols() - 1) {
          peephole.ColView(0, 1); // input gate
          gate.ScalePlusvElemProdv(1.0f, *cbw, peephole);
        }
        gate.Sigmoid();

        gate.ColRowView(t, 1, wstride_ * 2, wstride_); // select forget gate
        if (t != (int)v.NumCols() - 1) {
          peephole.ColView(1, 1); // forget gate
          gate.ScalePlusvElemProdv(1.0f, *cbw, peephole);
        }
        gate.Sigmoid();

        ni.ColRowView(t, 1, 0, wstride_); // select node input
        ni.Tanh();

        // Step 3(e). cell state
        if (t != (int)v.NumCols() - 1) {
          cbw->ScalePlusvElemProdv(0, *cbw, gate);
        }

        gate.ColRowView(t, 1, wstride_, wstride_); // select input gate
        cbw->ScalePlusvElemProdv(t == (int)v.NumCols() - 1 ? 0 : 1.0f, ni, gate);


        cbw->ApplyFloor(-50);
        cbw->ApplyCeiling(50);

        // Step 3(g). non-linear cell state
        *cbwnl = *cbw;
        cbwnl->Tanh();

        // Step 3(f). peephole for og
        gate.ColRowView(t, 1, wstride_ * 3, wstride_); // select input gate
        peephole.ColView(2, 1); // output gate
        gate.ScalePlusvElemProdv(1.0f, *cbw, peephole);
        gate.Sigmoid();

        // Step 3(h). final output
        //ubwCol.ColView(t, 1);
        ubw.ColRowView(t, 1, wstride_, wstride_);
        ubw.ScalePlusvElemProdv(0, *cbwnl, gate);

      }
    }



  }

  template<>
  void xnnBLSTMLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix>::forwardProp(const xnnFloat8RuntimeMatrix &v /*input*/, xnnFloatRuntimeMatrix &u /*output*/, std::vector<void *> &intermediate_states) const {
    // prepare intermediate states
    xnnFloatRuntimeMatrix *Ofw, *cfw, *cfwnl, *Buf_ufw, *Buf_cfw, *Obw, *cbw, *cbwnl;
    Ofw = cfw = cfwnl = Buf_ufw = Buf_cfw = Obw = cbw = cbwnl = NULL;
    Ofw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[0]);
    cfw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[1]);
    cfwnl = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[2]);
    Buf_ufw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[3]);
    Buf_cfw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[4]);
    if (isBidirectional_) {
      Obw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[5]);
      cbw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[6]);
      cbwnl = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[7]);
    }
    xnnFloat8RuntimeMatrix u_float8;


    Ofw->Resize(wstride_ * 4, v.NumCols());
    cfw->Resize(wstride_, 1);
    cfwnl->Resize(wstride_, 1);
    Buf_ufw->Resize(wstride_, 1);
    Buf_cfw->Resize(wstride_, 1);
    if (isBidirectional_) {
      Obw->Resize(wstride_ * 4, v.NumCols());
      cbw->Resize(wstride_, 1);
      cbwnl->Resize(wstride_, 1);
    }

    u.Resize(uDim(), v.NumCols());

    // a lot of views to different matrices
    xnnFloatRuntimeMatrixView ufw(u), ubw(u);

    xnnFloatRuntimeMatrixView /*ufwCol(ufw),*/  ofwCol(*Ofw); // obwCol is not used when isBidirectional_ == false
    xnnFloatRuntimeMatrixView bufufw(*Buf_ufw), bufcfw(*Buf_cfw);

    // Step 1. activation calcuated on non-recurrent input
    Ofw->SetZero();
    Ofw->PlusMatTMat(Wfw_, v);
    Ofw->Plusv(bfw_);

    for (size_t t = 0; t < v.NumCols(); ++t) {

      ofwCol.ColView(t, 1);

      if (t == 0) {
        u_float8.quantize(bufufw, 0);
        ofwCol.PlusMatTMat(Rfw_, u_float8);
      }
      else {
        ufw.ColRowView(t - 1, 1, 0, wstride_);
        u_float8.quantize(ufw, 0);
        ofwCol.PlusMatTMat(Rfw_, u_float8);
      }

      xnnFloatRuntimeMatrixView ni(*Ofw), gate(*Ofw), peephole(pfw_);

      gate.ColRowView(t, 1, wstride_, wstride_); // select input gate
      if (t != 0) {
        peephole.ColView(0, 1); // input gate
        gate.ScalePlusvElemProdv(1.0f, *cfw, peephole);
      }
      else {
        peephole.ColView(0, 1);
        gate.ScalePlusvElemProdv(1.0f, bufcfw, peephole);
      }
      gate.Sigmoid();

      gate.ColRowView(t, 1, wstride_ * 2, wstride_); // select forget gate
      if (t != 0) {
        peephole.ColView(1, 1); // forget gate
        gate.ScalePlusvElemProdv(1.0f, *cfw, peephole);
      }
      else {
        peephole.ColView(1, 1);
        gate.ScalePlusvElemProdv(1.0f, bufcfw, peephole);
      }
      gate.Sigmoid();

      ni.ColRowView(t, 1, 0, wstride_); // select node input
      ni.Tanh();

      // Step 2(e). cell state
      if (t != 0) {
        cfw->ScalePlusvElemProdv(0, *cfw, gate);
      }
      else {
        cfw->ScalePlusvElemProdv(0, bufcfw, gate);
      }

      gate.ColRowView(t, 1, wstride_, wstride_); // select input gate
      //cfw_[threadId].ScalePlusvElemProdv(t == 0 ? 0 : 1.0f, ni, gate);
      cfw->ScalePlusvElemProdv(1.0f, ni, gate);

      cfw->ApplyFloor(-50);
      cfw->ApplyCeiling(50);

      // Step 2(g). non-linear cell state
      *cfwnl = *cfw;
      cfwnl->Tanh();

      // Step 2(f). peephole for og
      gate.ColRowView(t, 1, wstride_ * 3, wstride_); // select input gate
      peephole.ColView(2, 1); // output gate
      gate.ScalePlusvElemProdv(1.0f, *cfw, peephole);
      gate.Sigmoid();

      // Step 2(h). final output
      //ufwCol.ColView(t, 1);
      ufw.ColRowView(t, 1, 0, wstride_);
      ufw.ScalePlusvElemProdv(0, *cfwnl, gate);


      if (t == window_shift_ - 1) {
        bufufw.Setv(ufw);
        bufcfw.Setv(*cfw);
      }
    }

    if (isBidirectional_) {

      xnnFloatRuntimeMatrixView /*ubwCol(ubw),*/  obwCol(*Obw);
      Obw->SetZero();
      Obw->PlusMatTMat(Wbw_, v);
      Obw->Plusv(bbw_);

      for (int t = (int)v.NumCols() - 1; t >= 0; --t) {
        obwCol.ColView(t, 1);

        if (t != (int)v.NumCols() - 1) {
          //ubwCol.ColView(t + 1, 1);
          ubw.ColRowView(t + 1, 1, wstride_, wstride_);
          u_float8.quantize(ubw, 0);
          obwCol.PlusMatTMat(Rbw_, u_float8);
        }

        xnnFloatRuntimeMatrixView ni(*Obw), gate(*Obw), peephole(pbw_);

        gate.ColRowView(t, 1, wstride_, wstride_); // select input gate
        if (t != (int)v.NumCols() - 1) {
          peephole.ColView(0, 1); // input gate
          gate.ScalePlusvElemProdv(1.0f, *cbw, peephole);
        }
        gate.Sigmoid();

        gate.ColRowView(t, 1, wstride_ * 2, wstride_); // select forget gate
        if (t != (int)v.NumCols() - 1) {
          peephole.ColView(1, 1); // forget gate
          gate.ScalePlusvElemProdv(1.0f, *cbw, peephole);
        }
        gate.Sigmoid();

        ni.ColRowView(t, 1, 0, wstride_); // select node input
        ni.Tanh();

        // Step 3(e). cell state
        if (t != (int)v.NumCols() - 1) {
          cbw->ScalePlusvElemProdv(0, *cbw, gate);
        }

        gate.ColRowView(t, 1, wstride_, wstride_); // select input gate
        cbw->ScalePlusvElemProdv(t == (int)v.NumCols() - 1 ? 0 : 1.0f, ni, gate);


        cbw->ApplyFloor(-50);
        cbw->ApplyCeiling(50);

        // Step 3(g). non-linear cell state
        *cbwnl = *cbw;
        cbwnl->Tanh();

        // Step 3(f). peephole for og
        gate.ColRowView(t, 1, wstride_ * 3, wstride_); // select input gate
        peephole.ColView(2, 1); // output gate
        gate.ScalePlusvElemProdv(1.0f, *cbw, peephole);
        gate.Sigmoid();

        // Step 3(h). final output
        //ubwCol.ColView(t, 1);
        ubw.ColRowView(t, 1, wstride_, wstride_);
        ubw.ScalePlusvElemProdv(0, *cbwnl, gate);

      }
    }
  }



  template<> void xnnProjectedBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>::forwardProp(const xnnFloatRuntimeMatrix &v /*input*/, xnnFloatRuntimeMatrix &u /*output*/, std::vector<void *> &intermediate_states) const {
    // prepare intermediate states
    xnnFloatRuntimeMatrix *Ofw, *cfw, *cfwnl, *Buf_ufw, *Buf_cfw, *Obw, *cbw, *cbwnl, *mfwnl, *mbwnl;
    Ofw = cfw = cfwnl = Buf_ufw = Buf_cfw = Obw = cbw = cbwnl = mfwnl = mbwnl = NULL;
    Ofw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[0]);
    cfw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[1]);
    cfwnl = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[2]);
    mfwnl = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[3]);
    Buf_ufw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[4]);
    Buf_cfw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[5]);
    if (isBidirectional_) {
      Obw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[6]);
      cbw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[7]);
      cbwnl = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[8]);
      mbwnl = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[9]);
    }



    Ofw->Resize(wstride_ * 4, v.NumCols());
    cfw->Resize(wstride_, 1);
    cfwnl->Resize(wstride_, 1);
    mfwnl->Resize(wstride_, 1);
    Buf_ufw->Resize(pstride_, 1);
    Buf_cfw->Resize(wstride_, 1);
    if (isBidirectional_) {
      Obw->Resize(wstride_ * 4, v.NumCols());
      cbw->Resize(wstride_, 1);
      cbwnl->Resize(wstride_, 1);
      mbwnl->Resize(wstride_, 1);
    }

    u.Resize(uDim(), v.NumCols());
    u.SetZero();

    // a lot of views to different matrices
    xnnFloatRuntimeMatrixView ufw(u), ubw(u);

    xnnFloatRuntimeMatrixView /*ufwCol(ufw),*/  ofwCol(*Ofw); // obwCol is not used when isBidirectional_ == false
    xnnFloatRuntimeMatrixView bufufw(*Buf_ufw), bufcfw(*Buf_cfw);

    // Step 1. activation calcuated on non-recurrent input
    Ofw->SetZero();
    Ofw->PlusMatTMat(Wfw_, v);
    Ofw->Plusv(bfw_);

    mfwnl->SetZero();

    for (size_t t = 0; t < v.NumCols(); ++t) {

      ofwCol.ColView(t, 1);

      if (t == 0) {
        ofwCol.PlusMatTMat(Rfw_, bufufw);
      }
      else {
        ufw.ColRowView(t - 1, 1, 0, pstride_);
        ofwCol.PlusMatTMat(Rfw_, ufw);
      }

      xnnFloatRuntimeMatrixView ni(*Ofw), gate(*Ofw), peephole(pfw_);

      gate.ColRowView(t, 1, wstride_, wstride_); // select input gate
      if (t != 0) {
        peephole.ColView(0, 1); // input gate
        gate.ScalePlusvElemProdv(1.0f, *cfw, peephole);
      }
      else {
        peephole.ColView(0, 1);
        gate.ScalePlusvElemProdv(1.0f, bufcfw, peephole);
      }
      gate.Sigmoid();

      gate.ColRowView(t, 1, wstride_ * 2, wstride_); // select forget gate
      if (t != 0) {
        peephole.ColView(1, 1); // forget gate
        gate.ScalePlusvElemProdv(1.0f, *cfw, peephole);
      }
      else {
        peephole.ColView(1, 1);
        gate.ScalePlusvElemProdv(1.0f, bufcfw, peephole);
      }
      gate.Sigmoid();

      ni.ColRowView(t, 1, 0, wstride_); // select node input
      ni.Tanh();

      // Step 2(e). cell state
      if (t != 0) {
        cfw->ScalePlusvElemProdv(0, *cfw, gate);
      }
      else {
        cfw->ScalePlusvElemProdv(0, bufcfw, gate);
      }

      gate.ColRowView(t, 1, wstride_, wstride_); // select input gate
      cfw->ScalePlusvElemProdv(1.0f, ni, gate);

      cfw->ApplyFloor(-50);
      cfw->ApplyCeiling(50);

      // Step 2(g). non-linear cell state
      *cfwnl = *cfw;
      cfwnl->Tanh();

      // Step 2(f). peephole for og
      gate.ColRowView(t, 1, wstride_ * 3, wstride_); // select input gate
      peephole.ColView(2, 1); // output gate
      gate.ScalePlusvElemProdv(1.0f, *cfw, peephole);
      gate.Sigmoid();

      // Step 2(h). final output
      mfwnl->ScalePlusvElemProdv(0, *cfwnl, gate);

      ufw.ColRowView(t, 1, 0, pstride_);
      ufw.PlusMatTMat(Mfw_, *mfwnl);

      if (t == window_shift_ - 1) {
        bufufw.Setv(ufw);
        bufcfw.Setv(*cfw);
      }
    }


    if (isBidirectional_) {

      xnnFloatRuntimeMatrixView /*ubwCol(ubw),*/  obwCol(*Obw);
      Obw->SetZero();
      Obw->PlusMatTMat(Wbw_, v);
      Obw->Plusv(bbw_);

      mbwnl->SetZero();

      for (int t = (int)v.NumCols() - 1; t >= 0; --t) {
        obwCol.ColView(t, 1);

        if (t != (int)v.NumCols() - 1) {
          ubw.ColRowView(t + 1, 1, pstride_, pstride_);
          obwCol.PlusMatTMat(Rbw_, ubw);
        }

        xnnFloatRuntimeMatrixView ni(*Obw), gate(*Obw), peephole(pbw_);

        gate.ColRowView(t, 1, wstride_, wstride_); // select input gate
        if (t != (int)v.NumCols() - 1) {
          peephole.ColView(0, 1); // input gate
          gate.ScalePlusvElemProdv(1.0f, *cbw, peephole);
        }
        gate.Sigmoid();

        gate.ColRowView(t, 1, wstride_ * 2, wstride_); // select forget gate
        if (t != (int)v.NumCols() - 1) {
          peephole.ColView(1, 1); // forget gate
          gate.ScalePlusvElemProdv(1.0f, *cbw, peephole);
        }
        gate.Sigmoid();

        ni.ColRowView(t, 1, 0, wstride_); // select node input
        ni.Tanh();

        // Step 3(e). cell state
        if (t != (int)v.NumCols() - 1) {
          cbw->ScalePlusvElemProdv(0, *cbw, gate);
        }

        gate.ColRowView(t, 1, wstride_, wstride_); // select input gate
        cbw->ScalePlusvElemProdv(t == (int)v.NumCols() - 1 ? 0 : 1.0f, ni, gate);


        cbw->ApplyFloor(-50);
        cbw->ApplyCeiling(50);

        // Step 3(g). non-linear cell state
        *cbwnl = *cbw;
        cbwnl->Tanh();

        // Step 3(f). peephole for og
        gate.ColRowView(t, 1, wstride_ * 3, wstride_); // select input gate
        peephole.ColView(2, 1); // output gate
        gate.ScalePlusvElemProdv(1.0f, *cbw, peephole);
        gate.Sigmoid();

        // Step 3(h). final output

        mbwnl->ScalePlusvElemProdv(0, *cbwnl, gate);

        ubw.ColRowView(t, 1, pstride_, pstride_);
        ubw.PlusMatTMat(Mbw_, *mbwnl);



      }
    }



  }


  template<> void xnnProjectedBLSTMLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix>::forwardProp(const xnnFloat16RuntimeMatrix &v /*input*/, xnnFloatRuntimeMatrix &u /*output*/, std::vector<void *> &intermediate_states) const {
    // prepare intermediate states
    xnnFloatRuntimeMatrix *Ofw, *cfw, *cfwnl, *Buf_ufw, *Buf_cfw, *Obw, *cbw, *cbwnl, *mfwnl, *mbwnl;
    Ofw = cfw = cfwnl = Buf_ufw = Buf_cfw = Obw = cbw = cbwnl = mfwnl = mbwnl = NULL;
    Ofw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[0]);
    cfw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[1]);
    cfwnl = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[2]);
    mfwnl = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[3]);
    Buf_ufw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[4]);
    Buf_cfw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[5]);
    if (isBidirectional_) {
      Obw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[6]);
      cbw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[7]);
      cbwnl = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[8]);
      mbwnl = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[9]);
    }
    xnnFloat16RuntimeMatrix u_float16;


    Ofw->Resize(wstride_ * 4, v.NumCols());
    cfw->Resize(wstride_, 1);
    cfwnl->Resize(wstride_, 1);
    mfwnl->Resize(wstride_, 1);
    Buf_ufw->Resize(pstride_, 1);
    Buf_cfw->Resize(wstride_, 1);
    if (isBidirectional_) {
      Obw->Resize(wstride_ * 4, v.NumCols());
      cbw->Resize(wstride_, 1);
      cbwnl->Resize(wstride_, 1);
      mbwnl->Resize(wstride_, 1);
    }

    u.Resize(uDim(), v.NumCols());
    u.SetZero();
    // a lot of views to different matrices
    xnnFloatRuntimeMatrixView ufw(u), ubw(u);

    xnnFloatRuntimeMatrixView /*ufwCol(ufw),*/  ofwCol(*Ofw); // obwCol is not used when isBidirectional_ == false
    xnnFloatRuntimeMatrixView bufufw(*Buf_ufw), bufcfw(*Buf_cfw);

    // Step 1. activation calcuated on non-recurrent input
    Ofw->SetZero();
    Ofw->PlusMatTMat(Wfw_, v);
    Ofw->Plusv(bfw_);
    mfwnl->SetZero();
    for (size_t t = 0; t < v.NumCols(); ++t) {

      ofwCol.ColView(t, 1);

      if (t == 0) {
        u_float16.quantize(bufufw, 3);
        ofwCol.PlusMatTMat(Rfw_, u_float16);
      }
      else {
        ufw.ColRowView(t - 1, 1, 0, pstride_);
        u_float16.quantize(ufw, 3);
        ofwCol.PlusMatTMat(Rfw_, u_float16);
      }

      xnnFloatRuntimeMatrixView ni(*Ofw), gate(*Ofw), peephole(pfw_);

      gate.ColRowView(t, 1, wstride_, wstride_); // select input gate
      if (t != 0) {
        peephole.ColView(0, 1); // input gate
        gate.ScalePlusvElemProdv(1.0f, *cfw, peephole);
      }
      else {
        peephole.ColView(0, 1);
        gate.ScalePlusvElemProdv(1.0f, bufcfw, peephole);
      }
      gate.Sigmoid();

      gate.ColRowView(t, 1, wstride_ * 2, wstride_); // select forget gate
      if (t != 0) {
        peephole.ColView(1, 1); // forget gate
        gate.ScalePlusvElemProdv(1.0f, *cfw, peephole);
      }
      else {
        peephole.ColView(1, 1);
        gate.ScalePlusvElemProdv(1.0f, bufcfw, peephole);
      }
      gate.Sigmoid();

      ni.ColRowView(t, 1, 0, wstride_); // select node input
      ni.Tanh();

      // Step 2(e). cell state
      if (t != 0) {
        cfw->ScalePlusvElemProdv(0, *cfw, gate);
      }
      else {
        cfw->ScalePlusvElemProdv(0, bufcfw, gate);
      }

      gate.ColRowView(t, 1, wstride_, wstride_); // select input gate
      //cfw_[threadId].ScalePlusvElemProdv(t == 0 ? 0 : 1.0f, ni, gate);
      cfw->ScalePlusvElemProdv(1.0f, ni, gate);

      cfw->ApplyFloor(-50);
      cfw->ApplyCeiling(50);

      // Step 2(g). non-linear cell state
      *cfwnl = *cfw;
      cfwnl->Tanh();

      // Step 2(f). peephole for og
      gate.ColRowView(t, 1, wstride_ * 3, wstride_); // select input gate
      peephole.ColView(2, 1); // output gate
      gate.ScalePlusvElemProdv(1.0f, *cfw, peephole);
      gate.Sigmoid();

      // Step 2(h). final output

      mfwnl->ScalePlusvElemProdv(0, *cfwnl, gate);
      u_float16.quantize(*mfwnl, 3);
      ufw.ColRowView(t, 1, 0, pstride_);

      ufw.PlusMatTMat(Mfw_, u_float16);

      if (t == window_shift_ - 1) {
        bufufw.Setv(ufw);
        bufcfw.Setv(*cfw);
      }
    }

    if (isBidirectional_) {

      xnnFloatRuntimeMatrixView /*ubwCol(ubw),*/  obwCol(*Obw);
      Obw->SetZero();
      Obw->PlusMatTMat(Wbw_, v);
      Obw->Plusv(bbw_);
      mbwnl->SetZero();
      for (int t = (int)v.NumCols() - 1; t >= 0; --t) {
        obwCol.ColView(t, 1);

        if (t != (int)v.NumCols() - 1) {
          ubw.ColRowView(t + 1, 1, pstride_, pstride_);
          u_float16.quantize(ubw, 3);
          obwCol.PlusMatTMat(Rbw_, u_float16);
        }

        xnnFloatRuntimeMatrixView ni(*Obw), gate(*Obw), peephole(pbw_);

        gate.ColRowView(t, 1, wstride_, wstride_); // select input gate
        if (t != (int)v.NumCols() - 1) {
          peephole.ColView(0, 1); // input gate
          gate.ScalePlusvElemProdv(1.0f, *cbw, peephole);
        }
        gate.Sigmoid();

        gate.ColRowView(t, 1, wstride_ * 2, wstride_); // select forget gate
        if (t != (int)v.NumCols() - 1) {
          peephole.ColView(1, 1); // forget gate
          gate.ScalePlusvElemProdv(1.0f, *cbw, peephole);
        }
        gate.Sigmoid();

        ni.ColRowView(t, 1, 0, wstride_); // select node input
        ni.Tanh();

        // Step 3(e). cell state
        if (t != (int)v.NumCols() - 1) {
          cbw->ScalePlusvElemProdv(0, *cbw, gate);
        }

        gate.ColRowView(t, 1, wstride_, wstride_); // select input gate
        cbw->ScalePlusvElemProdv(t == (int)v.NumCols() - 1 ? 0 : 1.0f, ni, gate);


        cbw->ApplyFloor(-50);
        cbw->ApplyCeiling(50);

        // Step 3(g). non-linear cell state
        *cbwnl = *cbw;
        cbwnl->Tanh();

        // Step 3(f). peephole for og
        gate.ColRowView(t, 1, wstride_ * 3, wstride_); // select input gate
        peephole.ColView(2, 1); // output gate
        gate.ScalePlusvElemProdv(1.0f, *cbw, peephole);
        gate.Sigmoid();

        // Step 3(h). final output

        mbwnl->ScalePlusvElemProdv(0, *cbwnl, gate);
        u_float16.quantize(*mbwnl, 3);
        ubw.ColRowView(t, 1, pstride_, pstride_);
        ubw.PlusMatTMat(Mbw_, u_float16);


      }
    }



  }

  template<> void xnnProjectedBLSTMLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix>::forwardProp(const xnnFloat8RuntimeMatrix &v /*input*/, xnnFloatRuntimeMatrix &u /*output*/, std::vector<void *> &intermediate_states) const {
    // prepare intermediate states
    xnnFloatRuntimeMatrix *Ofw, *cfw, *cfwnl, *Buf_ufw, *Buf_cfw, *Obw, *cbw, *cbwnl, *mfwnl, *mbwnl;
    Ofw = cfw = cfwnl = Buf_ufw = Buf_cfw = Obw = cbw = cbwnl = mfwnl = mbwnl = NULL;
    Ofw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[0]);
    cfw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[1]);
    cfwnl = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[2]);
    mfwnl = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[3]);
    Buf_ufw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[4]);
    Buf_cfw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[5]);
    if (isBidirectional_) {
      Obw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[6]);
      cbw = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[7]);
      cbwnl = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[8]);
      mbwnl = static_cast<xnnFloatRuntimeMatrix*>(intermediate_states[9]);
    }
    xnnFloat8RuntimeMatrix u_float8;


    Ofw->Resize(wstride_ * 4, v.NumCols());
    cfw->Resize(wstride_, 1);
    cfwnl->Resize(wstride_, 1);
    mfwnl->Resize(wstride_, 1);
    Buf_ufw->Resize(pstride_, 1);
    Buf_cfw->Resize(wstride_, 1);
    if (isBidirectional_) {
      Obw->Resize(wstride_ * 4, v.NumCols());
      cbw->Resize(wstride_, 1);
      cbwnl->Resize(wstride_, 1);
      mbwnl->Resize(wstride_, 1);
    }

    u.Resize(uDim(), v.NumCols());
    u.SetZero();
    // a lot of views to different matrices
    xnnFloatRuntimeMatrixView ufw(u), ubw(u);

    xnnFloatRuntimeMatrixView /*ufwCol(ufw),*/  ofwCol(*Ofw); // obwCol is not used when isBidirectional_ == false
    xnnFloatRuntimeMatrixView bufufw(*Buf_ufw), bufcfw(*Buf_cfw);

    // Step 1. activation calcuated on non-recurrent input
    Ofw->SetZero();
    Ofw->PlusMatTMat(Wfw_, v);
    Ofw->Plusv(bfw_);
    mfwnl->SetZero();
    for (size_t t = 0; t < v.NumCols(); ++t) {

      ofwCol.ColView(t, 1);

      if (t == 0) {
        u_float8.quantize(bufufw, 0);
        ofwCol.PlusMatTMat(Rfw_, u_float8);
      }
      else {
        ufw.ColRowView(t - 1, 1, 0, pstride_);
        u_float8.quantize(ufw, 0);
        ofwCol.PlusMatTMat(Rfw_, u_float8);
      }

      xnnFloatRuntimeMatrixView ni(*Ofw), gate(*Ofw), peephole(pfw_);

      gate.ColRowView(t, 1, wstride_, wstride_); // select input gate
      if (t != 0) {
        peephole.ColView(0, 1); // input gate
        gate.ScalePlusvElemProdv(1.0f, *cfw, peephole);
      }
      else {
        peephole.ColView(0, 1);
        gate.ScalePlusvElemProdv(1.0f, bufcfw, peephole);
      }
      gate.Sigmoid();

      gate.ColRowView(t, 1, wstride_ * 2, wstride_); // select forget gate
      if (t != 0) {
        peephole.ColView(1, 1); // forget gate
        gate.ScalePlusvElemProdv(1.0f, *cfw, peephole);
      }
      else {
        peephole.ColView(1, 1);
        gate.ScalePlusvElemProdv(1.0f, bufcfw, peephole);
      }
      gate.Sigmoid();

      ni.ColRowView(t, 1, 0, wstride_); // select node input
      ni.Tanh();

      // Step 2(e). cell state
      if (t != 0) {
        cfw->ScalePlusvElemProdv(0, *cfw, gate);
      }
      else {
        cfw->ScalePlusvElemProdv(0, bufcfw, gate);
      }

      gate.ColRowView(t, 1, wstride_, wstride_); // select input gate
      //cfw_[threadId].ScalePlusvElemProdv(t == 0 ? 0 : 1.0f, ni, gate);
      cfw->ScalePlusvElemProdv(1.0f, ni, gate);

      cfw->ApplyFloor(-50);
      cfw->ApplyCeiling(50);

      // Step 2(g). non-linear cell state
      *cfwnl = *cfw;
      cfwnl->Tanh();

      // Step 2(f). peephole for og
      gate.ColRowView(t, 1, wstride_ * 3, wstride_); // select input gate
      peephole.ColView(2, 1); // output gate
      gate.ScalePlusvElemProdv(1.0f, *cfw, peephole);
      gate.Sigmoid();

      // Step 2(h). final output

      mfwnl->ScalePlusvElemProdv(0, *cfwnl, gate);
      u_float8.quantize(*mfwnl, 0);
      ufw.ColRowView(t, 1, 0, pstride_);

      ufw.PlusMatTMat(Mfw_, u_float8);

      if (t == window_shift_ - 1) {
        bufufw.Setv(ufw);
        bufcfw.Setv(*cfw);
      }
    }

    if (isBidirectional_) {

      xnnFloatRuntimeMatrixView /*ubwCol(ubw),*/  obwCol(*Obw);
      Obw->SetZero();
      Obw->PlusMatTMat(Wbw_, v);
      Obw->Plusv(bbw_);
      mbwnl->SetZero();
      for (int t = (int)v.NumCols() - 1; t >= 0; --t) {
        obwCol.ColView(t, 1);

        if (t != (int)v.NumCols() - 1) {
          ubw.ColRowView(t + 1, 1, pstride_, pstride_);
          u_float8.quantize(ubw, 0);
          obwCol.PlusMatTMat(Rbw_, u_float8);
        }

        xnnFloatRuntimeMatrixView ni(*Obw), gate(*Obw), peephole(pbw_);

        gate.ColRowView(t, 1, wstride_, wstride_); // select input gate
        if (t != (int)v.NumCols() - 1) {
          peephole.ColView(0, 1); // input gate
          gate.ScalePlusvElemProdv(1.0f, *cbw, peephole);
        }
        gate.Sigmoid();

        gate.ColRowView(t, 1, wstride_ * 2, wstride_); // select forget gate
        if (t != (int)v.NumCols() - 1) {
          peephole.ColView(1, 1); // forget gate
          gate.ScalePlusvElemProdv(1.0f, *cbw, peephole);
        }
        gate.Sigmoid();

        ni.ColRowView(t, 1, 0, wstride_); // select node input
        ni.Tanh();

        // Step 3(e). cell state
        if (t != (int)v.NumCols() - 1) {
          cbw->ScalePlusvElemProdv(0, *cbw, gate);
        }

        gate.ColRowView(t, 1, wstride_, wstride_); // select input gate
        cbw->ScalePlusvElemProdv(t == (int)v.NumCols() - 1 ? 0 : 1.0f, ni, gate);


        cbw->ApplyFloor(-50);
        cbw->ApplyCeiling(50);

        // Step 3(g). non-linear cell state
        *cbwnl = *cbw;
        cbwnl->Tanh();

        // Step 3(f). peephole for og
        gate.ColRowView(t, 1, wstride_ * 3, wstride_); // select input gate
        peephole.ColView(2, 1); // output gate
        gate.ScalePlusvElemProdv(1.0f, *cbw, peephole);
        gate.Sigmoid();

        // Step 3(h). final output

        mbwnl->ScalePlusvElemProdv(0, *cbwnl, gate);
        u_float8.quantize(*mbwnl, 0);
        ubw.ColRowView(t, 1, pstride_, pstride_);
        ubw.PlusMatTMat(Mbw_, u_float8);
      }
    }
  }



  template<> void xnnConvolutionalLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>::forwardProp(const xnnFloatRuntimeMatrix &v /*input*/, xnnFloatRuntimeMatrix &u /*output*/, std::vector<void *> &intermediate_states) const {
    u.Resize(uDim(), v.NumCols());
    if ((patch_stride_ - patch_dim_) % patch_step_ != 0) {
      IDEC_ERROR << ": (patch_stride_ - patch_dim_) % patch_step_ != 0 " << ": patch_stride_ " << patch_stride_ << "patch_dim_ " << patch_dim_ << "patch_step_ " << patch_step_;
    }
    size_t num_patches = 1 + (patch_stride_ - patch_dim_) / patch_step_;

    if (v.NumRows() % patch_stride_ != 0) {
      IDEC_ERROR << ": input_dim_ % patch_stride_ !=0    " << "input_dim_ " << v.NumRows() << "patch_stride_ " << patch_stride_;
    }

    size_t num_splice = v.NumRows() / patch_stride_;

    // filter dim:
    size_t filter_dim = num_splice * patch_dim_;
    // num filters:
    if (u.NumRows() % num_patches != 0) {
      IDEC_ERROR << ": output_dim_ % num_patches !=0    " << "output_dim_ " << u.NumRows() << "num_patches " << num_patches;
    }

    size_t num_filters = u.NumRows() / num_patches;
    // check parameter dims:
    if (num_filters != W_.NumCols()) {
      IDEC_ERROR << ": num_filters != W_.NumCols()    " << "num_filters " << num_filters << "W_.NumCols() " << W_.NumCols();
    }
    if (num_filters != b_.NumRows()) {
      IDEC_ERROR << ": num_filters != bias_.Dim()    " << "num_filters " << num_filters << "bias_.Dim() " << b_.NumRows();
    }
    if (filter_dim != W_.NumRows()) {
      IDEC_ERROR << ": num_filters != W_.NumRows()    " << "filter_dim " << filter_dim << "W_.NumRows() " << W_.NumRows();
    }

    /*---------------------------------------F1---------------------------------------------------------------------
    xnnFloatRuntimeMatrix vectorized_feature_patches_;
    std::vector<size_t> column_map_;
    vectorized_feature_patches_.Resize(filter_dim * num_patches, v.NumCols());
    column_map_.resize(filter_dim * num_patches);

    for (size_t p = 0, index = 0; p < num_patches; p++) {
    for (size_t s = 0; s < num_splice; s++) {
    for (size_t d = 0; d < patch_dim_; d++, index++) {
    column_map_[index] = p * patch_step_ + s * patch_stride_ + d;
    }
    }
    }

    vectorized_feature_patches_.CopyRows(v, column_map_);

    for (int32 p = 0; p < num_patches; p++) {
    xnnFloatRuntimeMatrixView tgt(u), patch(vectorized_feature_patches_);
    tgt.ColRowView(0, v.NumCols(), p * num_filters, num_filters);
    patch.ColRowView(0, vectorized_feature_patches_.NumCols(), p * filter_dim, filter_dim);

    tgt.Setv(b_);
    tgt.PlusMatTMat(W_, patch);
    }
    ---------------------------------------F1---------------------------------------------------------------------*/


    /*---------------------------------------F2---------------------------------------------------------------------*/
    xnnFloatRuntimeMatrix patches(filter_dim, v.NumCols() * num_patches);

    for (size_t p = 0; p < num_patches; p++) {
      for (size_t s = 0; s < num_splice; s++) {
        patches.CopyFloatSubMatrix(v, 0, p * patch_step_ + s * patch_stride_, p * v.NumCols(), s * patch_dim_, v.NumCols(), patch_dim_);
      }
    }

    xnnFloatRuntimeMatrix tgt(num_filters, v.NumCols() * num_patches);
    tgt.Setv(b_);
    tgt.PlusMatTMat(W_, patches);

    for (size_t s = 0; s < num_patches; ++s) {
      u.CopyFloatSubMatrix(tgt, s * v.NumCols(), 0, 0, s * num_filters, v.NumCols(), num_filters);
    }

  }


  template<> void xnnConvolutionalLayer<xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>::forwardProp(const xnnFloatRuntimeMatrix &v /*input*/, xnnFloatRuntimeMatrix &u /*output*/, std::vector<void *> &intermediate_states) const {
    u.Resize(uDim(), v.NumCols());
    if ((patch_stride_ - patch_dim_) % patch_step_ != 0) {
      IDEC_ERROR << ": (patch_stride_ - patch_dim_) % patch_step_ != 0 " << ": patch_stride_ " << patch_stride_ << "patch_dim_ " << patch_dim_ << "patch_step_ " << patch_step_;
    }
    size_t num_patches = 1 + (patch_stride_ - patch_dim_) / patch_step_;

    if (v.NumRows() % patch_stride_ != 0) {
      IDEC_ERROR << ": input_dim_ % patch_stride_ !=0    " << "input_dim_ " << v.NumRows() << "patch_stride_ " << patch_stride_;
    }

    size_t num_splice = v.NumRows() / patch_stride_;

    // filter dim:
    size_t filter_dim = num_splice * patch_dim_;
    // num filters:
    if (u.NumRows() % num_patches != 0) {
      IDEC_ERROR << ": output_dim_ % num_patches !=0    " << "output_dim_ " << u.NumRows() << "num_patches " << num_patches;
    }

    size_t num_filters = u.NumRows() / num_patches;
    // check parameter dims:
    if (num_filters != W_.NumCols()) {
      IDEC_ERROR << ": num_filters != W_.NumCols()    " << "num_filters " << num_filters << "W_.NumCols() " << W_.NumCols();
    }
    if (num_filters != b_.NumRows()) {
      IDEC_ERROR << ": num_filters != bias_.Dim()    " << "num_filters " << num_filters << "bias_.Dim() " << b_.NumRows();
    }
    if (filter_dim != W_.NumRows()) {
      IDEC_ERROR << ": num_filters != W_.NumRows()    " << "filter_dim " << filter_dim << "W_.NumRows() " << W_.NumRows();
    }


    /*---------------------------------------F1---------------------------------------------------------------------
    xnnFloatRuntimeMatrix vectorized_feature_patches_;
    xnnFloat16RuntimeMatrix u_float16;

    std::vector<size_t> column_map_;
    vectorized_feature_patches_.Resize(filter_dim * num_patches, v.NumCols());
    column_map_.resize(filter_dim * num_patches);

    for (size_t p = 0, index = 0; p < num_patches; p++) {
    for (size_t s = 0; s < num_splice; s++) {
    for (size_t d = 0; d < patch_dim_; d++, index++) {
    column_map_[index] = p * patch_step_ + s * patch_stride_ + d;
    }
    }
    }

    vectorized_feature_patches_.CopyRows(v, column_map_);

    for (int32 p = 0; p < num_patches; p++) {
    xnnFloatRuntimeMatrixView tgt(u);
    xnnFloatRuntimeMatrixView patch(vectorized_feature_patches_);
    tgt.ColRowView(0, v.NumCols(), p * num_filters, num_filters);
    patch.ColRowView(0, vectorized_feature_patches_.NumCols(), p * filter_dim, filter_dim);
    u_float16.quantize(patch, 3);
    tgt.Setv(b_);
    tgt.PlusMatTMat(W_, u_float16);
    }
    ---------------------------------------F1---------------------------------------------------------------------*/


    /*---------------------------------------F2---------------------------------------------------------------------*/
    xnnFloat16RuntimeMatrix u_float16;
    xnnFloatRuntimeMatrix patches(filter_dim, v.NumCols() * num_patches);

    for (size_t p = 0; p < num_patches; p++) {
      for (size_t s = 0; s < num_splice; s++) {
        patches.CopyFloatSubMatrix(v, 0, p * patch_step_ + s * patch_stride_, p * v.NumCols(), s * patch_dim_, v.NumCols(), patch_dim_);
      }
    }

    xnnFloatRuntimeMatrix tgt(num_filters, v.NumCols() * num_patches);
    u_float16.quantize(patches, 3);
    tgt.Setv(b_);
    tgt.PlusMatTMat(W_, u_float16);


    for (size_t s = 0; s < num_patches; ++s) {
      u.CopyFloatSubMatrix(tgt, s * v.NumCols(), 0, 0, s * num_filters, v.NumCols(), num_filters);
    }
  }

  template<> void xnnConvolutionalLayer<xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix>::forwardProp(const xnnFloatRuntimeMatrix &v /*input*/, xnnFloatRuntimeMatrix &u /*output*/, std::vector<void *> &intermediate_states) const {
    u.Resize(uDim(), v.NumCols());
    if ((patch_stride_ - patch_dim_) % patch_step_ != 0) {
      IDEC_ERROR << ": (patch_stride_ - patch_dim_) % patch_step_ != 0 " << ": patch_stride_ " << patch_stride_ << "patch_dim_ " << patch_dim_ << "patch_step_ " << patch_step_;
    }
    int32 num_patches = 1 + (int32)((patch_stride_ - patch_dim_) / patch_step_);

    if (v.NumRows() % patch_stride_ != 0) {
      IDEC_ERROR << ": input_dim_ % patch_stride_ !=0    " << "input_dim_ " << v.NumRows() << "patch_stride_ " << patch_stride_;
    }

    size_t num_splice = v.NumRows() / patch_stride_;

    // filter dim:
    size_t filter_dim = num_splice * patch_dim_;
    // num filters:
    if (u.NumRows() % num_patches != 0) {
      IDEC_ERROR << ": output_dim_ % num_patches !=0    " << "output_dim_ " << u.NumRows() << "num_patches " << num_patches;
    }

    size_t num_filters = u.NumRows() / num_patches;
    // check parameter dims:
    if (num_filters != W_.NumCols()) {
      IDEC_ERROR << ": num_filters != W_.NumCols()    " << "num_filters " << num_filters << "W_.NumCols() " << W_.NumCols();
    }
    if (num_filters != b_.NumRows()) {
      IDEC_ERROR << ": num_filters != bias_.Dim()    " << "num_filters " << num_filters << "bias_.Dim() " << b_.NumRows();
    }
    if (filter_dim != W_.NumRows()) {
      IDEC_ERROR << ": num_filters != W_.NumRows()    " << "filter_dim " << filter_dim << "W_.NumRows() " << W_.NumRows();
    }


    /*---------------------------------------F1---------------------------------------------------------------------
    xnnFloatRuntimeMatrix vectorized_feature_patches_;
    xnnFloat8RuntimeMatrix u_float8;

    std::vector<size_t> column_map_;
    vectorized_feature_patches_.Resize(filter_dim * num_patches, v.NumCols());
    column_map_.resize(filter_dim * num_patches);

    for (size_t p = 0, index = 0; p < num_patches; p++) {
    for (size_t s = 0; s < num_splice; s++) {
    for (size_t d = 0; d < patch_dim_; d++, index++) {
    column_map_[index] = p * patch_step_ + s * patch_stride_ + d;
    }
    }
    }

    vectorized_feature_patches_.CopyRows(v, column_map_);

    for (int32 p = 0; p < num_patches; p++) {
    xnnFloatRuntimeMatrixView tgt(u);
    xnnFloatRuntimeMatrixView patch(vectorized_feature_patches_);
    tgt.ColRowView(0, v.NumCols(), p * num_filters, num_filters);
    patch.ColRowView(0, vectorized_feature_patches_.NumCols(), p * filter_dim, filter_dim);
    u_float8.quantize(patch, 0);
    tgt.Setv(b_);
    tgt.PlusMatTMat(W_, u_float8);
    }
    ---------------------------------------F1---------------------------------------------------------------------*/


    /*---------------------------------------F2---------------------------------------------------------------------*/
    xnnFloat8RuntimeMatrix u_float8;
    xnnFloatRuntimeMatrix patches(filter_dim, v.NumCols() * num_patches);

    for (size_t p = 0; p < (size_t)num_patches; p++) {
      for (size_t s = 0; s < (size_t)num_splice; s++) {
        patches.CopyFloatSubMatrix(v, 0, p * patch_step_ + s * patch_stride_, p * v.NumCols(), s * patch_dim_, v.NumCols(), patch_dim_);
      }
    }

    xnnFloatRuntimeMatrix tgt(num_filters, v.NumCols() * num_patches);
    u_float8.quantize(patches, 0);
    tgt.Setv(b_);
    tgt.PlusMatTMat(W_, u_float8);


    for (size_t s = 0; s < num_patches; ++s) {
      u.CopyFloatSubMatrix(tgt, s * v.NumCols(), 0, 0, s * num_filters, v.NumCols(), num_filters);
    }
  }

};
