#ifndef XNN_NET_BLSTM_LAYER_H_
#define XNN_NET_BLSTM_LAYER_H_

#include "am/xnn_net/layer_base.h"

namespace idec {

//// BLSTM layer [zhijiey]
template<class WMatrix, class BMatrix, class MMatrix, class InputMatrix, class OutputMatrix>
class xnnBLSTMLayer : public xnnLayerBase < InputMatrix, OutputMatrix > {
  friend class xnnBLSTMLayer < xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat16RuntimeMatrix, xnnFloatRuntimeMatrix >;
  friend class xnnBLSTMLayer < xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloat8RuntimeMatrix, xnnFloatRuntimeMatrix >;

 protected:
  WMatrix Wfw_;        // forward weights applied to the input [W^{fw}(ni); W^{fw}(ig); W^{fw}(fg); W^{fw}(og)]^\top
  WMatrix Wbw_;        // backward weights applied to the input [W^{bw}(ni); W^{bw}(ig); W^{bw}(fg); W^{bw}(og)]^\top
  WMatrix Rfw_;        // forward weights applied to the recurrent output [R^{fw}(ni); R^{fw}(ig); R^{fw}(fg); R^{fw}(og)]^\top
  WMatrix Rbw_;        // backward weights applied to the recurrent output [R^{bw}(ni); R^{bw}(ig); R^{bw}(fg); R^{bw}(og)]^\top
  BMatrix bfw_;        // forward bias [b^{fw}(ni); b^{fw}(ig); b^{fw}(fg); b^{fw}(og)]
  BMatrix bbw_;        // backward bias [b^{bw}(ni); b^{bw}(ig); b^{bw}(fg); b^{bw}(og)]
  BMatrix pfw_;        // forward peephole weights [p^{fw}(ig) p^{fw}(fg) p^{fw}(og)]
  BMatrix pbw_;        // backward peephole weights [p^{bw}(ig) p^{bw}(fg) p^{bw}(og)]

  bool isBidirectional_;
  size_t wstride_;
  size_t window_size_;
  size_t window_shift_;

 protected:
  virtual void InitIntermediateStates(std::vector<void *> &intermediate_states) {
    intermediate_states.resize(isBidirectional_ ? 8 : 5);

    for (size_t i = 0; i < 5; i++) {
      // order is as follows:
      // Ofw, cfw, cfwnl, Buf_ufw, Buf_cfw
      intermediate_states[i] = new MMatrix();
    }

    if (isBidirectional_) {
      for (size_t i = 5; i < 8; i++) {
        // order is as follows:
        // Obw, cbw, cbwnl
        intermediate_states[i] = new MMatrix();
      }
    }
  }

  virtual void DeleteIntermediateStates(std::vector<void *> &intermediate_states) {
    for (size_t i = 0; i < 5; i++) {
      // order is as follows:
      // Ofw, cfw, cfwnl, Buf_ufw, Buf_cfw
      delete static_cast<MMatrix *>(intermediate_states[i]);
    }

    if (isBidirectional_) {
      for (size_t i = 5; i < 8; i++) {
        // order is as follows:
        // Obw, cbw, cbwnl
        delete static_cast<MMatrix *>(intermediate_states[i]);
      }
    }

    intermediate_states.clear();
  }

 public:
  using xnnLayerBase<InputMatrix, OutputMatrix>::supportBlockEval_;
  bool isForwardAppro_;

  xnnBLSTMLayer() : isBidirectional_(true) {
    supportBlockEval_ = false;
    isForwardAppro_ = false;
  }
  //xnnBLSTMLayer(const WMatrix &Wfw, const BMatrix &b) : Wfw_(Wfw), b_(b) {};
  xnnBLSTMLayer(const xnnBLSTMLayer<xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix, xnnFloatRuntimeMatrix> &layer);
  virtual void forwardProp(const InputMatrix &v /*input*/, OutputMatrix &u /*output*/, std::vector<void *> &intermediate_states) const;

  virtual size_t vDim() const { return Wfw_.NumRows(); }
  virtual size_t uDim() const { return (Wfw_.NumCols() + Wbw_.NumCols()) / 4; }

  virtual XnnLayerType getLayerType() const { return blstmLayer; }

  void setForwardAppro(bool isForwardAppro) {isForwardAppro_ = isForwardAppro;}

  void setBidirectional(bool isB) { isBidirectional_ = isB; }

  void readKaldiLayerNnet1(std::istream &is/*, size_t nThread = 1*/);

  virtual void Serialize(SerializeHelper &helper) {
    xnnLayerBase<InputMatrix, OutputMatrix>::Serialize(helper);
    Wfw_.Serialize(helper);
    Wbw_.Serialize(helper);
    Rfw_.Serialize(helper);
    Rbw_.Serialize(helper);
    bfw_.Serialize(helper);
    bbw_.Serialize(helper);
    pfw_.Serialize(helper);
    pbw_.Serialize(helper);

    helper.Serialize(isBidirectional_);
    helper.Serialize(wstride_);
  }

  virtual void Deserialize(SerializeHelper &helper) {
    xnnLayerBase<InputMatrix, OutputMatrix>::Deserialize(helper);
    Wfw_.Deserialize(helper);
    Wbw_.Deserialize(helper);
    Rfw_.Deserialize(helper);
    Rbw_.Deserialize(helper);
    bfw_.Deserialize(helper);
    bbw_.Deserialize(helper);
    pfw_.Deserialize(helper);
    pbw_.Deserialize(helper);

    helper.Deserialize(isBidirectional_);
    helper.Deserialize(wstride_);
  }

  void SetWindowSize(size_t window_size) { window_size_ = window_size; }
  void SetWindowShift(size_t window_shift) { window_shift_ = window_shift; }
  size_t GetWindowSize() const { return window_size_; }
  size_t GetWindowShift() const { return window_shift_; }

  void ResetStateBuffer(std::vector<void *> intermediate_states) {
    MMatrix *Buf_ufw = static_cast<MMatrix *>(intermediate_states[3]);
    MMatrix *Buf_cfw = static_cast<MMatrix *>(intermediate_states[4]);

    Buf_ufw->SetZero();
    Buf_cfw->SetZero();
  }

  void CopyStateBuffer(std::vector<void *> intermediate_states_from,
                       std::vector<void *> intermediate_states_to) {
    for (size_t i = 0; i < 5; i++) {
      MMatrix *m_from = static_cast<MMatrix *>(intermediate_states_from[i]);
      MMatrix *m_to = static_cast<MMatrix *>(intermediate_states_to[i]);
      *m_to = *m_from;
    }

    if (isBidirectional_) {
      for (size_t i = 5; i < 8; i++) {
        MMatrix *m_from = static_cast<MMatrix *>(intermediate_states_from[i]);
        MMatrix *m_to = static_cast<MMatrix *>(intermediate_states_to[i]);
        *m_to = *m_from;
      }
    }
  }
};



template<class WMatrix, class BMatrix, class MMatrix, class InputMatrix, class OutputMatrix>
void xnnBLSTMLayer<WMatrix, BMatrix, MMatrix, InputMatrix, OutputMatrix>::readKaldiLayerNnet1(
  std::istream &is /*,size_t nThread*/) {
  using namespace xnnKaldiUtility;

  bool binary = true; // always use binary format

  BaseFloat clip_gradient;
  BaseFloat learnrate_coef;

  std::string token;
  if ('<' == Peek(is, binary)) {
    ReadToken(is, binary, &token);
    if (token == "<LearnRateCoef>") {
      ReadBasicType(is, binary, &learnrate_coef);
      ExpectToken(is, binary, "<ClipGradient>");
      ReadBasicType(is, binary, &clip_gradient);
    } else if (token == "<ClipGradient>") {
      ReadBasicType(is, binary, &clip_gradient);
    } else {
      IDEC_ERROR << "expect <LearnRateCoef> or <ClipGradient> here";
    }
    //ExpectToken(is, binary, "<ClipGradient>");
    //ReadBasicType(is, binary, &clip_gradient);
  }

  // forward weights applied to the input
  int peekval = is.peek();
  if (peekval != 'F')
    IDEC_ERROR << "Only uncompressed matrix supported";

  //std::string token;
  ReadToken(is, binary, &token);
  if (token != "FM") {
    IDEC_ERROR << ": Expected token " << "FM" << ", got " << token;
  }

  int32 rows, cols;
  ReadBasicType(is, binary, &rows);  // throws on error.
  ReadBasicType(is, binary, &cols);  // throws on error.

  wstride_ = rows / 4;
  Wfw_.Resize(cols, rows);
  // weights
  for (int32 i = 0; i < rows; i++) {
    is.read(reinterpret_cast<char *>(Wfw_.Col(i)), sizeof(Real)*cols);
    if (is.fail()) IDEC_ERROR << "read forward weights applied to the input error";
  }

  // forward weights applied to the recurrent output
  peekval = is.peek();
  if (peekval != 'F')
    IDEC_ERROR << "Only uncompressed matrix supported";

  ReadToken(is, binary, &token);
  if (token != "FM") {
    IDEC_ERROR << ": Expected token " << "FM" << ", got " << token;
  }

  ReadBasicType(is, binary, &rows);  // throws on error.
  ReadBasicType(is, binary, &cols);  // throws on error.

  Rfw_.Resize(cols, rows);
  // weights
  for (int32 i = 0; i < rows; i++) {
    is.read(reinterpret_cast<char *>(Rfw_.Col(i)), sizeof(Real)*cols);
    if (is.fail()) IDEC_ERROR << "read forward weights applied to the recurrent output error";
  }

  // forward bias
  peekval = is.peek();
  if (peekval != 'F')
    IDEC_ERROR << "Only uncompressed vector supported";

  ReadToken(is, binary, &token);
  if (token != "FV")
    IDEC_ERROR << ": Expected token " << "FV" << ", got " << token;

  int32 size;
  ReadBasicType(is, binary, &size);  // throws on error.
  bfw_.Resize(size, 1);
  if (size > 0)
    is.read(reinterpret_cast<char *>(bfw_.Col(0)), sizeof(Real)*size);
  if (is.fail()) IDEC_ERROR << "read forward bias error";

  // forward peephole weights
  peekval = is.peek();
  if (peekval != 'F')
    IDEC_ERROR << "Only uncompressed vector supported";

  ReadToken(is, binary, &token);
  if (token != "FV")
    IDEC_ERROR << ": Expected token " << "FV" << ", got " << token;

  ReadBasicType(is, binary, &size);  // throws on error.
  pfw_.Resize(size, 1);
  if (size > 0)
    is.read(reinterpret_cast<char *>(pfw_.Col(0)), sizeof(Real)*size);
  if (is.fail()) 
	  IDEC_ERROR << "read forward peephole weights pi error";

  peekval = is.peek();
  if (peekval != 'F')
    IDEC_ERROR << "Only uncompressed vector supported";

  ReadToken(is, binary, &token);
  if (token != "FV")
    IDEC_ERROR << ": Expected token " << "FV" << ", got " << token;

  ReadBasicType(is, binary, &size);  // throws on error.
  pfw_.Resize(size, 2);
  if (size > 0)
    is.read(reinterpret_cast<char *>(pfw_.Col(1)), sizeof(Real)*size);
  if (is.fail()) IDEC_ERROR << "read forward peephole weights pf error";

  peekval = is.peek();
  if (peekval != 'F')
    IDEC_ERROR << "Only uncompressed vector supported";

  ReadToken(is, binary, &token);
  if (token != "FV")
    IDEC_ERROR << ": Expected token " << "FV" << ", got " << token;

  ReadBasicType(is, binary, &size);  // throws on error.
  pfw_.Resize(size, 3);
  if (size > 0)
    is.read(reinterpret_cast<char *>(pfw_.Col(2)), sizeof(Real)*size);
  if (is.fail()) IDEC_ERROR << "read forward peephole weights po error";

  if (isBidirectional_) {
    // backward weights applied to the input
    peekval = is.peek();
    if (peekval != 'F')
      IDEC_ERROR << "Only uncompressed matrix supported";

    ReadToken(is, binary, &token);
    if (token != "FM") {
      IDEC_ERROR << ": Expected token " << "FM" << ", got " << token;
    }

    ReadBasicType(is, binary, &rows);  // throws on error.
    ReadBasicType(is, binary, &cols);  // throws on error.

    Wbw_.Resize(cols, rows);
    // weights
    for (int32 i = 0; i < rows; i++) {
      is.read(reinterpret_cast<char *>(Wbw_.Col(i)), sizeof(Real)*cols);
      if (is.fail()) IDEC_ERROR << "read backward weights applied to the input error";
    }

    // backward weights applied to the recurrent output
    peekval = is.peek();
    if (peekval != 'F')
      IDEC_ERROR << "Only uncompressed matrix supported";

    ReadToken(is, binary, &token);
    if (token != "FM") {
      IDEC_ERROR << ": Expected token " << "FM" << ", got " << token;
    }

    ReadBasicType(is, binary, &rows);  // throws on error.
    ReadBasicType(is, binary, &cols);  // throws on error.

    Rbw_.Resize(cols, rows);
    // weights
    for (int32 i = 0; i < rows; i++) {
      is.read(reinterpret_cast<char *>(Rbw_.Col(i)), sizeof(Real)*cols);
      if (is.fail()) IDEC_ERROR << "read backward weights applied to the recurrent output error";
    }

    // backward bias
    peekval = is.peek();
    if (peekval != 'F')
      IDEC_ERROR << "Only uncompressed vector supported";

    ReadToken(is, binary, &token);
    if (token != "FV")
      IDEC_ERROR << ": Expected token " << "FV" << ", got " << token;

    ReadBasicType(is, binary, &size);  // throws on error.
    bbw_.Resize(size, 1);
    if (size > 0)
      is.read(reinterpret_cast<char *>(bbw_.Col(0)), sizeof(Real)*size);
    if (is.fail()) IDEC_ERROR << "read backward bias error";

    // backward peephole weights
    peekval = is.peek();
    if (peekval != 'F')
      IDEC_ERROR << "Only uncompressed vector supported";

    ReadToken(is, binary, &token);
    if (token != "FV")
      IDEC_ERROR << ": Expected token " << "FV" << ", got " << token;


    ReadBasicType(is, binary, &size);  // throws on error.
    pbw_.Resize(size, 1);
    if (size > 0)
      is.read(reinterpret_cast<char *>(pbw_.Col(0)), sizeof(Real)*size);
    if (is.fail()) IDEC_ERROR << "read backward peephole weights pi error";

    peekval = is.peek();
    if (peekval != 'F')
      IDEC_ERROR << "Only uncompressed vector supported";

    ReadToken(is, binary, &token);
    if (token != "FV")
      IDEC_ERROR << ": Expected token " << "FV" << ", got " << token;

    ReadBasicType(is, binary, &size);  // throws on error.
    pbw_.Resize(size, 2);
    if (size > 0)
      is.read(reinterpret_cast<char *>(pbw_.Col(1)), sizeof(Real)*size);
    if (is.fail()) IDEC_ERROR << "read backward peephole weights pf error";

    peekval = is.peek();
    if (peekval != 'F')
      IDEC_ERROR << "Only uncompressed vector supported";

    ReadToken(is, binary, &token);
    if (token != "FV")
      IDEC_ERROR << ": Expected token " << "FV" << ", got " << token;


    ReadBasicType(is, binary, &size);  // throws on error.
    pbw_.Resize(size, 3);
    if (size > 0)
      is.read(reinterpret_cast<char *>(pbw_.Col(2)), sizeof(Real)*size);
    if (is.fail()) IDEC_ERROR << "read backward peephole weights po error";
  }
}
};

#endif
