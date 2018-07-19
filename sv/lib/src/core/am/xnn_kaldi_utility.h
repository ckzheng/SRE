// xnn-evaluator/xnn-kaldiutility.h

// Copyright 2015 Alibaba-inc  [zhijie.yzj]

#ifndef XNN_KALDIUTILITY_H_
#define XNN_KALDIUTILITY_H_

#include <assert.h>
#include "base/log_message.h"
#include <limits>
#include <ctype.h>
#include <cctype>

namespace idec {

////////////////////////////////////////////////////////////////////////
//
// Functions to load Kaldi model format
//
////////////////////////////////////////////////////////////////////////
namespace xnnKaldiUtility {

inline void CheckToken(const char *token) {
  assert(*token != '\0');  // check it's nonempty.
  while (*token != '\0') {
    assert(!::isspace(*token));
    token++;
  }
}

inline void ReadToken(std::istream &is, bool binary, std::string *str) {
  assert(str != NULL);
  if (!binary) is >> std::ws;  // consume whitespace.
  is >> *str;
  if (is.fail()) {
    IDEC_ERROR << "ReadToken, failed to read token at file position "
               << is.tellg();
  }
  if (!isspace(is.peek())) {
    IDEC_ERROR << "ReadToken, expected space after token, saw instead "
               << static_cast<char>(is.peek())
               << ", at file position " << is.tellg();
  }
  is.get();  // consume the space.
}

inline void ExpectToken(std::istream &is, bool binary, const char *token) {
  int pos_at_start = (int)is.tellg();
  assert(token != NULL);
  CheckToken(token);  // make sure it's valid (can be read back)
  if (!binary) is >> std::ws;  // consume whitespace.
  std::string str;
  is >> str;
  is.get();  // consume the space.
  if (is.fail()) {
    IDEC_ERROR << "Failed to read token [started at file position "
               << pos_at_start << "], expected " << token;
  }
  if (strcmp(str.c_str(), token) != 0) {
    IDEC_ERROR << "Expected token \"" << token << "\", got instead \""
               << str << "\".";
  }
}

inline void ExpectToken(std::istream &is, bool binary,
                        const std::string &token) {
  ExpectToken(is, binary, token.c_str());
}

// This is like ExpectToken but for two tokens, and it
// will either accept token1 and then token2, or just token2.
// This is useful in Read functions where the first token
// may already have been consumed.
inline static void ExpectOneOrTwoTokens(std::istream &is, bool binary,
                                        const std::string &token1,
                                        const std::string &token2) {
  assert(token1 != token2);
  std::string temp;
  ReadToken(is, binary, &temp);
  if (temp == token1) {
    ExpectToken(is, binary, token2);
  } else {
    if (temp != token2) {
      IDEC_ERROR << "Expecting token " << token1 << " or " << token2
                 << " but got " << temp;
    }
  }
}

// Template that covers integers.
template<class T> inline void ReadBasicType(std::istream &is,
    bool binary, T *t) {
  //KALDI_PARANOID_ASSERT(t != NULL);
  // Compile time assertion that this is not called with a wrong type.
  //IDEC_ASSERT_IS_INTEGER_TYPE(T);
  if (binary) {
    int len_c_in = is.get();
    if (len_c_in == -1)
      IDEC_ERROR << "ReadBasicType: encountered end of stream.";
    char len_c = static_cast<char>(len_c_in), len_c_expected
                 = (std::numeric_limits<T>::is_signed ? 1 : -1)
                   * static_cast<char>(sizeof(*t));

    if (len_c != len_c_expected) {
      IDEC_ERROR << "ReadBasicType: did not get expected integer type, "
                 << static_cast<int>(len_c)
                 << " vs. " << static_cast<int>(len_c_expected)
                 << ".  You can change this code to successfully"
                 << " read it later, if needed.";
      // insert code here to read "wrong" type.  Might have a switch statement.
    }
    is.read(reinterpret_cast<char *>(t), sizeof(*t));
  } else {
    if (sizeof(*t) == 1) {
      int16 i;
      is >> i;
      *t = i;
    } else {
      is >> *t;
    }
  }
  if (is.fail()) {
    IDEC_ERROR << "Read failure in ReadBasicType, file position is "
               << is.tellg() << ", next char is " << is.peek();
  }
}

inline std::string CharToString(const char &c) {
  char buf[20];
  if (std::isprint(c))
    sprintf(buf, "\'%c\'", c);
  else
    sprintf(buf, "[character %d]", (int)c);
  return (std::string) buf;
}

template<>
inline void ReadBasicType<bool>(std::istream &is, bool binary, bool *b) {
  IDEC_ASSERT(b != NULL);
  if (!binary) is >> std::ws;  // eat up whitespace.
  char c = is.peek();
  if (c == 'T') {
    *b = true;
    is.get();
  } else if (c == 'F') {
    *b = false;
    is.get();
  } else {
    IDEC_ERROR << "Read failure in ReadBasicType<bool>, file position is "
               << is.tellg() << ", next char is " << CharToString(c);
  }
}

template<class T> inline void ReadIntegerVector(std::istream &is, bool binary, std::vector<T> *v) {
  //IDEC_ASSERT_IS_INTEGER_TYPE(T);
  assert(v != NULL);
  if (binary) {
    int sz = is.peek();
    if (sz == sizeof(T)) {
      is.get();
    } else { // this is currently just a check.
      IDEC_ERROR << "ReadIntegerVector: expected to see type of size "
                 << sizeof(T) << ", saw instead " << sz << ", at file position " << is.tellg();
    }

    int32 vecsz;
    is.read(reinterpret_cast<char *>(&vecsz), sizeof(vecsz));
    if (is.fail() || vecsz < 0) goto bad;
    v->resize(vecsz);
    if (vecsz > 0) {
      is.read(reinterpret_cast<char *>(&((*v)[0])), sizeof(T)*vecsz);
    }
  } else {
    std::vector<T> tmp_v;  // use temporary so v doesn't use extra memory
    // due to resizing.
    is >> std::ws;
    if (is.peek() != static_cast<int>('[')) {
      IDEC_ERROR << "ReadIntegerVector: expected to see [, saw "
                 << is.peek() << ", at file position " << is.tellg();
    }
    is.get();  // consume the '['.
    is >> std::ws;  // consume whitespace.
    while (is.peek() != static_cast<int>(']')) {
      if (sizeof(T) == 1) {  // read/write chars as numbers.
        int16 next_t;
        is >> next_t >> std::ws;
        if (is.fail()) goto bad;
        else
          tmp_v.push_back((T)next_t);
      } else {
        T next_t;
        is >> next_t >> std::ws;
        if (is.fail()) goto bad;
        else
          tmp_v.push_back(next_t);
      }
    }
    is.get();  // get the final ']'.
    *v = tmp_v;  // could use std::swap to use less temporary memory, but this
    // uses less permanent memory.
  }
  if (!is.fail()) return;
bad:
  IDEC_ERROR << "ReadIntegerVector: read failure at file position " << is.tellg();
}

inline void ReadSpliceComponent(std::istream &is, bool binary) {
  int32 input_dim_, const_component_dim_;
  std::vector<int32> context_;
  ExpectOneOrTwoTokens(is, binary, "<SpliceComponent>", "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  std::string token;
  ReadToken(is, false, &token);
  if (token == "<LeftContext>") {
    int32 left_context = 0, right_context = 0;
    std::vector<int32> context;
    ReadBasicType(is, binary, &left_context);
    ExpectToken(is, binary, "<RightContext>");
    ReadBasicType(is, binary, &right_context);
    for (int32 i = -1 * left_context; i <= right_context; i++)
      context.push_back(i);
    context_ = context;
  } else  if (token == "<Context>") {
    ReadIntegerVector(is, binary, &context_);
  } else  {
    IDEC_ERROR << "Unknown token" << token << ", the model might be corrupted";
  }

  ExpectToken(is, binary, "<ConstComponentDim>");
  ReadBasicType(is, binary, &const_component_dim_);
  ExpectToken(is, binary, "</SpliceComponent>");
}

inline void SkipHead_AffineComponentPreconditionedOnline(std::istream &is,
    std::string &startTok, bool binary) {
  ExpectOneOrTwoTokens(is, binary, startTok.c_str(), "<LearningRate>");
  Real learning_rate;
  ReadBasicType(is, binary, &learning_rate);
}

inline void SkipTail_AffineComponentPreconditionedOnline(std::istream &is,
    bool binary) {
  std::string tok;
  int32 rank_in, rank_out,
        update_period; // useless parameters, we just skip them
  Real num_samples_history, alpha, max_change_per_sample;
  ReadToken(is, binary, &tok);
  if (tok == "<Rank>") {  // back-compatibility (temporary)
    ReadBasicType(is, binary, &rank_in);
    rank_out = rank_in;
  } else {
    assert(tok == "<RankIn>");
    ReadBasicType(is, binary, &rank_in);
    ExpectToken(is, binary, "<RankOut>");
    ReadBasicType(is, binary, &rank_out);
  }
  ReadToken(is, binary, &tok);
  if (tok == "<UpdatePeriod>") {
    ReadBasicType(is, binary, &update_period);
    ExpectToken(is, binary, "<NumSamplesHistory>");
  } else {
    update_period = 1;
    assert(tok == "<NumSamplesHistory>");
  }
  ReadBasicType(is, binary, &num_samples_history);
  ExpectToken(is, binary, "<Alpha>");
  ReadBasicType(is, binary, &alpha);
  ExpectToken(is, binary, "<MaxChangePerSample>");
  ReadBasicType(is, binary, &max_change_per_sample);
}

inline void SkipTail_AffineComponent(std::istream &is, bool binary) {
  std::string tok;
  bool is_gradient;
  ReadToken(is, binary, &tok);
  //if (tok == "<AvgInput>") { // discard the following.
  //    CuVector<BaseFloat> avg_input;
  //    avg_input.Read(is, binary);
  //    BaseFloat avg_input_count;
  //    ExpectToken(is, binary, "<AvgInputCount>");
  //    ReadBasicType(is, binary, &avg_input_count);
  //    ReadToken(is, binary, &tok);
  //}
  if (tok == "<IsGradient>") {
    ReadBasicType(is, binary, &is_gradient);
  } else {
    is_gradient = false;
  }
}

inline void ReadVector(std::istream &is, xnnFloatRuntimeMatrix &mat) {
  bool binary = true;
  int peekval = is.peek();
  const char *my_token = (sizeof(Real) == 4 ? "FV" : "DV");
  char other_token_start = (sizeof(Real) == 4 ? 'D' : 'F');
  if (peekval ==
      other_token_start) {  // need to instantiate the other type to read it.
    IDEC_ERROR << "Only uncompressed vector supported";
    //typedef typename OtherReal<Real>::Real OtherType;  // if Real == float, OtherType == double, and vice versa.
    //Vector<OtherType> other(this->Dim());
    //other.Read(is, binary, false);  // add is false at this point.
    //if (this->Dim() != other.Dim()) this->Resize(other.Dim());
    //this->CopyFromVec(other);
    //return;
  }

  std::string token;
  ReadToken(is, binary, &token);
  if (token != my_token) {
    IDEC_ERROR << ": Expected token " << my_token << ", got " << token;
    //goto bad;
  }

  int32 size;
  ReadBasicType(is, binary, &size);  // throws on error.
  //if ((int32)size != this->Dim())  this->Resize(size);
  //Real *garbage = new Real[size];
  mat.Resize(size, 1);
  if (size > 0)
    is.read(reinterpret_cast<char *>(mat.Col(0)), sizeof(Real)*size);
  if (is.fail()) {
    IDEC_ERROR << "Error reading vector data (binary mode); truncated "
               "stream? (size = " << size << ")";
    //goto bad;
    //delete[] garbage;
  }
  return;
}

inline int Peek(std::istream &is, bool binary) {
  if (!binary) is >> std::ws;  // eat up whitespace.
  return is.peek();
}
}
}

#endif
