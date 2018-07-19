// matrix/kaldi-vector.cc

// Copyright 2009-2011  Microsoft Corporation;  Lukas Burget;
//                      Saarland University;   Go Vivace Inc.;  Ariya Rastrow;
//                      Petr Schwarz;  Yanmin Qian;  Jan Silovsky;
//                      Haihua Xu; Wei Shi


// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <string>
#include "kaldi/kaldi-vector.h"
#include "kaldi/kaldi-error.h"
#include "kaldi/io-funcs.h"


namespace idec {
namespace kaldi {

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4056 4305 4800 4267 4996 4756 4661)
#define __restrict__
#endif

template<typename Real, typename OtherReal>
Real VecVec(const VectorBase<Real> &ra,
            const VectorBase<OtherReal> &rb) {
  MatrixIndexT adim = ra.Dim();
  KALDI_ASSERT(adim == rb.Dim());
  const Real *a_data = ra.Data();
  const OtherReal *b_data = rb.Data();
  Real sum = 0.0;
  for (MatrixIndexT i = 0; i < adim; i++)
    sum += a_data[i]*b_data[i];
  return sum;
}

// instantiate the template above.
template
float VecVec<>(const VectorBase<float> &ra,
               const VectorBase<double> &rb);
template
double VecVec<>(const VectorBase<double> &ra,
                const VectorBase<float> &rb);

template<typename Real>
inline void Vector<Real>::Init(const MatrixIndexT dim) {
  if (dim == 0) {
    this->dim_ = 0;
    this->data_ = NULL;
    return;
  }
  MatrixIndexT size;
  void *data;
  void *free_data;

  size = dim * sizeof(Real);

  if ((data = IDEC_MEMALIGN(16, size, &free_data)) != NULL) {
    this->data_ = static_cast<Real *> (data);
    this->dim_ = dim;
  } else {
    throw std::bad_alloc();
  }
}


template<typename Real>
void Vector<Real>::Resize(const MatrixIndexT dim,
                          MatrixResizeType resize_type) {

  // the next block uses recursion to handle what we have to do if
  // resize_type == kCopyData.
  if (resize_type == kCopyData) {
    if (this->data_ == NULL
        || dim == 0) resize_type = kSetZero;  // nothing to copy.
    else if (this->dim_ == dim) { return; } // nothing to do.
    else {
      // set tmp to a vector of the desired size.
      Vector<Real> tmp(dim, kUndefined);
      if (dim > this->dim_) {
        memcpy(tmp.data_, this->data_, sizeof(Real)*this->dim_);
        memset(tmp.data_+this->dim_, 0, sizeof(Real)*(dim-this->dim_));
      } else {
        memcpy(tmp.data_, this->data_, sizeof(Real)*dim);
      }
      tmp.Swap(this);
      // and now let tmp go out of scope, deleting what was in *this.
      return;
    }
  }
  // At this point, resize_type == kSetZero or kUndefined.

  if (this->data_ != NULL) {
    if (this->dim_ == dim) {
      if (resize_type == kSetZero) this->SetZero();
      return;
    } else {
      Destroy();
    }
  }
  Init(dim);
  if (resize_type == kSetZero) this->SetZero();
}

template<typename Real>
void Vector<Real>::Swap(Vector<Real> *other) {
  std::swap(this->data_, other->data_);
  std::swap(this->dim_, other->dim_);
}


/// Copy data from another vector
template<typename Real>
void VectorBase<Real>::CopyFromVec(const VectorBase<Real> &v) {
  KALDI_ASSERT(Dim() == v.Dim());
  if (data_ != v.data_) {
    std::memcpy(this->data_, v.data_, dim_ * sizeof(Real));
  }
}


/// Load data into the vector
template<typename Real>
void VectorBase<Real>::CopyFromPtr(const Real *data, MatrixIndexT sz) {
  KALDI_ASSERT(dim_ == sz);
  std::memcpy(this->data_, data, Dim() * sizeof(Real));
}

template<typename Real>
template<typename OtherReal>
void VectorBase<Real>::CopyFromVec(const VectorBase<OtherReal> &other) {
  KALDI_ASSERT(dim_ == other.Dim());
  const OtherReal *other_ptr = other.Data();
  for (MatrixIndexT i = 0; i < dim_; i++) { data_[i] = other_ptr[i]; }
}

template void VectorBase<float>::CopyFromVec(const VectorBase<double> &other);
template void VectorBase<double>::CopyFromVec(const VectorBase<float> &other);

// Remove element from the vector. The vector is non reallocated
template<typename Real>
void Vector<Real>::RemoveElement(MatrixIndexT i) {
  KALDI_ASSERT(i <  this->dim_ && "Access out of vector");
  for (MatrixIndexT j = i + 1; j <  this->dim_; j++)
    this->data_[j-1] =  this->data_[j];
  this->dim_--;
}


/// Deallocates memory and sets object to empty vector.
template<typename Real>
void Vector<Real>::Destroy() {
  /// we need to free the data block if it was defined
  if (this->data_ != NULL)
    IDEC_MEMALIGN_FREE(this->data_);
  this->data_ = NULL;
  this->dim_ = 0;
}

template<typename Real>
void VectorBase<Real>::SetZero() {
  std::memset(data_, 0, dim_ * sizeof(Real));
}

template<typename Real>
bool VectorBase<Real>::IsZero(Real cutoff) const {
  Real abs_max = 0.0;
  for (MatrixIndexT i = 0; i < Dim(); i++)
    abs_max = std::max(std::abs((double)data_[i]),(double)abs_max);
  return (abs_max <= cutoff);
}


template<typename Real>
void VectorBase<Real>::Set(Real f) {
  // Why not use memset here?
  for (MatrixIndexT i = 0; i < dim_; i++) { data_[i] = f; }
}

template<typename Real>
template<typename OtherReal>
void VectorBase<Real>::AddVec(const Real alpha,
                              const VectorBase<OtherReal> &v) {
  KALDI_ASSERT(dim_ == v.dim_);
  // remove __restrict__ if it causes compilation problems.
  register Real *__restrict__ data = data_;
  register OtherReal *__restrict__ other_data = v.data_;
  MatrixIndexT dim = dim_;
  if (alpha != 1.0)
    for (MatrixIndexT i = 0; i < dim; i++)
      data[i] += alpha*other_data[i];
  else
    for (MatrixIndexT i = 0; i < dim; i++)
      data[i] += other_data[i];
}

template
void VectorBase<float>::AddVec(const float alpha, const VectorBase<double> &v);
template
void VectorBase<double>::AddVec(const double alpha,
                                const VectorBase<float> &v);


template<typename Real>
void VectorBase<Real>::Read(std::istream &is,  bool binary, bool add) {
  if (add) {
    Vector<Real> tmp(Dim());
    tmp.Read(is, binary, false);  // read without adding.
    if (this->Dim() != tmp.Dim()) {
      KALDI_ERR << "VectorBase::Read, size mismatch " << this->Dim()
                <<" vs. "<<tmp.Dim();
    }
    this->AddVec(1.0, tmp);
    return;
  } // now assume add == false.

  //  In order to avoid rewriting this, we just declare a Vector and
  // use it to read the data, then copy.
  Vector<Real> tmp;
  tmp.Read(is, binary, false);
  if (tmp.Dim() != Dim())
    KALDI_ERR << "VectorBase<Real>::Read, size mismatch "
              << Dim() << " vs. " << tmp.Dim();
  CopyFromVec(tmp);
}


template<typename Real>
void Vector<Real>::Read(std::istream &is,  bool binary, bool add) {
  if (add) {
    Vector<Real> tmp(this->Dim());
    tmp.Read(is, binary, false);  // read without adding.
    if (this->Dim() == 0) this->Resize(tmp.Dim());
    if (this->Dim() != tmp.Dim()) {
      KALDI_ERR << "Vector<Real>::Read, adding but dimensions mismatch "
                << this->Dim() << " vs. " << tmp.Dim();
    }
    this->AddVec(1.0, tmp);
    return;
  } // now assume add == false.

  std::ostringstream specific_error;
  MatrixIndexT pos_at_start = is.tellg();

  if (binary) {
    int peekval = Peek(is, binary);
    const char *my_token =  (sizeof(Real) == 4 ? "FV" : "DV");
    char other_token_start = (sizeof(Real) == 4 ? 'D' : 'F');
    if (peekval ==
        other_token_start) {  // need to instantiate the other type to read it.
      typedef typename OtherReal<Real>::Real
      OtherType;  // if Real == float, OtherType == double, and vice versa.
      Vector<OtherType> other(this->Dim());
      other.Read(is, binary, false);  // add is false at this point.
      if (this->Dim() != other.Dim()) this->Resize(other.Dim());
      this->CopyFromVec(other);
      return;
    }
    std::string token;
    ReadToken(is, binary, &token);
    if (token != my_token) {
      specific_error << ": Expected token " << my_token << ", got " << token;
      goto bad;
    }
    int32 size;
    ReadBasicType(is, binary, &size);  // throws on error.
    if ((MatrixIndexT)size != this->Dim())  this->Resize(size);
    if (size > 0)
      is.read(reinterpret_cast<char *>(this->data_), sizeof(Real)*size);
    if (is.fail()) {
      specific_error << "Error reading vector data (binary mode); truncated "
                     "stream? (size = " << size << ")";
      goto bad;
    }
    return;
  } else {  // Text mode reading; format is " [ 1.1 2.0 3.4 ]\n"
    std::string s;
    is >> s;
    // if ((s.compare("DV") == 0) || (s.compare("FV") == 0)) {  // Back compatibility.
    //  is >> s;  // get dimension
    //  is >> s;  // get "["
    // }
    if (is.fail()) { specific_error << "EOF while trying to read vector."; goto bad; }
    if (s.compare("[]") == 0) { Resize(0); return; } // tolerate this variant.
    if (s.compare("[")) { specific_error << "Expected \"[\" but got " << s; goto bad; }
    std::vector<Real> data;
    while (1) {
      int i = is.peek();
      if (i == '-' || (i >= '0' && i <= '9')) {  // common cases first.
        Real r;
        is >> r;
        if (is.fail()) { specific_error << "Failed to read number."; goto bad; }
        if (! std::isspace(is.peek()) && is.peek() != ']') {
          specific_error << "Expected whitespace after number.";
          goto bad;
        }
        data.push_back(r);
        // But don't eat whitespace... we want to check that it's not newlines
        // which would be valid only for a matrix.
      } else if (i == ' ' || i == '\t') {
        is.get();
      } else if (i == ']') {
        is.get();  // eat the ']'
        this->Resize(data.size());
        for (size_t j = 0; j < data.size(); j++)
          this->data_[j] = data[j];
        i = is.peek();
        if (static_cast<char>(i) == '\r') {
          is.get();
          is.get();  // get \r\n (must eat what we wrote)
        } else if (static_cast<char>(i) == '\n') { is.get(); } // get \n (must eat what we wrote)
        if (is.fail()) {
          KALDI_WARN << "After end of vector data, read error.";
          // we got the data we needed, so just warn for this error.
        }
        return;  // success.
      } else if (i == -1) {
        specific_error << "EOF while reading vector data.";
        goto bad;
      } else if (i == '\n' || i == '\r') {
        specific_error << "Newline found while reading vector (maybe it's a matrix?)";
        goto bad;
      } else {
        is >> s;  // read string.
        if (!IDEC_STRCASECMP(s.c_str(), "inf") ||
            !IDEC_STRCASECMP(s.c_str(), "infinity")) {
          data.push_back(std::numeric_limits<Real>::infinity());
          KALDI_WARN << "Reading infinite value into vector.";
        } else if (!IDEC_STRCASECMP(s.c_str(), "nan")) {
          data.push_back(std::numeric_limits<Real>::quiet_NaN());
          KALDI_WARN << "Reading NaN value into vector.";
        } else {
          specific_error << "Expecting numeric vector data, got " << s;
          goto  bad;
        }
      }
    }
  }
  // we never reach this line (the while loop returns directly).
bad:
  KALDI_ERR << "Failed to read vector from stream.  " << specific_error.str()
            << " File position at start is "
            << pos_at_start<<", currently "<<is.tellg();
}


template<typename Real>
void VectorBase<Real>::Write(std::ostream &os, bool binary) const {
  if (!os.good()) {
    KALDI_ERR << "Failed to write vector to stream: stream not good";
  }
  if (binary) {
    std::string my_token = (sizeof(Real) == 4 ? "FV" : "DV");
    WriteToken(os, binary, my_token);

    int32 size = Dim();  // make the size 32-bit on disk.
    KALDI_ASSERT(Dim() == (MatrixIndexT) size);
    WriteBasicType(os, binary, size);
    os.write(reinterpret_cast<const char *>(Data()), sizeof(Real) * size);
  } else {
    os << " [ ";
    for (MatrixIndexT i = 0; i < Dim(); i++)
      os << (*this)(i) << " ";
    os << "]\n";
  }
  if (!os.good())
    KALDI_ERR << "Failed to write vector to stream";
}

template class Vector<float>;
template class Vector<double>;
template class VectorBase<float>;
template class VectorBase<double>;

}  // namespace kaldi
}

