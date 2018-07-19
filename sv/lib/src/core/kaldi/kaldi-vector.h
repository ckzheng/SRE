// matrix/kaldi-vector.h

// Copyright 2009-2012   Ondrej Glembek;  Microsoft Corporation;  Lukas Burget;
//                       Saarland University (Author: Arnab Ghoshal);
//                       Ariya Rastrow;  Petr Schwarz;  Yanmin Qian;
//                       Karel Vesely;  Go Vivace Inc.;  Arnab Ghoshal
//                       Wei Shi;

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

#ifndef KALDI_MATRIX_KALDI_VECTOR_H_
#define KALDI_MATRIX_KALDI_VECTOR_H_ 1

#include "kaldi/kaldi-idec-common.h"
#include "kaldi/kaldi-error.h"


namespace idec {
namespace kaldi {
typedef int MatrixIndexT;
typedef unsigned int UnsignedMatrixIndexT;
typedef enum {
  kSetZero,
  kUndefined,
  kCopyData
} MatrixResizeType;

/// This class provides a way for switching between double and float types.
template<typename T> class OtherReal {};  // useful in reading+writing routines
// to switch double and float.
/// A specialized class for switching from float to double.
template<> class OtherReal < float > {
 public:
  typedef double Real;
};
/// A specialized class for switching from double to float.
template<> class OtherReal < double > {
 public:
  typedef float Real;
};

/// \addtogroup matrix_group
/// @{

///  Provides a vector abstraction class.
///  This class provides a way to work with vectors in kaldi.
///  It encapsulates basic operations and memory optimizations.
template<typename Real>
class VectorBase {
 public:
  /// Set vector to all zeros.
  void SetZero();

  /// Returns true if matrix is all zeros.
  bool IsZero(Real cutoff = 1.0e-06) const;     // replace magic number

  /// Set all members of a vector to a specified value.
  void Set(Real f);

  /// Set vector to random normally-distributed noise.
  void SetRandn();

  /// This function returns a random index into this vector,
  /// chosen with probability proportional to the corresponding
  /// element.  Requires that this->Min() >= 0 and this->Sum() > 0.
  MatrixIndexT RandCategorical() const;

  /// Returns the  dimension of the vector.
  inline MatrixIndexT Dim() const { return dim_; }

  /// Returns the size in memory of the vector, in bytes.
  inline MatrixIndexT SizeInBytes() const { return (dim_*sizeof(Real)); }

  /// Returns a pointer to the start of the vector's data.
  inline Real *Data() { return data_; }

  /// Returns a pointer to the start of the vector's data (const).
  inline const Real *Data() const { return data_; }

  /// Indexing  operator (const).
  inline Real operator() (MatrixIndexT i) const {
    KALDI_PARANOID_ASSERT(static_cast<UnsignedMatrixIndexT>(i) <
                          static_cast<UnsignedMatrixIndexT>(dim_));
    return *(data_ + i);
  }

  /// Indexing operator (non-const).
  inline Real &operator() (MatrixIndexT i) {
    KALDI_PARANOID_ASSERT(static_cast<UnsignedMatrixIndexT>(i) <
                          static_cast<UnsignedMatrixIndexT>(dim_));
    return *(data_ + i);
  }




  /// Copy data from another vector (must match own size).
  void CopyFromVec(const VectorBase<Real> &v);


  /// Copy data from another vector of different type (double vs. float)
  template<typename OtherReal>
  void CopyFromVec(const VectorBase<OtherReal> &v);


  /// Add vector : *this = *this + alpha * rv (with casting between floats and
  /// doubles)
  template<typename OtherReal>
  void AddVec(const Real alpha, const VectorBase<OtherReal> &v);

  /// Reads from C++ stream (option to add to existing contents).
  /// Throws exception on failure
  void Read(std::istream &in, bool binary, bool add = false);

  /// Writes to C++ stream (option to write in binary).
  void Write(std::ostream &Out, bool binary) const;

  friend class VectorBase<double>;
  friend class VectorBase<float>;

 protected:
  /// Destructor;  does not deallocate memory, this is handled by child classes.
  /// This destructor is protected so this object so this object can only be
  /// deleted via a child.
  ~VectorBase() {}

  /// Empty initializer, corresponds to vector of zero size.
  explicit VectorBase(): data_(NULL), dim_(0) {
    IDEC_ASSERT_IS_FLOATING_TYPE(Real);
  }

// Took this out since it is not currently used, and it is possible to create
// objects where the allocated memory is not the same size as dim_ : Arnab
//  /// Initializer from a pointer and a size; keeps the pointer internally
//  /// (ownership or non-ownership depends on the child class).
//  explicit VectorBase(Real* data, MatrixIndexT dim)
//      : data_(data), dim_(dim) {}

  // Arnab : made this protected since it is unsafe too.
  /// Load data into the vector: sz must match own size.
  void CopyFromPtr(const Real *Data, MatrixIndexT sz);

  /// data memory area
  Real *data_;
  /// dimension of vector
  MatrixIndexT dim_;
  //IDEC_DISALLOW_COPY_AND_ASSIGN(VectorBase);
}; // class VectorBase

/** @brief A class representing a vector.
 *
 *  This class provides a way to work with vectors in kaldi.
 *  It encapsulates basic operations and memory optimizations.  */
template<typename Real>
class Vector: public VectorBase<Real> {
 public:
  /// Constructor that takes no arguments.  Initializes to empty.
  Vector(): VectorBase<Real>() {}

  /// Constructor with specific size.  Sets to all-zero by default
  /// if set_zero == false, memory contents are undefined.
  explicit Vector(const MatrixIndexT s,
                  MatrixResizeType resize_type = kSetZero)
    : VectorBase<Real>() { Resize(s, resize_type); }

/// Copy constructor.  The need for this is controversial.
  Vector(const Vector<Real> &v) : VectorBase<Real>()  { //  (cannot be explicit)
    Resize(v.Dim(), kUndefined);
    this->CopyFromVec(v);
  }

  /// Copy-constructor from base-class, needed to copy from SubVector.
  explicit Vector(const VectorBase<Real> &v) : VectorBase<Real>() {
    Resize(v.Dim(), kUndefined);
    this->CopyFromVec(v);
  }

  /// Type conversion constructor.
  template<typename OtherReal>
  explicit Vector(const VectorBase<OtherReal> &v): VectorBase<Real>() {
    Resize(v.Dim(), kUndefined);
    this->CopyFromVec(v);
  }


  /// Swaps the contents of *this and *other.  Shallow swap.
  void Swap(Vector<Real> *other);

  /// Destructor.  Deallocates memory.
  ~Vector() { Destroy(); }

  /// Read function using C++ streams.  Can also add to existing contents
  /// of matrix.
  void Read(std::istream &in, bool binary, bool add = false);

  /// Set vector to a specified size (can be zero).
  /// The value of the new data depends on resize_type:
  ///   -if kSetZero, the new data will be zero
  ///   -if kUndefined, the new data will be undefined
  ///   -if kCopyData, the new data will be the same as the old data in any
  ///      shared positions, and zero elsewhere.
  /// This function takes time proportional to the number of data elements.
  void Resize(MatrixIndexT length, MatrixResizeType resize_type = kSetZero);

  /// Remove one element and shifts later elements down.
  void RemoveElement(MatrixIndexT i);

  /// Assignment operator, protected so it can only be used by std::vector
  Vector<Real> &operator = (const Vector<Real> &other) {
    Resize(other.Dim(), kUndefined);
    this->CopyFromVec(other);
    return *this;
  }

  /// Assignment operator that takes VectorBase.
  Vector<Real> &operator = (const VectorBase<Real> &other) {
    Resize(other.Dim(), kUndefined);
    this->CopyFromVec(other);
    return *this;
  }
 private:
  /// Init assumes the current contents of the class are invalid (i.e. junk or
  /// has already been freed), and it sets the vector to newly allocated memory
  /// with the specified dimension.  dim == 0 is acceptable.  The memory contents
  /// pointed to by data_ will be undefined.
  void Init(const MatrixIndexT dim);

  /// Destroy function, called internally.
  void Destroy();

};


/// @} end of "addtogroup matrix_group"
/// \addtogroup matrix_funcs_io
/// @{
/// Output to a C++ stream.  Non-binary by default (use Write for
/// binary output).
template<typename Real>
std::ostream &operator << (std::ostream &out, const VectorBase<Real> &v);

/// Input from a C++ stream.  Will automatically read text or
/// binary data from the stream.
template<typename Real>
std::istream &operator >> (std::istream &in, VectorBase<Real> &v);

/// Input from a C++ stream. Will automatically read text or
/// binary data from the stream.
template<typename Real>
std::istream &operator >> (std::istream &in, Vector<Real> &v);
/// @} end of \addtogroup matrix_funcs_io

/// \addtogroup matrix_funcs_scalar
/// @{


template<typename Real>
bool ApproxEqual(const VectorBase<Real> &a,
                 const VectorBase<Real> &b, Real tol = 0.01) {
  return a.ApproxEqual(b, tol);
}

template<typename Real>
inline void AssertEqual(VectorBase<Real> &a, VectorBase<Real> &b,
                        float tol = 0.01) {
  KALDI_ASSERT(a.ApproxEqual(b, tol));
}


/// Returns dot product between v1 and v2.
template<typename Real>
Real VecVec(const VectorBase<Real> &v1, const VectorBase<Real> &v2);

template<typename Real, typename OtherReal>
Real VecVec(const VectorBase<Real> &v1, const VectorBase<OtherReal> &v2);



}  // namespace kaldi

// we need to include the implementation
#include "kaldi/kaldi-vector-inl.h"

}

#endif  // KALDI_MATRIX_KALDI_VECTOR_H_

