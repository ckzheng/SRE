#ifndef _NEW_MATRIX_XX_H_
#define _NEW_MATRIX_XX_H_

#include <new>
#include <math.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <memory.h>
#include <cstdlib>
#include <vector>
#include <Dense>
#include <Cholesky>
#include <src/Eigenvalues/EigenSolver.h>
#include <Eigenvalues>
#include <complex>
#include <algorithm>
#include <cassert>
#include "new_vector.h"

using namespace std;

class DoubleMatrix {
 public:
  DoubleMatrix(unsigned long Rows = 0, unsigned long Cols = 0) {
    this->m_ = Eigen::MatrixXd(Rows, Cols);
    this->m_.setZero();
  }

  DoubleMatrix(const DoubleMatrix &m) {
    this->m_.noalias() = m();
  }

  DoubleMatrix(const Eigen::MatrixXd &m) {
    m_ = m;
  }

  DoubleMatrix(const DoubleVector &v) {
    idec::IDEC_ASSERT((v.Rows() == 1) || (v.Cols() == 1));
    if (v.Cols() == 1) {
      m_.resize(v.Rows(), 1 );
    } else {
      m_.resize(1, v.Cols());
    }
    int size = (v.Rows() == 1) ? v.Cols() : v.Rows();
    memcpy(m_.data(), v.Data(), size * sizeof(double));
  }

  DoubleMatrix &operator=(const DoubleMatrix &m) {
    m_.noalias() = m();
    return *this;
  }

  DoubleMatrix &MulElements(const DoubleMatrix &m) {
    m_ = m_.cwiseProduct(m());
    return *this;
  }

  void DiagElements(DoubleVector &v) {
    v() = m_.diagonal();
  }

  void DiagElementsAdd(DoubleVector &v) {
    m_.diagonal() = m_.diagonal().array() + v().array();
  }

  bool operator==(const DoubleMatrix &m) const {
    return this->m_ == m();
  }

  bool operator!=(const DoubleMatrix &m) const { return !(*this == m); }

  ~DoubleMatrix() {}

  unsigned long Cols() const { return m_.cols(); }

  unsigned long Rows() const { return m_.rows(); }

  void Resize(const unsigned long Rows, const unsigned long Cols) {
    m_.resize(Rows, Cols);
    m_.setZero();
  }

  /// Transposes this matrix
  /// @return this matrix
  ///
  DoubleMatrix &Transpose() {
    const Eigen::MatrixXd &tmp = m_.transpose();
    m_.noalias() = tmp;
    return *this;
  }

  /// Transposes this constant matrix into a new matrix
  /// @return the new matrix
  ///
  DoubleMatrix Transpose() const {
    return DoubleMatrix(m_.transpose());
  }

  /// Inverts this matrix
  /// @return this matrix
  ///
  DoubleMatrix &Invert() {
    m_ = m_.inverse();
    return *this;
  }

  DoubleMatrix Invert() const {
    DoubleMatrix tmp = *this;
    return tmp.Invert();
  }

  const Eigen::MatrixXd &operator()() const {
    return m_;
  }

  double operator()(int row, int col) const {
    return m_(row, col);
  }

  double &operator()(int row, int col) {
    return m_(row, col);
  }

  DoubleMatrix operator*(const DoubleMatrix &m) const {
    const Eigen::MatrixXd &d = m_ * m();
    return DoubleMatrix(d);
  }

  DoubleMatrix &operator*=(const DoubleMatrix &m) {
    (*this) = (*this)*m;
    return *this;
  }

  DoubleMatrix &operator*=(double v) {
    m_ *= v;
    return *this;
  }

  DoubleMatrix operator*(double v) const {
    //DoubleMatrix tmp = *this;
    //tmp *= v;
    //return tmp;
    return DoubleMatrix(m_*v);
  }

  DoubleVector operator*(const DoubleVector &v) const {
    return DoubleVector(m_ * v());
  }

  DoubleMatrix operator+(const DoubleMatrix &m) const {
    idec::IDEC_ASSERT((m_.cols() == m.Cols()) && (m_.rows() == m.Rows()));
    //DoubleMatrix tmp(*this);
    //tmp += m;
    //return tmp;
    return DoubleMatrix(m() + m_);
  }

  DoubleMatrix &operator+=(const DoubleMatrix &m) {
    idec::IDEC_ASSERT((m_.cols() == m.Cols()) && (m_.rows() == m.Rows()));
    m_.noalias() += m();
    return *this;
  }

  DoubleMatrix operator-(const DoubleMatrix &m) const {
    //DoubleMatrix tmp(*this);
    //tmp -= m;
    //return tmp;
    return DoubleMatrix(m_ - m());
  }

  DoubleMatrix &operator-=(const DoubleMatrix &m) {
    idec::IDEC_ASSERT((m_.cols() == m.Cols()) && (m_.rows() == m.Rows()));
    m_.noalias() -= m();
    return *this;
  }

  DoubleMatrix Crop(unsigned long row, unsigned long col, unsigned long n_rows,
                    unsigned long n_cols) const {
    const Eigen::MatrixXd &m = m_.block(row, col,n_rows, n_cols);
    return DoubleMatrix(m);
  }

  const DoubleMatrix Col(int i) const {
    const Eigen::MatrixXd &c = m_.col(i);
    return DoubleMatrix(c);
  }

  void Col(int i, const DoubleVector &v) {
    m_.col(i) = v();
  }

  void Row(int i, const DoubleVector &v) {
    m_.row(i) = v();
  }

  const DoubleVector Colv(int i) const {
    return static_cast<DoubleVector>(m_.col(i));
  }

  const DoubleVector Rowv(int i) const {
    return static_cast<DoubleVector>(m_.row(i));
  }

  const DoubleMatrix Row(int i) const {
    const Eigen::MatrixXd &v = m_.row(i);
    return DoubleMatrix(v);
  }

  void ApplyPow(double power) {
    m_.noalias() = m_.array().pow(power).matrix();
  }

  void SetAllValues(double v) {
    m_.setConstant(v);
  }

  void Scale(double scale) {
    m_ *= scale;
  }

  double LogPosDefDet() const {
    Eigen::MatrixXd L(m_.llt().matrixL());
    double ret = 0.0;
    for (int i = 0; i < L.rows(); ++i) {
      ret += std::log(L(i, 0));
    }
    return (2 * ret);
  }

  void CholeskyDecompose(DoubleMatrix& m) const {
	  Eigen::MatrixXd L(m_.llt().matrixL());
	  *m.Pointer() = L;	  
  }

  void ComputeEigenProblem( DoubleMatrix &eigenVect, long rank) const {
    // EP has to be a square matrix
    DoubleVector eigenVal;
    eigenVal.Resize(Rows());
    eigenVal.SetAllValues(0.0);
    this->ComputeEigenProblem(eigenVect, eigenVal, rank);
  }

  double Trace() const {
    return m_.trace();
  }

  double LogDet() const {
    return log(m_.determinant());
  }

  double MaxAbsEig() const {
    // Compute Eigen Decomposition
    Eigen::EigenSolver<Eigen::MatrixXd> es(m_);
    complex<double> lambda = es.eigenvalues()[0];
    Eigen::MatrixXcd V = es.eigenvectors();

    // get and check eigen values
    vector<double> EV;
    unsigned long imagPart = 0;
    unsigned long _vectSize = m_.cols();
    for (unsigned long i = 0; i < _vectSize; i++) {
      if (imag(es.eigenvalues()[i]) != 0) {
        imagPart++;
      }
      EV[i] = real(es.eigenvalues()[i]);
    }

    if (imagPart > 0)
      cout << "WARNING " << imagPart << " eigenvalues have an imaginary part" <<
           endl;
    sort(EV.begin(), EV.end());
    return std::max(EV[0], -EV[EV.size() - 1]);
  }

  void ComputeEigenProblem(DoubleMatrix &eigenVect, DoubleVector &eigenVal,
                           long rank) const {
    int verboseLevel = 3;
    if (verboseLevel > 2) {
      cout << "Compute Eigen Problem." << endl;
    }

    // Compute Eigen Decomposition
    Eigen::EigenSolver<Eigen::MatrixXd> es(m_);
    complex<double> lambda = es.eigenvalues()[0];
    Eigen::MatrixXcd V = es.eigenvectors();

    // get and check eigen values
    vector<double> EV;
    unsigned long imagPart = 0;
    unsigned long _vectSize = m_.cols();
    for (unsigned long i = 0; i < _vectSize; i++) {
      if (imag(es.eigenvalues()[i]) != 0) {
        imagPart++;
      }
      EV[i] = real(es.eigenvalues()[i]);
    }

    if (imagPart > 0)
      cout << "WARNING " << imagPart << " eigenvalues have an imaginary part" <<
           endl;

    eigenVal.SetAllValues(0.0);

    // Order the EigenValues
    sort(EV.begin(), EV.end());
    for (unsigned long k = 0; k < _vectSize; k++) {
      for (int j = 0; j < rank; j++) {
        eigenVect(k, j) = real(V(k, j));
      }
    }

    for (int j = 0; j < rank; j++) {
      eigenVal(j) = EV[j];
    }

    if (verboseLevel > 3) {
      cerr << "EigenValues" << endl;
      for (int i = 0; i < rank; i++) {
        cerr << eigenVal()(i, i) << endl;
      }
    }
  }

  void Floor(double val) {
    for (int i = 0; i < m_.rows(); ++i) {
      for (int j = 0; j < m_.cols(); ++j) {
        if (m_(i, j) < val) {
          m_(i, j) = val;
        }
      }
    }
  }

  int EigenValueFloor(double val) {
    DoubleMatrix eigenVect;
    DoubleVector eigenVal;
    long rank = m_.cols();
    ComputeEigenProblem(eigenVect, eigenVal, rank);
    int num_floored = eigenVal.ApplyFloor(val);
    DoubleMatrix eigenValueMatrix(m_.cols(), m_.cols());
    for (int i = 0; i < m_.cols(); ++i) {
      eigenValueMatrix(i, i) = eigenVal(i);
    }
    DoubleMatrix new_matirx = eigenVect * eigenValueMatrix * eigenVect.Transpose();
    m_ = new_matirx();
    return num_floored;
  }

  void EraseRow(unsigned int rowToRemove) {
    unsigned int numRows = m_.rows() - 1;
    unsigned int numCols = m_.cols();

    if (rowToRemove < numRows) {
      m_.block(rowToRemove, 0, numRows - rowToRemove,
               numCols) = m_.block(rowToRemove + 1, 0, numRows - rowToRemove, numCols);
    }
    m_.conservativeResize(numRows, numCols);
  }

  void EraseColumn(unsigned int colToRemove) {
    unsigned int numRows = m_.rows();
    unsigned int numCols = m_.cols() - 1;

    if (colToRemove < numCols) {
      m_.block(0, colToRemove, numRows, numCols - colToRemove) = m_.block(0,
          colToRemove + 1, numRows, numCols - colToRemove);
    }
    m_.conservativeResize(numRows, numCols);
  }

  void BlockAssign(unsigned int row_start, unsigned int nRows, unsigned int col_start, unsigned int nCols, const DoubleMatrix& mat) {
	  idec::IDEC_ASSERT((row_start + nRows < m_.rows()) && (col_start + nCols < m_.cols()));
	  idec::IDEC_ASSERT((nRows == mat.Rows()) && (nCols == mat.Cols()));
	  m_.block(row_start, col_start, nRows, nCols) = mat();
  }

  double *Data() {
    return m_.data();
  }

  Eigen::MatrixXd *Pointer() {
    return &m_;
  }

  const Eigen::MatrixXd *Pointer() const {
    return &m_;
  }

 private:
  Eigen::MatrixXd m_;
};

#endif