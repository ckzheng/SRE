#ifndef _NEW_VECTOR_XX_H_
#define _NEW_VECTOR_XX_H_

#include <math.h>
#include <vector>
#include <Dense>
#include <Cholesky>
#include <src/Eigenvalues/EigenSolver.h>
#include <Eigenvalues>
#include <complex>
#include <algorithm>
#include "base/log_message.h"

class DoubleVector {
 public:
  explicit DoubleVector(unsigned long size = 0) {
    this->v_.noalias() = Eigen::VectorXd(size);
    this->v_.setZero();
  }

  DoubleVector(const Eigen::VectorXd &v) {
    this->v_.noalias() = v;
  }

  DoubleVector(const DoubleVector &v) {
    this->v_.noalias() = v();
  }

  DoubleVector &operator=(const DoubleVector &v) {
    v_.noalias() = v();
    return *this;
  }

  bool operator==(const DoubleVector &v) const {
    return this->v_ == v();
  }

  bool operator!=(const DoubleVector &v) const { return !(*this == v); }

  ~DoubleVector() {}

  void Resize(const unsigned long size) {
    v_.resize(size);
    v_.setZero();
  }

  int Size() const {
    return v_.size();
  }

  void ApplyExp() {
    v_ = v_.array().exp();
  }

  double LogSumExp(double  prune = -1.0) const {
    //const double kLogZeroDouble = -std::numeric_limits<double>::infinity();
    //double sum = kLogZeroDouble;
    double sum = -std::numeric_limits<double>::infinity();
    const double dbl_epsilon = 2.2204460492503131e-16;
    //const double kMinLogDiffDouble = log(dbl_epsilon);
    double max_elem = this->Max();
    //double cutoff = max_elem + kMinLogDiffDouble;
    double cutoff = max_elem + log(dbl_epsilon);
    if (prune > 0.0 && max_elem - prune > cutoff) {
      // explicit pruning...
      cutoff = max_elem - prune;
    }

    double sum_relto_max_elem = 0.0;

    for (int i = 0; i < this->Size(); i++) {
      double f = v_(i);
      if (f >= cutoff) {
        sum_relto_max_elem += exp(f - max_elem);
      }
    }
    return max_elem + log(sum_relto_max_elem);
  }

  void ApplyPow(double power) {
    v_ = v_.array().pow(power);
  }

  int ApplyFloor(double floor) {
    int num_floored = 0;
    for (int i = 0; i < v_.size(); ++i) {
      if (v_(i) < floor) {
        ++num_floored;
        v_(i) = floor;
      }
    }
    return num_floored;
  }

  double Max() const {
    return v_.maxCoeff();
  }

  double Min() const {
    return v_.minCoeff();
  }


  double Max(int *index_out) const {
    if (Size() == 0) {
      idec::IDEC_ERROR << "Empty vector";
    }
    double ans = v_.maxCoeff(index_out);
    return ans;
  }

  void Scale(double s) {
    v_ *= s;
  }

  double SumLog() const {
    double sum_log = 0.0;
    double prod = 1.0;
    for (int i = 0; i < Size(); i++) {
      prod *= v_(i);
      // Possible future work (arnab): change these magic values to pre-defined
      // constants
      if (prod < 1.0e-10 || prod > 1.0e+10) {
        sum_log += log(prod);
        prod = 1.0;
      }
    }
    if (prod != 1.0)
      sum_log += log(prod);
    return sum_log;
  }


  double ApplySoftMax() {
    double max = this->Max(), sum = 0.0;
    for (int i = 0; i < v_.size(); i++) {
      sum += (v_(i) = exp(v_(i) - max));
    }
    this->Scale(1.0 / sum);
    return max + log(sum);
  }

  double Norm(double p) const {
    idec::IDEC_ASSERT(p >= 0.0);
    double sum = 0.0;
    int dim = this->Size();
    if (p == 0.0) {
      for (int i = 0; i < dim; i++)
        if (v_(i) != 0.0) sum += 1.0;
      return sum;
    } else if (p == 1.0) {
      for (int i = 0; i < dim; i++)
        sum += std::abs(v_(i));
      return sum;
    } else if (p == 2.0) {
      for (int i = 0; i < dim; i++)
        sum += v_(i) * v_(i);
      return std::sqrt(sum);
    } else if (p == std::numeric_limits<double>::infinity()) {
      for (int i = 0; i < dim; i++)
        sum = std::max(sum, std::abs(v_(i)));
      return sum;
    } else {
      double tmp;
      bool ok = true;
      for (int i = 0; i < dim; i++) {
        tmp = pow(std::abs(v_(i)), p);
        if (tmp == HUGE_VAL) // HUGE_VAL is what pow returns on error.
          ok = false;
        sum += tmp;
      }
      tmp = pow(sum, static_cast<double>(1.0 / p));
      idec::IDEC_ASSERT(tmp != HUGE_VAL); // should not happen here.
      if (ok) {
        return tmp;
      } else {
        double maximum = this->Max(), minimum = this->Min(),
               max_abs = std::max(maximum, -minimum);
        idec::IDEC_ASSERT(max_abs > 0); // Or should not have reached here.
        DoubleVector tmp(*this);
        tmp.Scale(1.0 / max_abs);
        return tmp.Norm(p) * max_abs;
      }
    }
  }

  double Sum() const {
    return v_.sum();
  }

  void ApplyLog() {
    v_ = v_.array().log();
  }

  void InvertElements() {
    v_= v_.cwiseInverse();
  }

  /// Transposes this constant matrix into a new matrix
  /// @return the new matrix
  ///
  DoubleVector Transpose() const {
    const DoubleVector &tmp = *this;
    return tmp.Transpose();
  }

  /// Transposes this matrix
  /// @return this matrix
  ///
  DoubleVector &Transpose() {
    const Eigen::VectorXd &tmp = v_.transpose();
    v_.noalias() = tmp;
    return *this;
  }

  /// Inverts this matrix
  /// @return this matrix
  ///
  DoubleVector &Invert() {
    v_ = v_.inverse();
    return *this;
  }

  DoubleVector Invert() const {
    const DoubleVector &tmp = *this;
    return tmp.Invert();
  }

  const Eigen::VectorXd &operator()() const {
    return v_;
  }

  Eigen::VectorXd &operator()() {
    return v_;
  }

  double operator()(int idx) const {
    return v_(idx);
  }

  double &operator()(int idx) {
    return v_(idx);
  }

  double operator*(const DoubleVector &v) const {
    Eigen::MatrixXd d = v_.transpose() * v();
    return d(0, 0);
  }

  DoubleVector &operator*=(double c) {
    v_ *= c;
    return *this;
  }

  DoubleVector operator*(double v) const {
    DoubleVector tmp = *this;
    tmp *= v;
    return tmp;
  }

  int Rows() const {
    return v_.rows();
  }

  int Cols() const {
    return v_.cols();
  }

  DoubleVector operator+(const DoubleVector &m) const {
    idec::IDEC_ASSERT((v_.cols() == m.Cols()) && (v_.rows() == m.Rows()));
    DoubleVector tmp(*this);
    tmp += m;
    return tmp;
  }

  DoubleVector &operator+ (const double val) {
    Eigen::VectorXd v(v_.size());
    v.setConstant(val);
    v_.noalias() += v;

    return *this;
  }

  DoubleVector operator+ (const double val) const {
    Eigen::VectorXd v(v_.size());
    v.setConstant(val);
    v.noalias() += v_;
    return v;
  }

  DoubleVector &operator+=(const double val) {
    Eigen::VectorXd v(v_.size());
    v.setConstant(val);
    v_.noalias() += v;
    return *this;
  }

  DoubleVector &operator+=(const DoubleVector &m) {
    idec::IDEC_ASSERT((v_.cols() == m.Cols()) && (v_.rows() == m.Rows()));
    v_.noalias() += m();
    return *this;
  }

  DoubleVector operator-(const DoubleVector &m) const {
    DoubleVector tmp(*this);
    tmp -= m;
    return tmp;
  }

  DoubleVector &operator-=(const DoubleVector &m) {
    idec::IDEC_ASSERT((v_.cols() == m.Cols()) && (v_.rows() == m.Rows()));
    v_.noalias() -= m();
    return *this;
  }

  void SetAllValues(double v) {
    v_.setConstant(v);
  }

  void Erase(int pos) {
    idec::IDEC_ASSERT((pos < Size()) && (Size() >= 1));
    Eigen::VectorXd tmp(Size() - 1);
    for (int i = 0, j = 0; i < Size(); ++i) {
      if (i == pos) {
        continue;
      }
      tmp(j++) = v_(i);
    }
    v_ = tmp;
  }

  double *Data() {
    return v_.data();
  }

  const double *Data() const {
    return v_.data();
  }

  Eigen::VectorXd *Pointer() {
    return &v_;
  }

  const Eigen::VectorXd *Pointer() const {
    return &v_;
  }

 private:
  Eigen::VectorXd v_;
};

#endif