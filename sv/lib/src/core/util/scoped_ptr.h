#ifndef UTIL_SCOPE_PTR__
#define UTIL_SCOPE_PTR__

#include <cstddef>
#include <cstdlib>

namespace idec {

template <class T> class Scoped_Ptr {
 public:
  explicit Scoped_Ptr(T *content = NULL) : c_(content) {}

  ~Scoped_Ptr() {
    if (c_!= NULL) {
      delete c_;
    }
  }
  T *Get() { return c_; }
  const T *Get() const { return c_; }

  T &operator*() { return *c_; }
  const T &operator*() const { return *c_; }

  T *operator->() { return c_; }
  const T *operator->() const { return c_; }

  T &operator[](std::size_t idx) { return c_[idx]; }
  const T &operator[](std::size_t idx) const { return c_[idx]; }

  void Reset(T *to = NULL) {
    Scoped_Ptr<T> other(c_);
    c_ = to;
  }

 private:
  T *c_;

  Scoped_Ptr(const Scoped_Ptr &);
  void operator=(const Scoped_Ptr &);
};

} // namespace idec

#endif // UTIL_SCOPE_PTR__
