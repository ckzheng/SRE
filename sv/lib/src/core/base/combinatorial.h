#ifndef COMBINATORIAL_H
#define COMBINATORIAL_H

#include <vector>
#if __cplusplus >= 201103L
#include <stdlib.h>
#endif
#include <cstddef>
namespace idec {
// help function to generate the cartesian product of multiple sets
// http://www.martinbroadhurst.com/combinatorial-algorithms.html#cartesian-product

template<typename T>
class CartesianProduct {
 public:
  CartesianProduct(std::vector<std::vector<T> > &sets) :sets_(sets) {
    cur_pos_.resize(sets.size(), 0);
    cur_value_.resize(sets.size());
    for (size_t i = 0; i < sets.size(); i++) {
      cur_value_[i] = sets[i][cur_pos_[i]];
    }
    done_ = false;
  }

  bool Done() {
    return done_;
  }
  void Next() {
    unsigned int changed = 0;
    unsigned int finished = 0;


    for (size_t i = sets_.size() - 1; !changed && !finished; i--) {
      if (cur_pos_[i] < sets_[i].size() - 1) {
        /* Increment */
        cur_pos_[i]++;
        changed = 1;
      } else {
        /* Roll over */
        cur_pos_[i] = 0;
      }
      finished = i == 0;
    }

    done_ = (changed == 0);
    if (!done_) {
      for (size_t i = 0; i < sets_.size(); i++) {
        cur_value_[i] = sets_[i][cur_pos_[i]];
      }
    }
  }

  std::vector<T> &Value() {
    return cur_value_;
  }
  void Reset() {
    std::fill(cur_pos_.begin(), cur_pos_.end(), 0);
    done_ = false;
  }

  // the sets to be iterate
  std::vector<std::vector<T> > sets_;
  std::vector<T> cur_value_;
  std::vector<size_t> cur_pos_;
  bool done_;
};

}
#endif


