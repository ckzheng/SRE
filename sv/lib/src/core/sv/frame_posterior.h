#ifndef _FRAME_POSTERIOR_H_
#define _FRAME_POSTERIOR_H_

#include <cmath>
#include <vector>
#include "full_gmm.h"
#include "new_vector.h"
#include "base/log_message.h"

using namespace std;

struct CompareReverseSecond {
  bool operator() (const std::pair<int, double> &a,
                   const std::pair<int, double> &b) {
    return (a.second > b.second);
  }
};

class FramePosterior {
 public:
  FramePosterior() {}
  ~FramePosterior() {}
  //void CalculatePosterior(FullGmm &fgmm, const DoubleVector &frame, vector<int> &gselect);
  double VectorToPosteriorEntry(const DoubleVector &log_likes, int num_gselect,
                                double min_post);
  void Add(const std::pair<int, double> &pair) {
    post_entry.push_back(pair);
  }

  unsigned int Size() const {
    return post_entry.size();
  }

  void Clear() {
    post_entry.clear();
  }

  bool Empty() const {
    return post_entry.empty();
  }

  const vector<std::pair<int, double> > &Data() const {
    return post_entry;
  }

  std::pair<int, double> &operator[](int ndx) {
    return post_entry[ndx];
  }

  const std::pair<int, double> &operator[](int ndx) const {
    return post_entry[ndx];
  }

 private:
  vector<std::pair<int, double> > post_entry;
};

#endif
