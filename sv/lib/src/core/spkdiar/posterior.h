#ifndef _POSTERIOR_H_
#define _POSTERIOR_H_
#include <vector>
#include "full_gmm.h"
#include "frame_posterior.h"

using namespace std;

class  Posterior {
 public:
  explicit Posterior(int num_frames) {
    post = vector<FramePosterior>(num_frames);
  }

  Posterior() {

  }

  ~Posterior() {
    post.clear();
  }

  void Scale(double scale) {
    if (scale == 1.0) {
      return;
    }
    for (int i = 0; i < post.size(); i++) {
      if (scale == 0.0) {
        post[i].Clear();
      } else {
        for (int j = 0; j < post[i].Size(); j++)
          post[i][j].second *= scale;
      }
    }
  }

  double Total() {
    double sum =  0.0;
    unsigned int T = post.size();
    for (unsigned int t = 0; t < T; t++) {
      unsigned int I = post[t].Size();
      for (unsigned int i = 0; i < I; i++) {
        sum += post[t][i].second;
      }
    }
    return sum;
  }

  void UttPosterior(DoubleMatrix &feats, FullGmm &fgmm, vector<int>&gselect) {
    int feat_dim = feats.Cols();
    int frames_num = feats.Rows();
    DoubleVector framefeats;
    for (int i = 0; i < frames_num; i++) {
      framefeats = feats.Rowv(i);
      post[i].CalculatePosterior(fgmm, framefeats, gselect);
    }
  }

  int Size() const {
    return post.size();
  }

  void Resize(int num) {
    post.resize(num);
  }

  const FramePosterior &operator[](int ndx) const {
    return post[ndx];
  }


  FramePosterior &operator[](int ndx) {
    return post[ndx];
  }

 private:
  vector<FramePosterior> post;
};

#endif

