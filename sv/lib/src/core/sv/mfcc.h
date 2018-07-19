#ifndef _MFCC_XX_H_
#define _MFCC_XX_H_

#include <string>
#include <fstream>
#include "new_matrix.h"
#include "base/log_message.h"

using namespace std;

class Mfcc {
 public:
  Mfcc() {
    this->dims_ = 60;
    this->frames_ = 0;
  };

  void LoadMfcc(const char *path, DoubleMatrix &feats) {
    ifstream ifs(path);
    if (!ifs) {
      idec::IDEC_ERROR << "fail to open file " << path;
    }

    feats.Resize(1427, 60);

    string wave_name;
    string begin_separator;
    string end_separator;
    ifs >> wave_name;
    ifs >> begin_separator;
    if (begin_separator == "[") {
      cout << "read " << path << " begin.." << endl;
    }
    float value;
    int count = 0;
    while (true) {
      ifs >> value;
      //feature_.push_back(value);
      feats(frames_, count) = value;
      count++;
      if (count == 60) {
        ++frames_;
        // cout << "read line " << frames_ << endl;
        if (frames_ == 1427) {
          break;
        }
        count = 0;
      }
    }
  }

  const float *Frame(int index) {
    return &feature_[0] + index*dims_;
  }

  const float *Feature() {
    return &feature_[0];
  }

  int Dims() {
    return dims_;
  }

  int Frames() {
    return frames_;
  }

 private:
  vector<float> feature_;
  int frames_;
  int dims_;
};

#endif
