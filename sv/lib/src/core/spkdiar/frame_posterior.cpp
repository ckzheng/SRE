#include "frame_posterior.h"
#include "full_gmm.h"

void FramePosterior::CalculatePosterior(const FullGmm &fgmm,
                                        const DoubleVector &frame,
                                        vector<int> &gselect) {
  double this_tot_loglike = 0;
  bool frame_ok = true;
  DoubleVector loglikes;
  loglikes.Resize(gselect.size());
  fgmm.LogLikelihoodsPreselect(frame, gselect, loglikes);
  this_tot_loglike += loglikes.ApplySoftMax();
  // now "loglikes" contains posteriors.
  if (fabs(loglikes.Sum() - 1.0) > 0.01) {
    frame_ok = false;
  } else {
    const int min_post = 20;
    if (min_post != 0.0) {
      int max_index = 0;
      loglikes.Max(&max_index);
      for (int i = 0; i < loglikes.Size(); i++)
        if (loglikes(i) < min_post)
          loglikes(i) = 0.0;
      double sum = loglikes.Sum();
      if (sum == 0.0) {
        loglikes(max_index) = 1.0;
      } else {
        loglikes.Scale(1.0 / sum);
      }
    }

    for (int i = 0; i < loglikes.Size(); i++) {
      if (loglikes(i) != 0.0) {
        post_entry.push_back(std::make_pair(gselect[i], loglikes(i)));
      }
    }
  }
}

double FramePosterior::VectorToPosteriorEntry(const DoubleVector &log_likes,
    int  num_gselect,
    double  min_post) {
  if (num_gselect < 0 || min_post <= 0 || min_post > 1.0) {
    idec::IDEC_ERROR << "input parameter wrong!";
  }
  // we name num_gauss assuming each entry in log_likes represents a Gaussian;
  // it doesn't matter if they don't.
  int num_gauss = log_likes.Size();
  if (num_gauss <= 0) {
    idec::IDEC_ERROR << "mixture num less than 0!";
  }

  if (num_gselect > num_gauss) {
    num_gselect = num_gauss;
  }
  DoubleVector log_likes_normalized(log_likes);
  double ans = log_likes_normalized.ApplySoftMax();
  std::vector<std::pair<int, double> > temp_post(num_gauss);
  for (int g = 0; g < num_gauss; g++)
    temp_post[g] = std::pair<int, double>(g, log_likes_normalized(g));
  CompareReverseSecond compare;
  // Sort in decreasing order on posterior.  For efficiency we
  // first do nth_element and then sort, as we only need the part we're
  // going to output, to be sorted.
  std::nth_element(temp_post.begin(),
                   temp_post.begin() + num_gselect, temp_post.end(),
                   compare);
  std::sort(temp_post.begin(), temp_post.begin() + num_gselect,
            compare);

  post_entry.clear();
  post_entry.insert(post_entry.end(),
                    temp_post.begin(), temp_post.begin() + num_gselect);
  while (post_entry.size() > 1 && post_entry.back().second < min_post)
    post_entry.pop_back();
  // Now renormalize to sum to one after pruning.
  double tot = 0.0;
  size_t size = post_entry.size();
  for (size_t i = 0; i < size; i++) {
    tot += post_entry[i].second;
  }

  double inv_tot = 1.0 / tot;
  for (size_t i = 0; i < size; i++) {
    post_entry[i].second *= inv_tot;
  }
  return ans;
}
