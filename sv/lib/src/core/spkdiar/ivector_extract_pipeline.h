#ifndef _ACC_STATES_H_
#define _ACC_STATES_H_

#include "base/log_message.h"
#include "diag_gmm.h"
#include "posterior.h"
#include "ivector_estimator.h"

class IvectorExtractPipeline {
 public:
  IvectorExtractPipeline() {}

  ~IvectorExtractPipeline() {}

  int DiagGmmgSelect(DoubleMatrix &feats, DiagGmm &gmm, int num_gselect,
                     vector<vector<int>> &gselect) {
    idec::IDEC_ASSERT(num_gselect > 0);
    int num_gauss = gmm.NumGauss();
    if (num_gselect > num_gauss) {
      idec::IDEC_WARN << "You asked for " << num_gselect << " Gaussians but GMM "
                      << "only has " << num_gauss << ", returning this many. "
                      << "Note: this means the Gaussian selection is pointless.";
      num_gselect = num_gauss;
    }
	if (feats.Rows() != gselect.size()) {
    gselect.resize(feats.Rows());
	}
    int tot_t_this_file = feats.Rows();
    double tot_like_this_file = gmm.GaussianSelection(feats, num_gselect, gselect);
    idec::IDEC_INFO << tot_t_this_file << " frames is " << (tot_like_this_file /
                    tot_t_this_file);
	return 0;

  }

  int FullGmmgSelectToPost(DoubleMatrix &feats, FullGmm &fgmm, Posterior&post,
                           vector<vector<int>> &gselect, double min_post) {
    int tot_posts = 0;
    int num_frames = feats.Rows();
    int dims = feats.Cols();
	if (post.Size() != num_frames) {
		post.Resize(num_frames);
	}
    //Posterior post(num_frames);	
    double this_tot_loglike = 0;
    bool utt_ok = true;
	DoubleVector frame;
	DoubleVector loglikes;
    for (int t = 0; t < num_frames; t++) {
      //DoubleVector frame(mat, t);  
	  /*
      for(int j = 0; j < dims; ++j) {
        frame(j) = feats(t, j);
      }
	  */
	  frame = feats.Rowv(t);
      const std::vector<int> &this_gselect = gselect[t];
      idec::IDEC_ASSERT(!gselect[t].empty());     
      fgmm.LogLikelihoodsPreselect(frame, this_gselect, loglikes);
      this_tot_loglike += loglikes.ApplySoftMax();
      // now "loglikes" contains posteriors.
      if (fabs(loglikes.Sum() - 1.0) > 0.01) {
        utt_ok = false;
      } else {
        if (min_post != 0.0) {
          int max_index = 0; // in case all pruned away...
          loglikes.Max(&max_index);
		  for (int i = 0; i < loglikes.Size(); i++) {
			  //if (loglikes(i) < min_post)
			  //  loglikes(i) = 0.0;
			  loglikes(i) = (loglikes(i) < min_post) ? 0.0 : loglikes(i);
		  }
		  double sum = loglikes.Sum();
          if (sum == 0.0) {
            loglikes(max_index) = 1.0;
          } else {
            loglikes.Scale(1.0 / sum);
          }
        }

        for (int i = 0; i < loglikes.Size(); i++) {
          if (loglikes(i) != 0.0) {
            post[t].Add(std::make_pair(this_gselect[i], loglikes(i)));
            tot_posts++;
          }
        }
        idec::IDEC_ASSERT(!post[t].Empty());
      }
    }

    if (!utt_ok) {
      idec::IDEC_ERROR <<
                      "Skipping utterance because bad posterior-sum encountered (NaN?)";
      return -1;
    }
    return 0;
  }
  
  int DoIvectorExtract(DoubleMatrix &feats, Posterior &post, bool compute_objf_change, double max_count, double acoustic_weight, ResourceLoader& res_loader) {
    double tot_auxf_change = 0.0, tot_t = 0.0;    	
	//IvectorEstimator(const IvectorExtractOptions &opts, const FullGmm &fgmm) {
	const IvectorExtractOptions opts;
	IvectorEstimator ivector_estimator = IvectorEstimator(opts, res_loader.GetIvectorResource());
    if (static_cast<int>(post.Size()) != feats.Rows()) {
      idec::IDEC_WARN << "Size mismatch between posterior " << post.Size()
                      << " and features " << feats.Rows() << " for utterance ";
    }

    double *auxf_ptr = (compute_objf_change ? &tot_auxf_change : NULL);
    double this_t = acoustic_weight * post.Total(), max_count_scale = 1.0;
    if (max_count > 0 && this_t > max_count) {
      max_count_scale = max_count / this_t;
      idec::IDEC_INFO << "Scaling stats for utterance by scale "
                      << max_count_scale << " due to --max-count="<< max_count;
      this_t = max_count;
    }

    post.Scale(acoustic_weight * max_count_scale);
    //ScalePosterior(acoustic_weight * max_count_scale, post);
    // note: now, this_t == sum of posteriors.
    //sequencer.Run(new IvectorExtractTask(extractor, utt, mat, posterior, &ivector_writer, auxf_ptr));  
	DoubleVector ivector;
	ivector_estimator.Run(feats, post, ivector);
	if (compute_objf_change) {
		idec::IDEC_INFO << "Overall average objective-function change from estimating "
			<< "ivector was " << (tot_auxf_change / tot_t) << " per frame "
			<< " over " << tot_t << " (weighted) frames.";
	}
	return 0;
  }
 
};


#endif // !_ACC_STATES_H_
