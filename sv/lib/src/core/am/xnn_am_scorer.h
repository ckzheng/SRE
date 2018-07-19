#ifndef XNN_AM_SCORER_H
#define XNN_AM_SCORER_H

#include <string>
#include "am/am_scorer.h"
#include "am/xnn_net.h"

namespace idec {

    struct XNNAcousticModelScorerOpt {
        int   input_block_size;      // the input feature block size for evaluate the likelihood  
        int   output_block_size;     // the output likelihood block size for decoding
        float ac_scale;              // acoustic model score factor
        bool  lazy_evaluation;       // lazy evaluation
    };

    // the neural network observation model 
    class XNNAcousticModelScorer : public AcousticModelScorer {

    private:
        // the dnn network and evaluator
        xnnNet                        *net_;
        xnnAmEvaluator                *evaluator_;
        // option to set 
        XNNAcousticModelScorerOpt     opts_;

    public:
        XNNAcousticModelScorer(const XNNAcousticModelScorerOpt &opt, xnnNet *net_);
        virtual ~XNNAcousticModelScorer();

        virtual size_t NumPdfs() const { return net_->uDim(); };
        virtual void   PushFeatures(int start_frame, const xnnFloatRuntimeMatrix &feat);
        virtual float  GetFrameScore(int frame, int pdf_id);
        virtual float* GetFrameScores(int frame);
        virtual int    BeginUtterance() {
            evaluator_->reset();
            return(0);
        }
        virtual int    EndUtterance() {
            return(0);
        };
    };

};

#endif

