#ifndef FRM_SKIP_AM_SCORER_H
#define FRM_SKIP_AM_SCORER_H
#include "am/am_scorer.h"

namespace idec{
    class FrmSkipAcousticModelScorer : public AcousticModelScorer {
    public:
        FrmSkipAcousticModelScorer(AcousticModelScorer*base_scorer, int frm_skip_num) {
            base_scorer_ = base_scorer;
            frm_skip_num_ = frm_skip_num;

            if (frm_skip_num <= 0) {
                IDEC_ERROR << "frm_skip_num should be bigger than 0";
            }
        }

        virtual ~FrmSkipAcousticModelScorer() {
            IDEC_DELETE(base_scorer_);
        }

        virtual size_t NumPdfs() const { return base_scorer_->NumPdfs(); };

        // down sample the features and push in
        virtual void  PushFeatures(int start_frame, const xnnFloatRuntimeMatrix &feat) {
            if (frm_skip_num_ == 1) {
                base_scorer_->PushFeatures(start_frame, feat);
                return;
            }

            // make sure every push is multiple except the last push
            if ((start_frame % frm_skip_num_) != 0) {
                IDEC_ERROR << "you must push the frame as multiples of the frame skips frame start idx is" << start_frame;
            }
            
            // for the last push, the number of push maybe not a times of a frm_skip_num
            // e.g. block_size=8, frm_skip_num_ =2, we push 7 frames, then we still need 4 frame for evaluation
            size_t num_frm_after_downsample = ((feat.NumCols() % frm_skip_num_) == 0) ? (feat.NumCols() / frm_skip_num_) : (feat.NumCols() / frm_skip_num_ +1);

            feat_downsample_.Resize(feat.NumRows(), num_frm_after_downsample);

            for (size_t i = 0; i < feat_downsample_.NumCols(); ++i) {
                memcpy(feat_downsample_.Col(i), (void*)(feat.Col(i*frm_skip_num_)), feat.NumRows()*sizeof(float));
            }

            base_scorer_->PushFeatures(start_frame / frm_skip_num_, feat_downsample_);
        }

        virtual float  GetFrameScore(int frame, int pdf_id) {
            return base_scorer_->GetFrameScore(frame / frm_skip_num_, pdf_id);
        }

        virtual float* GetFrameScores(int frame) {
            return base_scorer_->GetFrameScores(frame / frm_skip_num_);
        }

        virtual int    BeginUtterance() {
            return base_scorer_->BeginUtterance();
        }

        virtual int    EndUtterance() {
            return base_scorer_->EndUtterance();
        };

        // implementations
        AcousticModelScorer *base_scorer_;
        xnnFloatRuntimeMatrix feat_downsample_;

        int frm_skip_num_;

    };
}

#endif


