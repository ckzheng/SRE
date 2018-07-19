#ifndef SIMPLE_WORD_SEG_H_
#define SIMPLE_WORD_SEG_H_

#include "base/idec_common.h"
#include "base/idec_types.h"
#include <vector>
#include <string>


namespace idec {
    class PronunciationLexicon;
    class SimpleWordSeg {
    public:
        // find all the text segmentation  according to the lex, 
        // oov are all removed
        static bool FindAllWordSeg(const std::string    &input_text,
                                   const PronunciationLexicon  &lex,
                                   std::vector<std::vector<std::string> >  *text_splits,
                                   int32 max_seg_allow = 100);
        typedef std::vector < std::string > WordSeg;
        struct BwdHyp {
            BwdHyp(size_t p) :start_pos(p) {}
            size_t      start_pos;
        };

        struct FwdHyp {
            FwdHyp(size_t p) :end_pos(p) {}
            size_t      end_pos;
        };

        // split the input_text into Chinese & English segments
        static bool SplitIntoChineseAndEnglish(const std::string &input_text,  std::vector<std::string>  *text_split);
        static bool IsEnglish(const std::string &input_text);
        static bool FindAllChineseWordSeg(const std::string &input_text, const PronunciationLexicon  &lex, std::vector<WordSeg> *segmentation);
        static bool FindAllEnglishWordSeg(const std::string &input_text, const PronunciationLexicon  &lex, std::vector<WordSeg> *segmentation);
        static void FwdHyp2Segmentations(const std::string &text, 
                                         std::vector<std::vector<FwdHyp> > &seg_hyps,
                                         size_t start_pos,
                                         std::vector<std::string> &partial_seg,
                                         std::vector<WordSeg> *segmentation);
    };
};

#endif
