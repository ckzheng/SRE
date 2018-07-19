#include "lex/simple_word_seg.h"
#include "lex/pronunciation_lexicon.h"
#include "util/text-utils.h"
#include "base/combinatorial.h"
#include <vector>

namespace idec {
    bool SimpleWordSeg::FindAllWordSeg(const std::string                         &input_text,
                                       const PronunciationLexicon                &lex,
                                       std::vector<std::vector<std::string> >   *all_segmentations,
                                       int32                                     max_seg_allow) {

         all_segmentations->clear();
        // 1) do Chinese-English segmentation
         std::vector<std::string> text_split;
         SplitIntoChineseAndEnglish(input_text, &text_split);


        // 2) do word segmentation on each sub field
         std::vector< std::vector<WordSeg > >per_field_segmentations(text_split.size());
         bool is_ok = true;
         for (size_t i = 0; i < text_split.size(); i++) {
             if (IsEnglish(text_split[i])) {
                 is_ok =  FindAllEnglishWordSeg(text_split[i], lex,&per_field_segmentations[i]);
             }
             else {
                 is_ok = FindAllChineseWordSeg(text_split[i], lex, &per_field_segmentations[i]);
             }
             if (!is_ok) {
                 return false;
             }
         }
         // check split result
         for (size_t i = 0; i < per_field_segmentations.size(); ++i) {
           if (per_field_segmentations[i].size() == 0) {
             return false;
           }
         }

         // 3) find all the cross-product of each seg
         CartesianProduct<WordSeg> p(per_field_segmentations);
         while (!p.Done()){
             std::vector<WordSeg> &one_combination = p.Value();
             if (all_segmentations->size() >= max_seg_allow) {
                 break;
             }
             else {
                 // re-arrange them into a string array
                 std::vector<std::string> one_segmentation;
                 for (size_t i = 0; i < one_combination.size(); i++) {
                     for (size_t j = 0; j < one_combination[i].size(); j++) {
                         one_segmentation.push_back(one_combination[i][j]);
                     }
                 }
                 all_segmentations->push_back(one_segmentation);
             }
             p.Next();
         }

         return true;
    }


    bool SimpleWordSeg::SplitIntoChineseAndEnglish(const std::string &input_text, std::vector<std::string> *text_split) {
        std::string in_text = "";

        bool last_char_is_alpha = false;
        for (size_t i = 0; i < input_text.size(); i++) {
            if (isascii(input_text[i]) && isalpha(input_text[i])) {
                if (!last_char_is_alpha && i != 0) {
                    in_text.push_back(' ');
                }
                in_text.push_back(input_text[i]);
                last_char_is_alpha = true;
            }
            else {
                if (last_char_is_alpha && i != 0) {
                    in_text.push_back(' ');
                }
                in_text.push_back(input_text[i]);
                last_char_is_alpha = false;
            }
        }
        text_split->resize(0);
        SplitStringToVector(in_text, "\n\t ", true, text_split);
        if (text_split->size() == 0) {
            return false;
        }
        return true;
    }

    bool SimpleWordSeg::IsEnglish(const std::string &input_text) {
        for (size_t i = 0; i < input_text.size(); i++) {
            if (!isascii(input_text[i])) {
                return false;
            }
        }
        return true;
    }

    // do all the Chinese segmentation
    bool SimpleWordSeg::FindAllChineseWordSeg(const std::string             &input_text, 
                                              const PronunciationLexicon    &lex,
                                              std::vector<WordSeg>         *segmentation) {

        size_t max_word_len = const_cast<PronunciationLexicon*>(&lex)->GetMaxLexicalUnitStrLen();
        size_t min_word_len = 1;
        // segmentation hyps
        std::vector<std::vector<BwdHyp> > seg_bwd_hyps;
        seg_bwd_hyps.resize(input_text.size());

        // 1) do all the segmentations, store backward hyps
        // try to matching [start_pos end_pos]
        for (size_t start_pos = 0; start_pos < input_text.size(); start_pos++) {
            if (start_pos == 0 || seg_bwd_hyps[start_pos - 1].size() != 0) {
                for (size_t len = min_word_len; len <= max_word_len && start_pos + len <= (size_t)input_text.size(); len++) {
                    size_t end_pos = start_pos + len - 1;

                    std::string sub_str = input_text.substr(start_pos, len);

                    LexUnitXId lux_id = lex.Str2LexUnitXId(sub_str.c_str());
                    if (lux_id != kInvalidLexUnitId) {
                        BwdHyp hyp(start_pos);
                        seg_bwd_hyps[end_pos].push_back(hyp);
                    }
                }
            }
        }

        // 2) convert into forward hyps
        std::vector<std::vector<FwdHyp> > seg_fwd_hyps;
        seg_fwd_hyps.resize(input_text.size());
        for (size_t i = 0; i < seg_bwd_hyps.size(); i++) {
            for (size_t j = 0; j < seg_bwd_hyps[i].size(); j++) {
                size_t start_pos = seg_bwd_hyps[i][j].start_pos;
                size_t end_pos = i;
                FwdHyp hyp(end_pos);
                seg_fwd_hyps[start_pos].push_back(hyp);
            }
        }
        // 3) all the segmentations
        std::vector<std::string> partial_seg;
        FwdHyp2Segmentations(input_text, seg_fwd_hyps, 0, partial_seg, segmentation);
        return true;
    }


    // just split by spaces
    bool SimpleWordSeg::FindAllEnglishWordSeg(const std::string &input_text,
                                              const PronunciationLexicon  &lex, 
                                              std::vector<WordSeg>*segmentation) {
        segmentation->resize(1);
        (*segmentation)[0].clear();


        std::vector<std::string> &text_split((*segmentation)[0]);
        SplitStringToVector(input_text, "\n\t ", true, &text_split);
        if (text_split.size() == 0) {
            segmentation->resize(0);
            return false;
        }

        for (size_t i = 0; i < text_split.size();i++){
            LexUnitXId lux_id = lex.Str2LexUnitXId(text_split[i].c_str());
            if (lux_id == kInvalidLexUnitId) {
                segmentation->resize(0);
                return false;
            }
        }

        return true;
    }


    void SimpleWordSeg::FwdHyp2Segmentations(const std::string &text,
        std::vector<std::vector<FwdHyp> > &seg_hyps,
        size_t start_pos,
        std::vector<std::string> &partial_seg,
        std::vector<WordSeg> *segmentation) {
        if (start_pos >= seg_hyps.size()) {
            segmentation->push_back(partial_seg);
            return;
        }

        std::vector<FwdHyp> &cur_fwd_hyp(seg_hyps[start_pos]);
        for (size_t i = 0; i < cur_fwd_hyp.size(); i++) {
            partial_seg.push_back(text.substr(start_pos, cur_fwd_hyp[i].end_pos - start_pos + 1));
            FwdHyp2Segmentations(text, seg_hyps, cur_fwd_hyp[i].end_pos + 1, partial_seg, segmentation);
            partial_seg.pop_back();
        }
        return;
    }

};
