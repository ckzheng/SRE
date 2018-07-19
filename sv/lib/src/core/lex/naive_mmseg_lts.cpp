#include "base/combinatorial.h"
#include "base/log_message.h"
#include "lex/naive_mmseg_lts.h"
#include "lex/pronunciation_lexicon.h"
#include "util/scoped_ptr.h"
#include "util/encode_converter.h"
#include "util/text-utils.h"

#include <algorithm>

namespace idec {



    // simple look up by lexicon look-up
    // first find the whole word
    // then try to do word segmentation and lookup
    // else return failed
    LtsRetCode NaiveMMSegLts::TextToPhone(const std::string &input_text, std::vector<PhoneId> *out_phones) {
        IDEC_ASSERT(base_lex_ != NULL);
        IDEC_ASSERT(0); // unimplemented
        return kLtsSuccess;
    }


    LtsRetCode NaiveMMSegLts::TextToPhone(const std::string & input_text, std::vector<std::vector<PhoneId> > *out_phones, const LtsOption &opt) {
        IDEC_ASSERT(base_lex_ != NULL);
        if (out_phones == NULL) {
            return kLtsNoneOutputBuffer;
        }
        out_phones->resize(0);

        // whole word match
        LexUnitXId lux_id = base_lex_->Str2LexUnitXId(input_text.c_str());
        if (lux_id != kInvalidLexUnitId) {
            for (Scoped_Ptr<LexUnitIterator> it(base_lex_->LexUnitXId2LexUnits(lux_id)); it->HasNext(); it->Next()) {
                LexUnit * lex_unit = it->CurrentItem();
                if (!base_lex_->IsSpecialSymbol(lex_unit)) {
                    std::vector<PhoneId> phones(lex_unit->phones, lex_unit->phones + lex_unit->num_phones);
                    out_phones->push_back(phones);

                    // add silence phones variant
                    // note there is chance to be duplicate with other prouns
                    if (opt.add_sil_each_word || opt.add_sil_whole_word) {
                        if (lex_unit->phones[lex_unit->num_phones - 1] != phone_set_->PhoneIdOfSilence()) {
                            phones.push_back(phone_set_->PhoneIdOfSilence());
                            out_phones->push_back(phones);
                        }
                    }
                }
             }
        }
        else {
            // do simple forward-mm-seg, remove the space
            std::vector<std::string >text_split;
            SimpleTNAndSplit(input_text, &text_split);
            if (text_split.size() == 0) {
                return kLtsCanNotHandleWord;
            }
            std::vector<Segment> segmentation;
            segmentation.reserve(32);
            LtsRetCode ret = kLtsSuccess;

            for (size_t i = 0; i < text_split.size(); i++) {
                if (IsEnglish(text_split[i])) {
                    ret = DoEnglishWordSeg(text_split[i], &segmentation);
                }
                else {
                    ret = DoChineseWordSeg(text_split[i], &segmentation);
                }
                if (ret != kLtsSuccess) {
                    return ret;
                }
            }

            // return all the combination of the word prouns based on the segmenting result
            std::vector<std::vector<LexUnit*> > all_prouns(segmentation.size());
            GeneratePronForEachWord(segmentation, all_prouns, opt);


            idec::CartesianProduct<LexUnit*> p(all_prouns);
            size_t total_prons = 0;
            std::vector<PhoneId> one_pron;
            one_pron.reserve(100);
            std::vector<PhoneId> one_pron_with_sil;
            one_pron_with_sil.reserve(100);

            while (!p.Done()) {

                if (opt.max_pron_per_word < 0 || out_phones->size() < (size_t)opt.max_pron_per_word) {
                    std::vector<LexUnit*> & one_pron_lu = p.Value();
                    IDEC_ASSERT(one_pron_lu.size() == segmentation.size());
                    one_pron.resize(0);
                    one_pron_with_sil.resize(0);
                    // convert the pronunciations
                    size_t seg_w = 0;
                    for (std::vector<LexUnit*>::iterator word_prons = one_pron_lu.begin();
                         word_prons != one_pron_lu.end();
                         ++word_prons, ++seg_w) {
                        LexUnit *pron_of_word((*word_prons));
                        for (size_t p = 0; p < pron_of_word->num_phones; p++) {
                            one_pron.push_back(pron_of_word->phones[p]);
                            if (opt.add_sil_each_word) {
                                one_pron_with_sil.push_back(pron_of_word->phones[p]);
                            }
                        }
                        // add silence after each word
                        if (opt.add_sil_each_word && one_pron_with_sil.back() != phone_set_->PhoneIdOfSilence()) {
                            one_pron_with_sil.push_back(phone_set_->PhoneIdOfSilence());
                        }

                        // only silence after last word
                        if (!opt.add_sil_each_word
                            && opt.add_sil_whole_word
                            && seg_w == one_pron_lu.size() - 1
                            &&  one_pron_with_sil.back() != phone_set_->PhoneIdOfSilence()) {
                            one_pron_with_sil.push_back(phone_set_->PhoneIdOfSilence());
                        }
                    }

                    // limit the max number of pronunciations generated for one word
                    out_phones->push_back(one_pron);
                    total_prons++;


                    if (one_pron_with_sil.size()>0 && one_pron.size() != one_pron_with_sil.size()) {
                        out_phones->push_back(one_pron_with_sil);
                        total_prons++;
                    }
                }
                else {
                    total_prons++;
                }

                p.Next();
            }

            if (total_prons > out_phones->size()) {
#ifdef _MSC_VER
                IDEC_VERB << EncodeConverter::UTF8ToLocaleAnsi(input_text) << "has total " << total_prons <<
                    " pronunciations truncated to " << opt.max_pron_per_word;
#else
                IDEC_VERB <<input_text << "has total" << total_prons <<
                    " pronunciations truncated to " << opt.max_pron_per_word;
#endif
            }
        }

        std::sort(out_phones->begin(), out_phones->end());
        std::vector<std::vector<PhoneId> >::iterator end = std::unique(out_phones->begin(), out_phones->end());
        out_phones->resize(end - out_phones->begin());
        return kLtsSuccess;
    }


    LtsRetCode NaiveMMSegLts::SimpleTNAndSplit(const std::string &input_text, std::vector<std::string> *text_split) {
        std::string in_text = "";

        // convert into upper case word and split English with Chinese
        bool last_char_is_alpha = false;
        for (size_t i = 0; i < input_text.size(); i++) {
            if (isascii(input_text[i]) && isalpha(input_text[i])) {
                if (!last_char_is_alpha && i != 0) {
                    in_text.push_back(' ');
                }
#ifdef CONVERT_ENGLISH_TO_LOWER_LEXICON
                if (isupper(input_text[i])) {
                    in_text.push_back(tolower(input_text[i]));
                }
                else
#else
                {
                    in_text.push_back(input_text[i]);
                }
#endif
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
            return kLtsCanNotHandleWord;
        }
        return kLtsSuccess;
    }

    bool NaiveMMSegLts::IsEnglish(const std::string &input_text) {
        for (size_t i = 0; i < input_text.size(); i++) {
#ifdef CONVERT_ENGLISH_TO_LOWER_LEXICON
            //if (isascii(input_text[i]) && islower(input_text[i])) {
#else
            if (isascii(input_text[i])) {
#endif
            }
            else {
                return false;
            }
        }
        return true;
    }

    LtsRetCode NaiveMMSegLts::DoChineseWordSeg(const std::string &input_text, std::vector<Segment> *segmentation) {
        size_t remain_text_start = 0;
        size_t remain_text_length = input_text.size();
        size_t max_word_len = base_lex_->GetMaxLexicalUnitStrLen();

        // deal with encoding problem?
        while (remain_text_start < input_text.size()) {
            size_t word_len_try = 1;
            size_t word_len_max_match = 1;
            bool found = false;
            while (word_len_try <= remain_text_length &&  word_len_try <= max_word_len) {
                std::string word_try = input_text.substr(remain_text_start, word_len_try);
                LexUnitXId lux_id = base_lex_->Str2LexUnitXId(word_try.c_str());
                if (kInvalidLexUnitId != lux_id) {
                    word_len_max_match = word_len_try;
                    found = true;
                }
                word_len_try++;
            }
            // real OOVs even after
            if (!found) {
                break;
            }
            else {
                std::string word_match = input_text.substr(remain_text_start, word_len_max_match);
                LexUnitXId lux_id = base_lex_->Str2LexUnitXId(word_match.c_str());
                Segment seg;
                seg.lux_id = lux_id;
                seg.is_spelling = false;
                seg.word_str_len = word_match.size();
                segmentation->push_back(seg);
                remain_text_start += word_len_max_match;
                remain_text_length -= word_len_max_match;
            }
        }

        // fail to find all sub-word
        if (remain_text_start < input_text.size()) {
            return kLtsCanNotHandleWord;
        }

        return kLtsSuccess;
    }

    LtsRetCode NaiveMMSegLts::DoEnglishWordSeg(const std::string &input_text, std::vector<Segment> *segmentation) {
        // try it as whole word
        LexUnitXId lux_id = base_lex_->Str2LexUnitXId(input_text.c_str());
        if (lux_id != kInvalidLexUnitId) {
            Segment seg;
            seg.lux_id = lux_id;
            seg.is_spelling = false;
            seg.word_str_len = input_text.size();
            segmentation->push_back(seg);
            return kLtsSuccess;
        }
        else {
            // split it as spelling: letter by letter
            for (size_t i = 0; i < input_text.size();i++) {
                char letter[2] = "";
                letter[0] = input_text[i];
                LexUnitXId lux_id = base_lex_->Str2LexUnitXId(letter);
                if (lux_id != kInvalidLexUnitId) {
                    Segment seg;
                    seg.lux_id = lux_id;
                    seg.is_spelling = true;
                    seg.word_str_len = input_text.size();
                    segmentation->push_back(seg);
                }
                else {
                    return kLtsCanNotHandleWord;
                }
            }
            return kLtsSuccess;
        }
    }

    void NaiveMMSegLts::GeneratePronForEachWord(std::vector<Segment> &segmentation, std::vector<std::vector<LexUnit*> > &all_prouns, const LtsOption &opt) {
        for (size_t w = 0; w < segmentation.size(); w++) {
            LexUnitXId &lux_id = segmentation[w].lux_id;

            for (Scoped_Ptr<LexUnitIterator> it(base_lex_->LexUnitXId2LexUnits(lux_id)); it->HasNext(); it->Next()) {
                LexUnit * lex_unit = it->CurrentItem();
                all_prouns[w].push_back(lex_unit);
                // adhoc rule, for long spelling, only keep 1 pron
                if (segmentation[w].is_spelling && segmentation[w].word_str_len > 2) {
                    break;
                }
            }
        }
    }

};
