#ifndef NAIVE_MMSEG_LTS_H__
#define NAIVE_MMSEG_LTS_H__
#include <vector>
#include <string>
#include "lts.h"
#include "phone_set.h"

namespace idec {

    class PronunciationLexicon;
    class PhoneSet;
    struct LexUnit;

    class NaiveMMSegLts:public Lts {
    public:
        NaiveMMSegLts(PronunciationLexicon  *base_lex, PhoneSet *phone_set) :base_lex_(base_lex), phone_set_(phone_set) {}
        LtsRetCode virtual Init() { return kLtsSuccess; };
        LtsRetCode virtual Unit() { return kLtsSuccess; };
        LtsRetCode virtual TextToPhone(const std::string &input_text, std::vector<std::string> *out_phones) { return kLtsSuccess; };
        LtsRetCode virtual TextToPhone(const std::string &input_text, std::vector<PhoneId> *out_phones);
        LtsRetCode virtual TextToPhone(const std::string &input_text, std::vector<std::vector<PhoneId> > *out_phones, const LtsOption &opt);

   

        virtual ~NaiveMMSegLts() {};

    private:
        struct Segment{
            int32 lux_id;
            bool is_spelling;
            size_t word_str_len;
        };
        LtsRetCode SimpleTNAndSplit(const std::string &input_text, std::vector<std::string>  *text_split);
        bool IsEnglish(const std::string& input_text);

        LtsRetCode DoChineseWordSeg(const std::string &input_text, std::vector<Segment> *segmentation);
        LtsRetCode DoEnglishWordSeg(const std::string &input_text, std::vector<Segment> *segmentation);
        void GeneratePronForEachWord(std::vector<Segment> &segmentation, std::vector<std::vector<LexUnit*> > &all_prouns, const LtsOption &opt);
    private:
        PronunciationLexicon *base_lex_;
        PhoneSet*phone_set_;
    };
};

#endif
