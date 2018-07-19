#ifndef SORTED_LEX_PROUNICATION_LEXICON_H
#define SORTED_LEX_PROUNICATION_LEXICON_H 1

#include <list>
#include <string>
#include <vector>
#include <list>
#include "lex/pronunciation_lexicon.h"
#include "base/idec_types.h"
namespace idec{

    // lexical unitx = lexical unit with the same string 
    // assume a lex_unitx mapping to a continuous array of lex_unit
    // 8 + 4 + 1 = 13
#pragma pack(push,1)
    typedef struct {
        LexUnitId     lex_unit_id_start;
        uint8_t       num_pron_vairant;
        char         *word_str;                 // word string
    }LexUnitX;
#pragma pack(pop)

    // for text loading
    struct LexUnitTxtLoading: public LexUnit {
        char         *word_str;
        LexUnitTxtLoading() {
            word_str = NULL;
        }
    };

    // fix cost=30 per word
    // + string + phone
    // 80k voc = 2.4M + string + phone = 4M?
    class SortedLexUnitIterator: public LexUnitIterator {
    public:
        virtual bool HasNext() const { return cur_ != end_; }
        virtual void Next() { cur_++; };
        virtual LexUnit* CurrentItem() { return cur_;}
        SortedLexUnitIterator(LexUnit *begin, LexUnit *end) {
            cur_ = begin;
            end_ = end;
        }
        private:
        LexUnit *end_;
        LexUnit *cur_;
    };

  
    // real sorted array by the string comparison (except the special symbols )
    // lexicon unit id is the index in the array, so it is sorted as well
    // all the lexicon unit of the same lexiconUnitX are grouped together
    // the search are binary search now, not using hash functions
    class SortedPronunciationLexicon :public PronunciationLexicon{
    public:
        SortedPronunciationLexicon(PhoneSet *phone_set);
        virtual ~SortedPronunciationLexicon();

        virtual LexUnitId NumOfLexUnit() const { return num_lex_unit_; }
        virtual LexUnitXId NumOfLexUnitX() const { return num_lex_unitx_; }
 
        // mapping string to lexicon unit-x id and back
        virtual LexUnitXId  Str2LexUnitXId(const char *str) const;                  // bsearch on lexUnitX 
        virtual const char* LexUnitXId2Str(const LexUnitXId &lex_unitx_id);    // array indexing;
                            

        // mapping id to it is real lexicon unit
        virtual const LexUnit*  LexUnitId2LexUnit(const LexUnitId &lex_unit_id);// array indexing


        // mapping string to all it pronunciations variations
        virtual LexUnitIterator*  Str2LexUnits(const char *str); //str=>lexUnitX=>LexUnit
        virtual LexUnitIterator*  LexUnitXId2LexUnits(const LexUnitXId &lex_unitx_id); //lexUnitX=>LexUnit
        virtual const char* LexUnitId2Str(const LexUnitId &lex_unit_id);
        virtual const LexUnitXId LexUnitIdToLexUnitXId(const LexUnitId &lex_unit_id);  // bsearch in lex_unitx_, key is lex_unit_id

        // special symbols dealing
        LexUnit *LexUnitSilence();
        LexUnit *LexUnitSentenceBegin();
        LexUnit *LexUnitSentenceEnd();

 
        // traversing all lex units in the lexicon
        virtual LexUnitIterator*  GetLexUnitIterator(); 

        // loading and saving
        virtual IDEC_RETCODE LoadTxtFile(const char *file_name);
        virtual IDEC_RETCODE Load(const char *file_name);
        virtual IDEC_RETCODE SaveTxtFile(const char *file_name);
        virtual IDEC_RETCODE Save(const char *file_name);
   


        // build from a word list
        void BuildFromWordList(const std::vector<std::string> &word_list,
                           PronunciationLexicon *base_lexicon, Lts *lts,
                           std::vector<std::string> &oov_list, const LtsOption &lts_opt);


    struct BinHeader;
    private:
        bool IsValidBinaryFile(const char* file_name);
        void Save(std::ostream &oss);
        void Load(std::istream &iss);
        void ReadLexUnit(std::istream &iss,  LexUnit *lex_unit);
        void WriteLexUnit(std::ostream &oss, const LexUnit &lex_unit);
        void ReadPhoneTable();
        void WritePhoneTable();

        void ReadLexUnitX(std::istream &iss, LexUnitX *lex_unit);
        void WriteLexUnitX(std::ostream &oss, const LexUnitX &lex_unit);
        void ReadLexUnit(const char *line, LexUnitTxtLoading **lex_unit_loading);
        void Clear();
        void InsertSpecials(std::list<LexUnitTxtLoading*> &all_lex_units);//i/o
        size_t NumOfSpecials() { return 4; }
        void BuildFromTxtLoadUnit(std::list<LexUnitTxtLoading*> &all_lex_units);
        void BuildCompact(std::list<LexUnitTxtLoading*> &all_lex_units);
        LexUnit* SearchSpecial(const char *str);

   

    private:
        // < or >?
        static bool CmpByLexStr(const LexUnitX &a, const LexUnitX &b) {
            return (strcmp(a.word_str, b.word_str) < 0);
        }

        static bool CmpByLexUnitId(const LexUnitX &a, const LexUnitX &b) {
            return (a.lex_unit_id_start < b.lex_unit_id_start);
        }

        static bool CmpByStringAndPhone(const LexUnitTxtLoading *a, const LexUnitTxtLoading *b) {
            int str_res = strcmp(a->word_str, b->word_str);
            if (str_res < 0) {
                return true;
            }
            else if (str_res>0) {
                return false;
            }
            else {
                return CmpByPhone(a, b);
            }
        }

        bool static CmpByPhone(const LexUnitTxtLoading *a, const LexUnitTxtLoading *b) {
            if (a->num_phones == b->num_phones) {
                if (a->num_phones == 0) {
                    return false;
                }
                return (memcmp(a->phones, b->phones, (size_t)(a->num_phones))<0);
            }
            else {
                return a->num_phones < b->num_phones;
            }
        }


        bool static EqualStringAndPhone(const LexUnitTxtLoading *a, const LexUnitTxtLoading *b) {
            return (a->num_phones == b->num_phones
                    && strcmp(a->word_str, b->word_str) == 0
                    && memcmp(a->phones, b->phones, (size_t)(a->num_phones)) == 0
                    );
        }
        bool static EqualString(const LexUnitTxtLoading *a, const LexUnitTxtLoading *b) {
            return (strcmp(a->word_str, b->word_str) == 0);
        }

        const LexUnitX * Str2LexUnitX(const char *str) const;

        private:
        // real things
        LexUnitX     *lex_unitx_;
        LexUnit      *lex_unit_;
        LexUnitId     num_lex_unit_;
        LexUnitXId    num_lex_unitx_;


        // NOTE: this a am important optimization using memory pools
        // if the phones and word string are allocated on a per-string basis
        // the memory will be blow up 6x times or more.
        PhoneId   *phone_buf_;    // all the phone string are all allocated here
        char      *word_str_buf_; // all the word string are all allocated here
        uint32    word_str_buf_size_;
        uint32    phone_buf_size_;



        // the special lex unit are placed in [0..num_special_lex_unit_-1]
        LexUnitId     num_special_lex_unit_;
        LexUnitXId    num_special_lex_unitx_;
    

        // other statistic
        // not owned by itself
        PhoneSet  *phone_set_;
    };
};

#endif
