#ifndef LEX_PROUNICATION_LEXICON_H
#define LEX_PROUNICATION_LEXICON_H 1
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>

#include "base/idec_types.h"
#include "base/idec_common.h"
#include "base/idec_return_code.h"


namespace idec{

    struct LtsOption;
#define NON_LEXUNIT_ID      -100

    // maximum length of a lexical unit including the end-of-string character 
#define MAX_LEXUNIT_LENGTH  1024 

    // maximum number of alternative pronunciations a lexical unit can have
#define MAX_LEXUNIT_PRONUNCIATIONS    1024

    // lexical unit types
#define LEX_UNIT_TYPE_STANDARD                    0
#define LEX_UNIT_TYPE_FILLER                      1
#define LEX_UNIT_TYPE_SENTENCE_DELIMITER          2
#define LEX_UNIT_TYPE_UNKNOWN                     3
#define LEX_UNIT_TYPE_SIL                         4

#define LEX_UNIT_SILENCE_SYMBOL                    "!@!SIL!@!"//"<SIL>"
#define LEX_UNIT_BEG_SENTENCE                      "<s>"
#define LEX_UNIT_END_SENTENCE                      "</s>"
#define LEX_UNIT_UNKNOWN                           "<unk>"

    // value for uninitialized variables that keep lexical unit indices
#define UNINITIALIZED_LEX_UNIT   LONG_MIN

    class Lts;
    class PhoneSet;
    typedef int32 LexUnitId;                               //  lexical unit id (unique correspondence id  <> lexical unit + pronunciation)
    typedef int32 LexUnitXId;                              //  lexical unitX id (unique correspondence id <> lexical unit of str format)
    const  LexUnitId kInvalidLexUnitId = (LexUnitId)(-1);
    const  LexUnitId kOOVLexUnitId = kInvalidLexUnitId;

    // maybe renamed in the future
    // LexUnitId   ==> LexPronId
    // LexUnitXId  ==> LexWordId

    // lexical unit transcription
#pragma pack(push,1)
    //  8 + 2 + 4 + 1 + 1 = 16
    typedef struct LexUnit {
        uint8_t        lex_unit_type;            // standard / filler / sentence delimiter
        LexUnitId      lex_unit_id;              // the lex unit id of this lexicon unit
        uint8_t        pron_variant_id;          // pronunciation variants id
        uint16         num_phones;               // number of phones in phones field

        // this pointer must be placed at the end of the structure
        // the I/O code use this assumption
        PhoneId*       phones;                   // phonetic transcription 

        LexUnit() {
            phones = NULL;
            num_phones = 0;
            lex_unit_id = kInvalidLexUnitId;
        }
        void CopyFrom(const LexUnit & other) {
            if (&other == this)
                return;

            IDEC_DELETE_ARRAY(phones);
            *this = other;
            if (other.phones != NULL) {
                this->phones = new PhoneId[other.num_phones];
            }
            memcpy(this->phones, other.phones,other.num_phones);
        }
    } LexUnit;
#pragma pack(pop)

    class LexUnitIterator {
    public:
        virtual ~LexUnitIterator() {};
        virtual bool HasNext() const = 0;
        virtual void Next() = 0;
        virtual LexUnit* CurrentItem() = 0;
    };


    // virtual interface for the pronunciation lexicon 
    class PronunciationLexicon {
    public:
        PronunciationLexicon(PhoneSet * phone_set) { max_lex_unit_str_len_ = 0; };
        virtual ~PronunciationLexicon() {}
        virtual LexUnitId  NumOfLexUnit() const = 0;
        virtual LexUnitXId NumOfLexUnitX() const = 0;



        // mapping string to lexicon unit-x id and back
        virtual LexUnitXId    Str2LexUnitXId(const char* str) const = 0;
        virtual const char*   LexUnitXId2Str(const LexUnitXId &lex_unitx_id) = 0;

        // mapping id to it is real lexicon unit
        virtual const LexUnit*  LexUnitId2LexUnit(const LexUnitId &lex_unit_id) = 0;

        // mapping string to all it pronunciations variations
        // NOTE: user are responsible to call of the delete the LexUnitIterator
        // a suggest usage is:
        // Scope_Ptr<Str2LexUnits> iter(Str2LexUnits(str))
        // or any better design?
        virtual LexUnitIterator*  Str2LexUnits(const char *str) =0; 
        virtual LexUnitIterator*  LexUnitXId2LexUnits(const LexUnitXId &lex_unitx_id) =0;


        virtual const char* LexUnitId2Str(const LexUnitId &lex_unit_id) =0;
        virtual const LexUnitXId LexUnitIdToLexUnitXId(const LexUnitId &lex_unit_id)=0; 
      
        // special symbols dealing
        virtual LexUnit *LexUnitSilence() = 0;
        virtual LexUnit *LexUnitSentenceBegin() = 0;
        virtual LexUnit *LexUnitSentenceEnd() = 0;


        // traversing all lex units in the lexicon
        virtual LexUnitIterator*  GetLexUnitIterator() = 0;

        // loading and saving
        virtual IDEC_RETCODE LoadTxtFile(const char* file_name) = 0;
        virtual IDEC_RETCODE Load(const char*file_name) = 0;
        virtual IDEC_RETCODE Save(const char*file_name) = 0;
        virtual IDEC_RETCODE SaveTxtFile(const char* file_name) = 0;



        // build from a word list
        virtual void BuildFromWordList(const std::vector<std::string> &word_list,
                           PronunciationLexicon *base_lexicon, Lts *lts,
                           std::vector<std::string> &oov_list, const LtsOption &lts_opt) = 0;

        static unsigned char GetLexUnitType(const char *lex_unit_str);

        bool IsSpecialSymbol(const LexUnit *lex_unit) {
            return (lex_unit->lex_unit_type == LEX_UNIT_TYPE_SIL)
                || (lex_unit->lex_unit_type == LEX_UNIT_TYPE_SENTENCE_DELIMITER)
                || (lex_unit->lex_unit_type == LEX_UNIT_TYPE_UNKNOWN);
        }


        // max length of the unit in the lexicon
        size_t GetMaxLexicalUnitStrLen(){
            // lazy scanning
            if (max_lex_unit_str_len_ == 0) {
                for (LexUnitXId id =0; id < NumOfLexUnitX();id++) {
                    max_lex_unit_str_len_ = std::max(max_lex_unit_str_len_, static_cast<size_t>(strlen(LexUnitXId2Str(id))));
                }
            }
            return max_lex_unit_str_len_;
        }
        size_t max_lex_unit_str_len_;

    private:
        IDEC_DISALLOW_COPY_AND_ASSIGN(PronunciationLexicon);
    };
};

#endif
