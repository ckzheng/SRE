#include "lex/pronunciation_lexicon.h"

namespace idec{

    unsigned char PronunciationLexicon::GetLexUnitType(const char *lex_unit_str) {
        unsigned int length = (unsigned int)strlen(lex_unit_str);
        IDEC_ASSERT(length > 0);

        // beginning/end of sentence
        if ((strcmp(lex_unit_str, LEX_UNIT_BEG_SENTENCE) == 0) || (strcmp(lex_unit_str, LEX_UNIT_END_SENTENCE) == 0)) {
            return LEX_UNIT_TYPE_SENTENCE_DELIMITER;
        }
        // unknown
        else if (strcmp(lex_unit_str, LEX_UNIT_UNKNOWN) == 0) {
            return LEX_UNIT_TYPE_UNKNOWN;
        }
        // filler
        else if ((lex_unit_str[0] == '<') && (lex_unit_str[length - 1] == '>')) {
            return LEX_UNIT_TYPE_FILLER;
        }
        // standard
        else {
            return LEX_UNIT_TYPE_STANDARD;
        }
    }
}
