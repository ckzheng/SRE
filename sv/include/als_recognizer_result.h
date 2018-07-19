#ifndef __ALS_RECOGNIZER_RESULT_H__
#define __ALS_RECOGNIZER_RESULT_H__

#include "als_error.h"

// return a word
struct AlsRecognizedWordUnit {
    const char*   text;               // the text of word after text normalization
    const char*   lexical_form;       // the text of word without any text normalization
    const char*   pronounication;     // the phonetic spelling of a recognized word.
    const char*   grammar_name;       // the grammar name of this word unit
    float         confidence;         // confidence measure£¬0.00~100.00
    float         start_time;         // start time of the word within the audio, in term of second
    float         end_time;           // end time of the word within the audio, in term of second
};

// return a sentence level candidate 
struct AlsRecognizedPhrase {
    AlsRecognizedWordUnit           *words;         // all the words in this sentence
    int                             nWords;         // the number of words
    float                           confidence;     // the confidence of whole utterances
};


// return the nbest result
struct AlsRecognizerResult {
    AlsRecognizedPhrase *alternates;     // return the list of speech recognition candidate phrase
    int                  nAlternates;    //
    static ALSAPI_EXPORT void Release(AlsRecognizerResult * p_this);  // TODO (HAOZHI) in this case, the app have to link with the the static library,
                                        // which is not an interface.
};

#endif
